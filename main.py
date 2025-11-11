import asyncio #引入异步线程，asyncio.run() 来启动事件循环， await 来协调它们的执行顺序

import json
import os #配置服务和端口如(os.getenv("SERVER_PORT", 8001))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
"""
通过创建 app = FastAPI()，你就实例化了一个 Web 应用，可以用它来定义 API 路由、处理 HTTP 请求等
fastapi的作用：1)统一控制应用的入口路由.2)集成WebSocket实时通信(/ws端点)
WebSocket 是一种允许服务器和客户端之间进行双向实时通信的协议，非常适合视频流
WebSocketDisconnect通过捕获这个异常来优雅地处理客户端的断开事件。
"""

from fastapi.staticfiles import StaticFiles
"""
StaticFiles类用于让 FastAPI 应用能够提供静态文件服务
app.mount("/mobile", StaticFiles(directory="mobile"), name="mobile")，它将 mobile 文件夹设置为一个静态资源目录。
当浏览器请求 /mobile/style.css 时，FastAPI 会自动返回 mobile 文件夹下的 style.css 文件。
"""

from fastapi.responses import HTMLResponse
""""
返回html文件，在 @app.get("/") 路由中，它被用来直接读取并返回 mobile/index.html 文件的内容
StaticFiles和HTMLResponse区别：HTMLResponse 精确地返回 index.html 文件的内容。StaticFiles 是用来将整个文件夹下的文件如html、css、js都挂载到浏览器。
过程：
1.用户访问网站根目录 /。
2.HTMLResponse 精确地返回 index.html 文件的内容。
3.浏览器解析 index.html，发现里面有 <link href="/mobile/style.css"> 和 <script src="/mobile/app.js">。
4.浏览器接着向服务器发起对 /mobile/style.css 和 /mobile/app.js 的请求。
5.StaticFiles 捕获这些请求，自动从 mobile 文件夹中找到并返回对应的文件。
"""

import cv2
import numpy as np
from dotenv import load_dotenv
#用于项目根目录下的 .env 文件中读取并加载环境变量
import base64
#base64 通常用于在客户端和服务器之间传输二进制数据（如图片或音频）


# 加载环境变量
load_dotenv()

#引用FastAPI功能，用于加载mobile网页
app = FastAPI()

# FastAPI挂载静态文件,保存到app变量
app.mount("/mobile", StaticFiles(directory="mobile"), name="mobile")

# 存储活跃的WebSocket连接
active_connections = []

#一、管理连接状态、向前端广播消息broadcast
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        #定义活跃连接，由WebSocket通信实现

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        """
        HTTP转换为WebSocket过程；
        HTTP通过载入app.js，发起websocket请求，websocket.accept()接受并确认WebSocket连接并返回给前端。
        后端定义端点如(@app.websocket("/ws/camera"))后，前端再接两者连接起来：
        const wsUrl = `${baseUrl}/ws/camera`;
        this.ws.camera = new WebSocket(wsUrl);
        
        本系统使用多个WebSocket连接：
        1.摄像头数据流 (/ws/camera) - 传输视频帧，通常后端不发送消息回传
        2.音频数据流 (/ws_audio) - 前端传输语音数据、控制语音指令（START、STOP、PROMPT），后端传给前端流式对话的指令如('OK:STARTED')
        3.UI更新 (/ws_ui) - 传输界面更新信息，前后端都传消息
        4.IMU数据 (/ws) - 传输陀螺仪和加速度计数据，通常后端不发送消息回传
        """

        # async def定义异步函数
        # 检查当前websocket连接是否已存在，避免重复添加
        if websocket not in self.active_connections:
            self.active_connections.append(websocket)
            print(f"[连接管理] 客户端已连接，当前连接数: {len(self.active_connections)}")
        else:
            print(f"[连接管理] 警告：连接已存在，跳过添加")
    
    def disconnect(self, websocket: WebSocket):
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                print(f"[连接管理] 客户端已断开，当前连接数: {len(self.active_connections)}")
            else:
                print(f"[连接管理] 警告：连接不在列表中")
        except ValueError as e:
            print(f"[连接管理] 断开连接时出错: {e}")
    
    #备用：向指定单个连接发送消息，用于特定状态使用
    async def send_message(self, websocket: WebSocket, message_type: str, data: dict):
        """发送带前缀的文本消息"""
        msg = f"{message_type}:{json.dumps(data, ensure_ascii=False)}"
        await websocket.send_text(msg)
    
    #定义broadcast()，用于向活跃连接广播消息：包括本地语音和JSON信息
    async def broadcast(self, message_type: str, data: dict):
        """
        broadcast发送广播消息：
        处理功能信息后，向用户UI界面推送各类信息。接收者：前端应用app.js中的handleUIMessage函数过程如下：
        1)broadcast()会检查每个connection的活跃状态
        2)只有活跃的connection才会收到消息、不活跃的connection会被标记为待移除(通过for connection in self.active_connections:判断)
        3)message、frame_data等接收前端通过特定websocket发送消息(接收方式后文说明)，再由对应后端处理发送给当前活跃的前端连接
        4)特别处理 AUDIO 类型的消息以匹配前端期望的格式。"AUDIO:UklGRiQ...=" 格式
        5)通过 websocket.send_text(msg) 发给消息给所有活跃的 WebSocket 连接(只有调用 manager.connect(websocket) 的/ws_ui和部分/ws_audio)，再由前端handleUIMessage函数处理
        """

        """
        前端app.js对于接收到msg的处理方式统一由handleUIMessage()处理：
        this.ws.ui.onmessage = (event) => {
            this.handleUIMessage(event.data);  // 处理所有收到的消息
        };
        handleUIMessage(message) {
            if (message.startsWith('INIT:')) { /* 处理初始化 */ }
            else if (message.startsWith('PARTIAL:')) { /* 处理部分识别 */ }
            else if (message.startsWith('FINAL:')) { /* 处理最终结果 */ }
            else if (message.startsWith('AUDIO:')) { /* 处理音频播放 */ }
        }
        不同接口调用这个函数情况不同：
        /ws_ui：有完整的 handleUIMessage() 函数，根据消息前缀（INIT:, PARTIAL:, FINAL:, AUDIO:）分类处理
        /ws_audio：只处理特定的控制响应（如 'OK:STARTED'）
        /ws/camera：没有 onmessage 处理器，即使收到消息也会被忽略        
        """

        msg = ""
        """
        msg用来保存要发送的消息：前缀+数据内容(以字符串通信传递)，如"AUDIO:base64数据"或"FINAL:{"text":"消息内容"}"
        INIT:{json}      - 初始化数据
        PARTIAL:文本     - 部分语音识别结果
        FINAL:文本       - 最终识别结果/系统消息
        AUDIO:base64     - 音频数据
        """

        if message_type == 'AUDIO' and 'audio' in data:
            msg = f"AUDIO:{data['audio']}"
        #广播本地音频文件消息：AUDIO:base64数据
        #message_type在视频和音频处理中，调用broadcast()函数时会被写

        else:
            msg = f"{message_type}:{json.dumps(data, ensure_ascii=False)}"     
        if not msg:
            return
        #非音频消息(INIT, PARTIAL, FINAL)，保持原有的 JSON 格式
        """
        由于WebSocket用字符串传递，所以需要将消息字符串化:
        dumps: Python对象 → JSON字符串（序列化）
        loads: JSON字符串 → Python对象（反序列化）
        前端JavaScript使用JSON.parse()进行反序列化
        """
        
        #创建需要移除的失效连接列表#
        connections_to_remove = []
        for connection in self.active_connections:
            try:
                if connection.client_state.name != "DISCONNECTED":
                    await connection.send_text(msg)
                else:
                    connections_to_remove.append(connection)
                    print(f"[广播] 发现已断开的连接，将移除")
            except Exception as e:
                print(f"[广播] 发送消息失败: {str(e)}")
                connections_to_remove.append(connection)
        
        # 移除失效的连接
        for connection in connections_to_remove:
            try:
                self.active_connections.remove(connection)
                print(f"[广播] 已移除失效连接，当前连接数: {len(self.active_connections)}")
            except ValueError:
                pass

manager = ConnectionManager()


##用app = FastAPI()加载index网页,@装饰器相当于赋予函数一个新功能
@app.get("/")
async def get():
    return HTMLResponse(content=open("mobile/index.html", encoding="utf-8").read())

#二、创建WebSocket多端点，处理前端传来的不同数据类型、管理ws连接状态
"""
注意：在此websocket、下方的handle_xxx_xxx、由handle实际调用的外部文件audio/video中的功能如opencv/yolo、qwen、pyttx3实现，三者是分开的
"""

"""
前端发送消息+后端接收的方式：
1.WebSocket 协议本身就支持两种数据帧类型：文本帧（Text Frame） - 用于传输文本数据、二进制帧（Binary Frame） - 用于传输二进制数据

2.app.js实际发送过程：
创建WebSocket 连接
this.ws = {
    camera: null,   // 视频帧专用连接
    audio: null,     // 音频数据专用连接  
    ui: null,        // UI更新专用连接
    imu: null        // IMU数据专用连接
};
this.ws.camera.send(blob);  // 发送二进制JPEG数据
this.ws.audio.send(pcm16);  // 发送二进制PCM数据
this.ws.audio.send(`PROMPT:${command}`); // 文本帧
this.ws.imu.send(JSON.stringify(imuPacket));  // 文本帧

3.后端每个 WebSocket 端点独立运行，各自监听对应类型的数据：
frame_data = await websocket.receive_bytes()# 视频端点 - 只等待二进制
message = await websocket.receive()# 音频端点 - 等待任意类型
data = await websocket.receive_text()# IMU端点 - 只等待文本
"""

# ========== WebSocket端点1: 视频流处理 ==========
@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """
    这里只处理视频帧数据
    1.前端发送: JPEG格式的二进制blob数据
    2.通过OpenCV解码成图片
    3.解码后的图片再交由handle_video_frame处理
    """
    await websocket.accept()
    #websocket.accept()告诉前端创建一个("/ws_audio")连接
    print("[摄像头WS] 客户端已连接")
    
    try:
        while True:
            # 接收二进制视频帧数据
            frame_data = await websocket.receive_bytes()
            
            # 解码JPEG图像
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 在这里处理视频帧
                # 例如：大模型识别、YOLO检测、盲道识别等
                await handle_video_frame(websocket, frame)
            
    except WebSocketDisconnect:
        print("[摄像头WS] 客户端已断开")
    except Exception as e:
        print(f"[摄像头WS] 错误: {e}")

# ========== WebSocket端点2: 音频处理 ==========
@app.websocket("/ws_audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    print("[音频WS] 客户端已连接")
    """
    这里只处理音频接收、流式调用ASR，实际调用功能在audio_processor.py中，以及在handle_button_command()中处理简单的本地语音
    """
    # 获取或利用全局空函数handle_audio_data()创建AudioProcessor类的实例
    from modules.audio_processor import AudioProcessor
    if not hasattr(handle_audio_data, 'processor'):
        #因为handle_audio_data没有processor这个属性，所以创建一个带属性的实例，也可以被其他函数引用
        handle_audio_data.processor = AudioProcessor(
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        #创建属性，是为了引入第三方库的方法，用于处理音视频对象
        #在audio_processor.py中有AudioProcessor()类，传递了API_KEY调用qwen(通过os.getenv从环境文件中传递)


        # 设置广播管理器
        handle_audio_data.processor.set_manager(manager)
        print("[音频WS] AudioProcessor已初始化并设置manager")
    
    audio_processor = handle_audio_data.processor
    
    try:
        while True:
            message = await websocket.receive()
            #以下用于判断接收的消息类型，如果是命令“text”则执行命令，如果是“bytes”则直接送给ASR识别

            # 处理命令文本指令
            if 'text' in message and message.get('text'):
                text_data = message['text']
                print(f"[音频WS] 收到文本消息: {text_data}")

                if text_data == 'START':
                    # 启动ASR流式识别
                    await audio_processor.start_asr_stream()
                    await websocket.send_text('OK:STARTED')
                    print("[音频WS] ASR流式识别已启动")
                
                elif text_data == 'STOP':
                    # 停止ASR流式识别
                    await audio_processor.stop_asr_stream()
                    await websocket.send_text('OK:STOPPED')
                    print("[音频WS] ASR流式识别已停止")
                
                elif text_data.startswith('PROMPT:'):
                    command = text_data[7:]
                    # 前端传来的按钮指令，将由下文的handle_button_command处理，指令不应阻塞音频流(异步asyncio创建)
                    asyncio.create_task(handle_button_command(command))
                
                else:
                    print(f"[音频WS] 收到未知文本消息: {text_data}")

            # 处理音频数据
            elif 'bytes' in message and message.get('bytes'):
                audio_data = message['bytes']
                # 直接发送给ASR（不等待结果）
                await audio_processor.process_audio_stream(audio_data, websocket)
    
    except WebSocketDisconnect:
        # 断开时停止ASR
        print("[音频WS] 客户端断开连接，正在清理...")
        try:
            await audio_processor.stop_asr_stream()
        except Exception as e:
            print(f"[音频WS] 停止ASR时出错: {e}")
        print("[音频WS] 客户端已断开")
    except Exception as e:
        print(f"[音频WS] 错误: {e}")
        import traceback
        traceback.print_exc()
        # 确保清理ASR连接
        try:
            await audio_processor.stop_asr_stream()
        except:
            pass

# ========== WebSocket端点3: UI消息推送 ==========
@app.websocket("/ws_ui")
async def ui_websocket(websocket: WebSocket):
    """
    向前端推送UI更新消息
    发送格式:
    - INIT:{json} - 初始化数据
    - PARTIAL:文本 - 部分语音识别结果
    - FINAL:文本 - 最终语音识别结果
    - AUDIO:base64 - 音频数据
    """
    # 创建ws_ui连接
    await manager.connect(websocket)
    print("[UI WS] 客户端已连接并加入广播列表")
    
    # 发送初始化数据
    init_data = {
        "partial": "等待语音输入...",
        "finals": []  # 历史消息列表
    }
    await websocket.send_text(f"INIT:{json.dumps(init_data)}")
    
    try:
        while True:
            # UI端点主要用于服务器向客户端推送消息
            # 这里是保持ws_ui连接，等待其他模块触发消息发送
            # 处理错误情况
            await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        print("[UI WS] 客户端已断开")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[UI WS] 错误: {e}")
        manager.disconnect(websocket)

# ========== WebSocket端点4: IMU陀螺仪数据 ==========
"""
暂时不用
@app.websocket("/ws")
async def imu_websocket(websocket: WebSocket):

    {
        "ts": 时间戳,
        "accel": {"x": 0, "y": 0, "z": 0},
        "gyro": {"x": 0, "y": 0, "z": 0},
        "orientation": {"alpha": 0, "beta": 0, "gamma": 0}
    }

    await websocket.accept()
    print("[IMU WS] 客户端已连接")
    
    try:
        while True:
            # 接收JSON格式的IMU数据
            data = await websocket.receive_text()
            imu_data = json.loads(data)
            
            # 在这里处理IMU数据
            # 例如：姿态估计、运动检测等
            await handle_imu_data(websocket, imu_data)
    
    except WebSocketDisconnect:
        print("[IMU WS] 客户端已断开")
    except Exception as e:
        print(f"[IMU WS] 错误: {e}")
"""
        
# ==========三、数据处理功能模块 ==========
#1.处理视频帧功能#
async def handle_video_frame(websocket: WebSocket, frame):
    """处理视频帧"""
    # 例如：大模型识别、YOLO物体检测、盲道识别、红绿灯检测
    try:
        from modules.video_processor import VideoProcessor
        
        # 使用单例模式
        if not hasattr(handle_video_frame, 'processor'):
            handle_video_frame.processor = VideoProcessor()
        
        video_processor = handle_video_frame.processor
        
        # 同步当前帧到AudioProcessor（用于多模态对话）
        if hasattr(handle_audio_data, 'processor'):
            import cv2
            # 转换格式：将原始frame编码为JPEG字节流
            _, jpeg_frame = cv2.imencode('.jpg', frame)
            #用_，表示占位符，并不需要返回的这个值，只需要取jpeg_frame作为处理后的实际图片文件
            handle_audio_data.processor.set_current_frame(jpeg_frame.tobytes())
            #原始frame是numpy数组，通过imencode()和tobytes()转换成体积小的字节流
            """注意：字节流存储在 AudioProcessor 的 current_frame 属性中，在audio_processor.py 被 base64 编码后用于向qwen传递图片信息"""

            # 同步搜索目标从AudioProcessor到VideoProcessor
            audio_search_target = handle_audio_data.processor.navigation_state.get('search_target')
            if audio_search_target != video_processor.search_target:
                if audio_search_target:
                    video_processor.set_search_target(audio_search_target)
                    print(f"[视频处理] 已设置搜索目标: {audio_search_target}")
                else:
                    video_processor.clear_search_target()
                    print(f"[视频处理] 已清除搜索目标")
        
        # 处理视频帧（所有逻辑在VideoProcessor中），通过OpenCV识别盲道等，返回识别结果
        result = await video_processor.process_frame(frame)
        
        # 转发结果到前端
        if result.get('messages'):
            for msg in result['messages']:
                await manager.broadcast(msg['type'], msg['data'])
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        error_message = f"视频处理错误: {str(e)}"
        print(error_message)
        await manager.broadcast("FINAL", {"text": f"[系统] {error_message}"})
        return {"status": "error", "message": error_message}

#2.处理音频功能#
async def handle_audio_data(websocket: WebSocket, audio_data: bytes):
    # 此函数保留用于兼容性，实际处理已移至audio_websocket
    # 作为全局单例容器存储 AudioProcessor 实例，实现全局单例模式：
    #1.audio_websocket() - 初始化并使用 AudioProcessor
    #2.handle_video_frame() - 访问实例用于多模态对话
    #3.handle_button_command() - 访问实例用于播放音频
    pass

#3.处理IMU陀螺仪功能#暂时不用
async def handle_imu_data(websocket: WebSocket, imu_data: dict):
    """处理IMU陀螺仪数据，暂时可不用"""
    # 这里将在后续模块中实现
    # 例如：姿态分析、运动检测
    try:
        from modules.imu_processor import IMUProcessor
        imu_processor = IMUProcessor()
        result = await imu_processor.process_imu_data(imu_data)
        
        if result:
            if result.get('fall_detected'):
                await manager.broadcast("FINAL", {
                    "text": "检测到跌倒!",
                    "timestamp": imu_data.get('ts', 0)
                })
            
            await manager.broadcast("FINAL", {
                "text": "IMU数据更新",
                "data": result
            })
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        error_message = f"IMU数据处理错误: {str(e)}"
        print(error_message)
        await manager.broadcast("FINAL", {
            "text": error_message
        })
        return {"status": "error", "message": error_message}

#4.点击前端按钮后播放对应语音#
async def handle_button_command(command: str):
    """
    处理前端按钮指令：
    1. 播放对应的本地音频
    2. 设置VideoProcessor的导航/过马路模式状态
    """
    print(f"[按钮指令] 收到: {command}")
    
    # 1. 定义指令与音频文件的映射关系
    command_to_audio = {
        "开始导航": "切换到盲道导航",
        "停止导航": "导航已被取消",
        "开始过马路": "过马路模式已启动",
        "过马路结束": "过马路模式已取消"
    }

    # 2. 查找对应的音频文件名
    audio_name = command_to_audio.get(command)

    # 3. 设置VideoProcessor的模式状态
    try:
        # 获取VideoProcessor实例
        if hasattr(handle_video_frame, 'processor'):
            video_processor = handle_video_frame.processor
            
            if command == "开始导航":
                video_processor.navigation_enabled = True
                video_processor.traffic_mode = False
                # 重置状态
                video_processor.last_blind_path_state = None
                print(f"[按钮指令] ✓ 已启动盲道导航模式")
                
            elif command == "停止导航":
                video_processor.navigation_enabled = False
                video_processor.last_blind_path_state = None
                print(f"[按钮指令] ✓ 已停止盲道导航模式")
                
            elif command == "开始过马路":
                video_processor.traffic_mode = True
                video_processor.navigation_enabled = False
                # 重置状态
                video_processor.last_traffic_light_color = None
                video_processor.last_zebra_distance = None
                print(f"[按钮指令] ✓ 已启动过马路模式")
                
            elif command == "过马路结束":
                video_processor.traffic_mode = False
                video_processor.last_traffic_light_color = None
                video_processor.last_zebra_distance = None
                print(f"[按钮指令] ✓ 已停止过马路模式")
        else:
            print(f"[按钮指令] 警告: VideoProcessor尚未初始化，将在首次视频帧到达时初始化")
    
    except Exception as e:
        print(f"[按钮指令] 设置VideoProcessor状态时出错: {e}")
        import traceback
        traceback.print_exc()

    # 4. 如果找到了匹配的音频，则发送播放指令
    if audio_name:
        from modules.audio_processor import AudioProcessor
        try:
            # 确保 AudioProcessor 已初始化（使用全局单例）
            if not hasattr(handle_button_command, 'processor'):
                print(f"[按钮指令] 初始化 AudioProcessor...")
                handle_button_command.processor = AudioProcessor(
                    api_key=os.getenv("DASHSCOPE_API_KEY")
                )
            
            audio_processor = handle_button_command.processor

            # 从 AudioProcessor 的缓存中获取音频数据
            if audio_name in audio_processor.audio_cache:
                audio_base64 = audio_processor.audio_cache[audio_name]
                
                print(f"[按钮指令] 准备发送音频: {audio_name}")
                print(f"[按钮指令] Base64长度: {len(audio_base64)}")
                
                # 构建并广播 AUDIO 类型的消息
                await manager.broadcast('AUDIO', {'audio': audio_base64})
                print(f"[按钮指令] ✓ 已发送音频播放指令: {audio_name}.wav")
                
                # 同时发送一条文本消息，用于在界面上显示状态
                #await manager.broadcast('FINAL', {'text': f"{command}已执行"})

            else:
                print(f"[按钮指令] ✗ 错误: 在音频缓存中未找到 '{audio_name}'")
                print(f"[按钮指令] 可用的音频文件: {list(audio_processor.audio_cache.keys())}")

        except Exception as e:
            print(f"[按钮指令] ✗ 发送音频时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[按钮指令] ✗ 未找到与 '{command}' 匹配的音频")
        print(f"[按钮指令] 可用的指令: {list(command_to_audio.keys())}")



#==========三、配置运行后的访问接口、地址功能模块 ==========
if __name__ == "__main__":
    import uvicorn
    import ssl #为实现https访问，载入ssl功能

    port = int(os.getenv("SERVER_PORT", 8001))
    # 检查证书文件是否存在
    cert_file = "cert.pem"
    key_file = "key.pem"
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        # 创建SSL上下文
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        
        print("启动HTTPS服务器...")
        print("访问地址: https://localhost:{port}")
        print("移动端访问: https://[您的IP地址]:{port}/mobile")
        
        #uvicorn用来接收https请求并运行app = FastAPI()加载网页，同时接收ssl文件
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,

            ssl_certfile=cert_file,
            ssl_keyfile=key_file
        )
"""
当Uvicorn配置了SSL证书(ssl_certfile+ssl_keyfile)后：
端口绑定：Uvicorn在指定端口（如8001）上只监听HTTPS流量
协议切换：所有连接必须通过SSL/TLS握手
HTTP请求被拒绝：未加密的HTTP请求无法通过SSL验证
这就是为什么启用HTTPS后，HTTP地址无法访问的原因 - 服务器完全切换到了HTTPS模式。
"""
