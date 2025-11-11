
# 盲人导航系统开发指南

## 目录
- [一、系统架构概述](#一系统架构概述)
- [二、核心模块详解](#二核心模块详解)
- [三、配置与部署](#三配置与部署)
- [四、API接口说明](#四api接口说明)
- [五、开发注意事项](#五开发注意事项)

## 一、系统架构概述

### 1.1 项目目标
搭建一个基于计算机视觉和语音交互的盲人导航系统，实现：
- 实时视频处理与物体检测
- 盲道追踪与红绿灯识别
- 语音识别与智能对话
- 物品查找与位置提示

### 1.2 技术栈
- **后端框架**: FastAPI + WebSocket
- **视觉处理**: OpenCV + YOLO11(备用)
- **AI服务**: 通义千问 (ASR + TTS + 对话)
- **语音转文字服务**:云端使用阿里云TTS/本地使用pyttsx3两种方案
- **前端**: HTML5 + WebSocket + MediaStream API
- **部署**: Docker + HTTPS

### 1.3 系统架构图
```
┌─────────────┐
│  移动端前端  │ (index.html + app.js)
│  - 摄像头   │
│  - 麦克风   │
│  - 陀螺仪   │
└──────┬──────┘
       │ WebSocket (4个端点)
       │
┌──────▼──────────────────────────┐
│      FastAPI 主服务 (main.py)    │
│  ┌────────────────────────────┐ │
│  │  /ws/camera  - 视频流      │ │
│  │  /ws_audio   - 音频流      │ │
│  │  /ws_ui      - UI消息推送  │ │
│  │  /ws         - IMU数据     │ │
│  └────────────────────────────┘ │
└──────┬──────────────────────────┘
       │
   ┌───┴────┬──────────┬──────────┐
   │        │          │          │
┌──▼───┐ ┌─▼────┐ ┌───▼────┐ ┌──▼────┐
│Video │ │Audio │ │ Qwen   │ │ IMU   │
│Proc. │ │Proc. │ │ API    │ │Proc.  │
└──────┘ └──────┘ └────────┘ └───────┘
```

## 二、核心模块详解

### 2.1 主程序 (main.py)

#### 2.1.1 现有架构分析
当前`main.py`已经实现了基础的WebSocket框架，包括：
- ✅ 4个WebSocket端点的定义
- ✅ ConnectionManager连接管理
- ✅ 基础的消息广播机制
- ✅ HTTPS服务器配置
#### 其他功能
**1. 导航模式控制**
前端通过按钮控制盲道导航和过马路模式，需要在WebSocket中接收控制指令。

**2. 音频播放功能**
需要播放`music/`文件夹下的预录音频文件。

**3. 物品查找功能**
通过语音识别物品名称，调用Qwen翻译，然后用OpenCV/YOLO检测。

#### 2.1.3 修改建议 - 最小化修改原则
- main.py只负责接收数据、调用处理器、转发结果
- 所有业务逻辑（导航控制、音频播放、指令识别等）都在processor中实现
- 保持main.py的简洁性和可维护性


### 2.2 视频处理模块 (modules/video_processor.py)
#### 2.2.1 设计原则
**video_processor负责所有视觉相关的功能：**
- 物体检测、盲道识别、红绿灯检测、斑马线检测
- 物品搜索和位置判断
- 音频文件管理和播放
- 导航状态管理
- 消息生成和格式化
**关键改进：**
1. **内置音频播放器**: VideoProcessor自己管理音频文件
2. **消息生成**: 统一在`_generate_messages`中处理所有消息
3. **返回格式**: 通过`messages`数组返回需要发送的消息
4. **状态管理**: 内部管理导航状态和搜索目标

### 2.3 音频处理模块 (modules/audio_processor.py)
#### 2.3.1 现有功能分析
当前`audio_processor.py`已实现：
- ✅ 语音活动检测（VAD）
- ✅ 语音识别（ASR）
- ✅ 大模型对话
- ✅ 语音合成（TTS）
#### 2.3.2 需要增强的功能
**AudioProcessor其他功能：**
1. **指令识别和分类**
   - 导航指令（盲道、过马路）
   - 物品搜索指令
   - 通用对话
2. **物品名称翻译**
   - 中文物品名 → 英文YOLO标签
   - 使用Qwen进行翻译
3. **导航状态管理**
   - 跟踪当前导航模式
   - 与VideoProcessor协同工作
**关键改进：**
1. **指令识别**: 自动识别导航、搜索、对话指令
2. **物品翻译**: 使用Qwen将中文翻译成英文YOLO标签
3. **状态管理**: 维护导航状态，与VideoProcessor协同
4. **消息格式**: 统一返回messages数组供main.py转发


## 三、本地配置与部署
### 3.1 环境变量配置
创建`.env`文件：
```bash
# 通义千问API密钥
DASHSCOPE_API_KEY=sk-your-api-key-here

# 服务器端口
SERVER_PORT=8001

# CUDA设备（如果有多个GPU）
CUDA_VISIBLE_DEVICES=0
```

### 3.2 依赖安装
```bash
# 创建虚拟环境(推荐python3.11及以下版本)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
依赖参考requirements.txt
```

### 3.3 HTTPS证书生成
运行generate_cert.py。仅是本地演示签名，实际上线需要向第三方申请 。

## 四、API接口说明

### 4.1 WebSocket端点

#### 4.1.1 视频流 `/ws/camera`
- **功能**: 接收前端摄像头的视频帧
- **数据格式**: 二进制JPEG图像
- **频率**: 30fps
- **返回**: 无（处理结果通过`/ws_ui`推送）

#### 4.1.2 音频流 `/ws_audio`
- **功能**: 接收麦克风音频并进行语音识别
- **数据格式**: PCM 16位，16kHz采样率
- **控制命令**:
  - `START`: 开始录音
  - `STOP`: 停止录音
  - `PROMPT:文本`: 发送文本指令

#### 4.1.3 UI消息 `/ws_ui`
- **功能**: 向前端推送各类消息
- **消息格式**:
由前端app.js中的handleUIMessage函数接收
  ```
  INIT:{json}      - 初始化数据
  PARTIAL:文本     - 部分语音识别结果
  FINAL:文本       - 最终识别结果/系统消息
  AUDIO:base64     - 音频数据
  ```
后端发送由main.py中的broadcast(self, message_type: str, data: dict)定义和发送


#### 4.1.4 IMU数据 `/ws`
- **功能**: 接收手机陀螺仪数据
- **数据格式**: JSON
  ```json
  {
    "ts": 1234567890,
    "accel": {"x": 0, "y": 0, "z": 9.8},
    "gyro": {"x": 0, "y": 0, "z": 0},
    "orientation": {"alpha": 0, "beta": 0, "gamma": 0}
  }
  ```

#### 4.1.5 控制指令 `/ws_control`
- **功能**: 接收前端按钮控制指令
- **数据格式**: JSON
  ```json
  {
    "action": "toggle_navigation",
    "enabled": true
  }
  ```

### 4.2 HTTP端点

#### 4.2.1 主页 `/`
- **方法**: GET
- **返回**: HTML页面（mobile/index.html）

#### 4.2.2 静态文件 `/mobile/*`
- **方法**: GET
- **功能**: 提供CSS、JS等静态资源



## 五、语音技术方案：本地录音+Qwen+ASR+TTS
流程图：

导航按钮部分：
用户点击"开始导航"按钮
    ↓
前端发送: "PROMPT:开始导航"
    ↓
main.py: handle_button_command()
    ↓
查找音频缓存: "切换到盲道导航"
    ↓
广播音频: "AUDIO:UklGRiQ..."
    ↓
前端接收并播放
```


语音对话流程：
用户说话
    ↓
前端持续发送PCM音频流
    ↓
ASR流式识别（paraformer-realtime-v2）
    ↓
实时广播部分结果: "PARTIAL:你好"
    ↓
识别完成，广播最终结果: "FINAL:你好世界"
    ↓
调用Qwen-VL多模态对话
  - 附加当前视频帧
  - 添加实时上下文
  - 优化prompt
    ↓
流式返回AI回复
  - "PARTIAL:[AI] 你好"
  - "PARTIAL:[AI] 你好，我是"
  - "FINAL:[AI] 你好，我是盲人导航助手"
    ↓
启动TTS子进程合成语音
    ↓
广播音频: "AUDIO:UklGRiQ..."
    ↓
前端接收并播放
```
---

### 5.1 本地语音实现基础导航响应

#### 5.1.1 功能说明
用户点击前端按钮时，立即播放预录的本地音频文件，实现零延迟响应。

#### 5.1.2 前后端通信接口

**前端 → 后端（通过 `/ws_audio`）**
```javascript
// 文本消息格式
"PROMPT:开始导航"      // 启动盲道导航
"PROMPT:停止导航"      // 停止盲道导航
"PROMPT:开始过马路"    // 启动过马路模式
"PROMPT:过马路结束"    // 停止过马路模式
```

**后端 → 前端（通过 `/ws_ui` 广播）**
```javascript
// 音频消息格式（完整WAV文件的Base64编码）
"AUDIO:UklGRiQAAABXQVZF..."

// 文本消息格式
"FINAL:{"text":"[系统] 切换到盲道导航"}"
```

#### 5.1.3 实现代码

```python
# main.py - 按钮指令处理
async def handle_button_command(command: str):
    """处理前端按钮指令，播放对应的本地音频"""
    
    # 指令与音频文件映射
    command_to_audio = {
        "开始导航": "切换到盲道导航",
        "停止导航": "导航已被取消",
        "开始过马路": "过马路模式已启动",
        "过马路结束": "过马路模式已取消"
    }
    
    audio_name = command_to_audio.get(command)
    
    if audio_name and audio_name in audio_processor.audio_cache:
        # 直接发送预加载的Base64音频
        audio_base64 = audio_processor.audio_cache[audio_name]
        await manager.broadcast('AUDIO', {'audio': audio_base64})
        print(f"[按钮指令] ✓ 已发送音频: {audio_name}.wav")

# modules/audio_processor.py - 音频预加载
def _load_audio_files(self):
    """启动时预加载所有音频文件到内存"""
    for audio_file in self.music_dir.glob("*.wav"):
        try:
            with open(audio_file, 'rb') as f:
                # 存储为Base64格式
                self.audio_cache[audio_file.stem] = base64.b64encode(f.read()).decode('utf-8')
            print(f"[音频处理器] 已加载音频: {audio_file.name}")
        except Exception as e:
            print(f"[音频处理器] 加载失败 {audio_file.name}: {e}")
```



---

### 5.2 ASR识别 + Qwen多模态对话方案

#### 5.2.1 功能说明
实时语音识别用户输入，结合当前视频帧和实时上下文，通过Qwen-VL进行多模态对话。

#### 5.2.2 前后端通信接口

**前端 → 后端（通过 `/ws_audio`）**
```javascript
// 控制信令
"START"  // 启动流式ASR识别
"STOP"   // 停止流式ASR识别

// 音频数据（二进制）
bytes: PCM 16位, 16kHz采样率, 持续流式发送
```

**后端 → 前端（通过 `/ws_ui` 广播）**
```javascript
// 部分识别结果（实时显示）
"PARTIAL:你好"
"PARTIAL:你好世界"

// 最终识别结果
"FINAL:你好世界"

// AI回复（流式显示）
"PARTIAL:[AI] 你好"
"PARTIAL:[AI] 你好，我是"
"FINAL:[AI] 你好，我是盲人导航助手"

// TTS合成音频
"AUDIO:UklGRiQAAABXQVZF..."  // Base64编码的完整WAV
```

#### 5.2.3 Qwen多模态通信接口详解

**API调用格式：**

```python
# modules/audio_processor.py

async def _chat_with_qwen(self, user_text: str):
    """使用Qwen-VL进行多模态对话"""
    
    # 1. 构建多模态输入
    content_list = []
    
    # 添加图像（如果有当前视频帧）
    if self.current_frame is not None:
        img_b64 = base64.b64encode(self.current_frame).decode('ascii')
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })
    
    # 添加文本
    content_list.append({
        "type": "text",
        "text": user_text
    })
    
    # 2. 获取实时上下文
    realtime_context = self._get_realtime_context()
    # 返回格式：
    # "当前时间：2025年1月8日 星期三 15:30，下午，天气：晴 15°C"
    
    # 3. 构建系统提示
    system_prompt = (
        "你是盲人导航助手。回答规则："
        "1. 不超过30字"
        "2. 只陈述事实"
        "3. 禁用情感词：很高兴、当然、非常等"
        "4. 直接回答核心问题"
    )
    if realtime_context:
        system_prompt += f"\n\n当前环境：{realtime_context}"
    
    # 4. 调用Qwen-VL API（OpenAI兼容接口）
    completion = self.oai_client.chat.completions.create(
        model="qwen-vl-max",  # 支持图像+文本多模态
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content_list  # 多模态内容列表
            }
        ],
        stream=True  # 流式响应
    )
    
    # 5. 处理流式响应
    text_buffer = []
    for chunk in completion:
        if not chunk.choices:
            continue
        
        delta = chunk.choices[0].delta
        if delta.content:
            text_buffer.append(delta.content)
            full_text = "".join(text_buffer)
            # 实时广播部分结果
            await self.manager.broadcast('PARTIAL', f'[AI] {full_text}')
    
    # 6. 广播最终结果
    final_text = "".join(text_buffer)
    await self.manager.broadcast('FINAL', f'[AI] {final_text}')
    
    # 7. 调用TTS合成语音
    await self._synthesize_speech(final_text)
```

**关键特性：**

1. **多模态输入**
   - 支持图像+文本组合
   - 图像使用Base64编码的Data URL格式
   - 自动附加当前视频帧作为视觉上下文

2. **实时上下文**
   ```python
   def _get_realtime_context(self) -> str:
       """获取实时上下文信息"""
       now = datetime.now()
       
       # 时间信息
       weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
       time_info = f"当前时间：{now.year}年{now.month}月{now.day}日 {weekdays[now.weekday()]} {now.hour}:{now.minute:02d}"
       
       # 时段判断
       hour = now.hour
       if 5 <= hour < 8:
           period = "清晨"
       elif 8 <= hour < 12:
           period = "上午"
       # ... 其他时段
       
       # 天气信息（使用wttr.in API）
       response = requests.get("https://wttr.in/?format=%C+%t", timeout=2)
       weather_info = f"，天气：{response.text.strip()}"
       
       return f"{time_info}，{period}{weather_info}"
   ```

3. **优化的Prompt**
   - 限制回复长度（30字以内）
   - 禁止主观情感表达
   - 强调事实陈述
   - 直接回答核心问题

4. **流式响应**
   - 边生成边显示
   - 提升用户体验
   - 降低感知延迟

---

### 5.3 pyttsx3本地TTS方案
对应文件：audio_processor(pyttsx).py+tts_synthesizer(back).py
注意！！部署到阿里云的版本因为操作系统原因使用的是阿里云的阿里云TTS

#### 5.3.1 问题背景

**遇到的问题：**
1. **pyttsx3状态冲突**：在asyncio环境中长时间运行后失效
2. **只有第一次能工作**：重启后第一次TTS正常，后续全部失败

**根本原因：**
- pyttsx3引擎与asyncio事件循环存在深层状态冲突
- 引擎初始化后的状态会累积，导致后续调用失败
- 线程锁、重新初始化等方案都无法彻底解决

#### 5.3.2 最终解决方案：子进程隔离

**核心思想：**
将pyttsx3完全隔离到独立的子进程中运行，每次合成都启动新进程，避免状态累积。

**架构图：**
```
用户对话 → Qwen生成文本 → _synthesize_speech()
    ↓
启动子进程: python tts_synthesizer.py "文本"
    ↓
子进程: pyttsx3生成WAV → Base64编码 → stdout
    ↓
主进程: 读取stdout → 解码Base64 → WAV数据
    ↓
_broadcast_tts_audio() → 发送给移动端
```

#### 5.3.3 实现代码
独立TTS脚本：modules/tts_synthesizer.py

#### 5.3.4 方案优势

1. **完全隔离**
   - 每次合成都在新进程中运行
   - 无状态累积问题
   - 进程退出自动释放所有资源

2. **异步非阻塞**
   - 使用`asyncio.create_subprocess_exec`
   - 不阻塞主事件循环
   - 支持并发处理

3. **稳定可靠**
   - 避免pyttsx3与asyncio的深层冲突
   - 连续多次调用都能正常工作
   - 子进程崩溃不影响主服务

#### 5.3.5 音频格式统一

**所有音频都使用相同格式：**

```python
# 格式：完整WAV文件的Base64编码
# 前端接收格式："AUDIO:UklGRiQAAABXQVZF..."

# 1. 预录音频（启动时加载）
with open(audio_file, 'rb') as f:
    audio_cache[name] = base64.b64encode(f.read()).decode('utf-8')

# 2. TTS音频（子进程生成）
wav_data = await _tts_to_wav(text)
wav_b64 = base64.b64encode(wav_data).decode('utf-8')

# 3. 统一广播
await manager.broadcast('AUDIO', {'audio': wav_b64})
```
---



### 5.4 关键技术要点总结

#### 5.4.1 ASR流式识别

```python
# 使用Dashscope paraformer-realtime-v2
recognition = Recognition(
    model='paraformer-realtime-v2',
    format='pcm',
    sample_rate=16000,
    callback=ASRCallback(processor, event_loop),
    api_key=api_key
)

# 启动长连接
recognition.start()

# 持续发送音频帧
recognition.send_audio_frame(audio_data)

# 回调接收结果
def on_event(self, result: RecognitionResult):
    sentence = result.get_sentence()
    if sentence and 'text' in sentence:
        text = sentence['text']
        end_time = sentence.get('end_time')
        if end_time is not None and end_time > 0:
            # 最终结果
            asyncio.run_coroutine_threadsafe(
                self.processor._handle_final_text(text),
                self.loop
            )
        else:
            # 部分结果
            asyncio.run_coroutine_threadsafe(
                self.processor._handle_partial_text(text),
                self.loop
            )
```

#### 5.4.2 Qwen多模态调用

```python
# 使用OpenAI兼容接口
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 多模态内容
content_list = [
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
    },
    {
        "type": "text",
        "text": user_text
    }
]

# 流式调用
completion = client.chat.completions.create(
    model="qwen-vl-max",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ],
    stream=True
)

# 处理流式响应
for chunk in completion:
    delta = chunk.choices[0].delta
    if delta.content:
        await manager.broadcast('PARTIAL', f'[AI] {delta.content}')
```

#### 5.4.3 pyttsx3子进程隔离

```python
# 主进程调用
proc = await asyncio.create_subprocess_exec(
    sys.executable,
    'modules/tts_synthesizer.py',
    text,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
)

stdout, stderr = await proc.communicate()

# 子进程输出Base64音频到stdout
if stdout:
    wav_b64 = stdout.decode('utf-8').strip()
    wav_data = base64.b64decode(wav_b64)
    await manager.broadcast('AUDIO', {'audio': wav_b64})
```

#### 5.4.4 音频格式统一

```python
# 所有音频都是完整WAV文件的Base64编码
# 广播格式："AUDIO:UklGRiQAAABXQVZF..."

# 预录音频
audio_cache[name] = base64.b64encode(wav_file).decode('utf-8')

# TTS音频
wav_b64 = base64.b64encode(wav_data).decode('utf-8')

# 统一广播
await manager.broadcast('AUDIO', {'audio': wav_b64})
```


## 六、OpenCV盲道、过马路检测功能、搜索物品功能

### 6.1 功能概述

系统提供两种导航模式：
1. **盲道导航模式**：检测黄色盲道，提供方向引导
2. **过马路模式**：检测斑马线和红绿灯，提供过马路指引

### 6.2 盲道检测实现

#### 6.2.1 检测原理

使用OpenCV颜色检测和形态学处理识别黄色盲道。检测流程：
1. 转换到HSV色彩空间
2. 提取黄色区域（盲道颜色）
3. 形态学操作去噪
4. 轮廓分析确定位置和方向

#### 6.2.2 语音提示逻辑

根据检测结果播放相应的语音提示：
- **未检测到**：播放"未识别盲道"
- **检测到**：先播放"已识别盲道"，然后根据位置提示方向
  - 盲道在左侧 → "向右"
  - 盲道在右侧 → "向左"
  - 盲道居中 → "已对中"

#### 6.2.3 播报控制

为避免频繁播报，实现了时间间隔控制：
- 同类型语音最小间隔：3秒
- 实时打印检测状态到控制台
- 状态变化时立即播报

#### 6.2.4 可用语音文件
music/下面
```

### 6.3 过马路模式实现

#### 6.3.1 斑马线检测

使用Canny边缘检测和霍夫直线变换识别斑马线的平行条纹：

**检测参数（已优化）：**
- Canny边缘检测阈值：(60, 180)
- 霍夫变换阈值：60
- 最小线段数量：6条
- 角度容差：18度
- 最小线段长度：50像素

**距离判断：**
- **远处**：斑马线在画面上方
- **正在靠近**：斑马线在画面中部
- **到达**：斑马线在画面下方（可以过马路）

#### 6.3.2 红绿灯检测

使用颜色检测识别红绿灯状态：

**检测方法：**
1. 转换到HSV色彩空间
2. 分别检测红、黄、绿三种颜色
3. 查找圆形轮廓（红绿灯特征）
4. 判断圆形度（> 0.7）
5. 返回检测到的颜色

**颜色范围：**
- 红色：HSV [0, 100, 100] - [10, 255, 255]
- 黄色：HSV [20, 100, 100] - [30, 255, 255]
- 绿色：HSV [40, 100, 100] - [80, 255, 255]

#### 11.3.3 语音提示逻辑

**斑马线提示：**
- 未发现 → "未发现斑马线"
- 远处发现 → "远处发现斑马线"
- 正在靠近 → "正在靠近斑马线"
- 到达 → "斑马线到了可以过马路"

**红绿灯提示：**
- 检测到红灯 → "红灯"
- 检测到黄灯 → "黄灯"
- 检测到绿灯 → "绿灯"

#### 6.3.4 可用语音文件
music/下面
```

### 6.4 导航状态管理
#### 6.4.1 导航状态控制
在[`VideoProcessor`](modules/video_processor.py:1)中维护导航状态：
#### 6.4.2 按钮控制
前端按钮通过[`handle_button_command()`](main.py:445)函数控制状态：

### 6.5 物品搜索功能
工作流程：
用户说"找水杯"
系统提示用户对准水杯，捕获中心区域作为模板
后续帧中使用ORB特征匹配搜索该模板
根据匹配结果播报位置："在画面左侧"/"在画面中间"/"在画面右侧"（本地语音文件）
加入其他功能指令后或者特定关键词如“结束”，自动退出搜索

OpenCV特征匹配方法：
cv2.ORB_create(): 创建ORB特征检测器
orb.detectAndCompute(): 检测关键点并计算描述符
cv2.BFMatcher(): 暴力匹配器
matcher.knnMatch(): k-近邻特征匹配
比率测试筛选优质匹配点




## 七、AI模型测试系统
参考/test文件夹下内容

## 八、Docker部署配置
1)用AI打包docker文件
2）用ftp工具将本地开发文件打包.zip上传到云服务器目录下（注意：.venv无需打包、cert.pemt和key.pem如有真实签名也不需要），用ssh命令行进入目录后，upzip filename.zip解压后再运行docker文件
3）ssh命令行运行docker文件（运行之前确认docker文件下载包连接国内镜像服务器）
1. 构建镜像
docker-compose build
2. 启动服务
docker-compose up -d
启动以后即可以查看xxx.xxx.xxx.xxx:8001是否可以访问
3. 查看日志
docker-compose logs -f
4. 停止服务
docker-compose down
5. 查看容器状态
docker ps
6. 进入容器调试
docker exec -it openglass-prod /bin/bash
4）运行generate_cert.py，生成签名文件
5）打开服务器对应端口，如本产品为8001

更新文件到服务器后需重启docker:
cd /opt/安装目录
docker-compose down
docker-compose build
docker-compose up -d
docker-compose logs -f

