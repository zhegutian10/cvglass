import asyncio
import threading
from pathlib import Path
import base64
import re
import audioop
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from openai import OpenAI
import dashscope
import numpy as np
import os
import pyttsx3
import io
import wave
from datetime import datetime
import requests
import sys
import time


class ASRCallback(RecognitionCallback):
    """流式ASR回调 - 符合qwen指南要求"""
    def __init__(self, processor, loop):
        self.processor = processor
        self.loop = loop  # 保存事件循环引用
        self.partial_text = ""

    def on_open(self):
        print("[ASR] 连接已建立")

    def on_event(self, result: RecognitionResult):
        """实时接收识别事件"""
        sentence = result.get_sentence()
        if sentence and 'text' in sentence:
            text = sentence['text']
            # 判断是否为最终结果
            # end_time存在且大于0表示最终结果
            end_time = sentence.get('end_time')
            if end_time is not None and end_time > 0:
                # 最终结果
                print(f"[ASR] 最终识别: {text}")
                # 使用线程安全的方式调度异步任务
                asyncio.run_coroutine_threadsafe(
                    self.processor._handle_final_text(text),
                    self.loop
                )
            else:
                # 部分结果
                self.partial_text = text
                asyncio.run_coroutine_threadsafe(
                    self.processor._handle_partial_text(text),
                    self.loop
                )

    def on_complete(self):
        print("[ASR] 识别完成")

    def on_error(self, result: RecognitionResult):
        print(f"[ASR] 错误: {result.get_message()}")

    def on_close(self):
        print("[ASR] 连接关闭")


class AudioProcessor:
    """
    音频处理器 - 符合qwen指南的完整实现
    负责ASR识别和Qwen-Omni多模态对话
    """
    
    def __init__(self, api_key, music_dir="./music"):
        """初始化音频处理器"""
        self.api_key = api_key
        dashscope.api_key = api_key
        
        # 初始化OpenAI客户端用于Qwen文本对话
        self.oai_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # ASR识别器（流式长连接）
        self.recognition = None
        self.asr_callback = None
        self.event_loop = None  # 保存事件循环引用
        
        # 对话历史
        self.conversation_history = [
            {
                'role': 'system',
                'content': (
                    '你是一个盲人导航助手。回答要求：'
                    '1. 简洁客观，不超过30字'
                    '2. 只陈述事实，不添加情感词汇'
                    '3. 避免使用"很高兴"、"当然"等主观表达'
                    '4. 直接回答问题核心'
                )
            }
        ]
        
        # 导航状态
        self.navigation_state = {
            'blind_path_enabled': False,
            'traffic_mode_enabled': False,
            'search_target': None
        }
        
        # 音频缓存
        self.music_dir = Path(music_dir)
        self.audio_cache = {}
        self._load_audio_files()
        
        # 当前视频帧（用于多模态对话）
        self.current_frame = None
        
        # 广播管理器引用
        self.manager = None
        
        # 当前播放任务
        self.current_audio_task = None
        
        print("[音频处理器] 初始化完成")
    
    def _get_realtime_context(self) -> str:
        """
        获取实时上下文信息（时间、天气等）
        参考qwen多模态使用指南第271-278行
        """
        try:
            now = datetime.now()
            
            # 1. 时间信息
            weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
            time_info = f"当前时间：{now.year}年{now.month}月{now.day}日 {weekdays[now.weekday()]} {now.hour}:{now.minute:02d}"
            
            # 2. 时段信息
            hour = now.hour
            if 5 <= hour < 8:
                period = "清晨"
            elif 8 <= hour < 12:
                period = "上午"
            elif 12 <= hour < 14:
                period = "中午"
            elif 14 <= hour < 18:
                period = "下午"
            elif 18 <= hour < 22:
                period = "晚上"
            else:
                period = "深夜"
            
            # 3. 天气信息（使用wttr.in API）
            weather_info = ""
            try:
                # 获取简单的天气信息
                response = requests.get(
                    "https://wttr.in/?format=%C+%t",
                    timeout=2
                )
                if response.status_code == 200:
                    weather_info = f"，天气：{response.text.strip()}"
            except:
                pass  # 天气获取失败不影响主流程
            
            context = f"{time_info}，{period}{weather_info}"
            return context
            
        except Exception as e:
            print(f"[上下文] 获取失败: {e}")
            return ""
    
    
    def _load_audio_files(self):
        """预加载音频文件"""
        if not self.music_dir.exists():
            print(f"[音频处理器] 警告: 音频目录不存在 {self.music_dir}")
            return
        
        for audio_file in self.music_dir.glob("*.wav"):
            try:
                with open(audio_file, 'rb') as f:
                    self.audio_cache[audio_file.stem] = base64.b64encode(f.read()).decode('utf-8')
                print(f"[音频处理器] 已加载音频: {audio_file.name}")
            except Exception as e:
                print(f"[音频处理器] 加载音频失败 {audio_file.name}: {e}")
    
    def set_manager(self, manager):
        """设置广播管理器"""
        self.manager = manager
    
    def set_current_frame(self, frame):
        """设置当前视频帧用于多模态对话"""
        self.current_frame = frame
    
    async def start_asr_stream(self):
        """启动流式ASR - 符合qwen指南，支持重连"""
        # 如果已有连接，先停止旧连接
        if self.recognition:
            print("[ASR] 检测到旧连接，先停止...")
            try:
                self.recognition.stop()
            except Exception as e:
                print(f"[ASR] 停止旧连接时出错: {e}")
            finally:
                self.recognition = None
                self.asr_callback = None
        
        try:
            # 获取当前事件循环
            self.event_loop = asyncio.get_running_loop()
            
            # 创建回调时传入事件循环
            self.asr_callback = ASRCallback(self, self.event_loop)
            self.recognition = Recognition(
                model='paraformer-realtime-v2',
                format='pcm',
                sample_rate=16000,
                callback=self.asr_callback,
                api_key=self.api_key
            )
            
            self.recognition.start()
            print("[ASR] 流式识别已启动")
        except Exception as e:
            print(f"[ASR] 启动失败: {e}")
            self.recognition = None
            self.asr_callback = None
            raise
    
    async def stop_asr_stream(self):
        """停止流式ASR"""
        if self.recognition:
            try:
                self.recognition.stop()
                print("[ASR] 流式识别已停止")
            except Exception as e:
                print(f"[ASR] 停止时出错: {e}")
            finally:
                self.recognition = None
                self.asr_callback = None
        else:
            print("[ASR] 没有活跃的识别连接")
    
    async def process_audio_stream(self, audio_data: bytes, websocket=None):
        """
        处理音频流 - 符合qwen指南的流式处理
        直接将音频帧发送给ASR，不做缓冲
        """
        try:
            # 确保ASR已启动
            if not self.recognition:
                await self.start_asr_stream()
            
            # 直接发送音频帧给ASR（流式处理）
            try:
                self.recognition.send_audio_frame(audio_data)
            except Exception as send_error:
                # 如果发送失败（连接已断开），尝试重启ASR
                print(f"[音频处理器] 发送音频帧失败: {send_error}")
                print("[音频处理器] 尝试重启ASR连接...")
                await self.start_asr_stream()
                # 重试发送
                self.recognition.send_audio_frame(audio_data)
            
            return {'status': 'streaming'}
            
        except Exception as e:
            print(f"[音频处理器] 错误: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    async def _handle_partial_text(self, text: str):
        """处理部分识别结果 - 实时显示"""
        if self.manager:
            # 直接发送文本，不包装在dict中
            await self.manager.broadcast('PARTIAL', text)
    
    async def _handle_final_text(self, text: str):
        """处理最终识别结果 - 触发AI处理"""
        if not text or not text.strip():
            return
        
        print(f"[音频处理器] 处理最终文本: {text}")
        
        # 广播用户输入（直接发送文本）
        if self.manager:
            await self.manager.broadcast('FINAL', text)
        
        # 检测打断热词
        if any(keyword in text for keyword in ["停止", "停下", "别说了"]):
            await self._interrupt_current_audio()
            if self.manager:
                await self.manager.broadcast('FINAL', '已停止')
            return
        
        # 处理指令
        await self._process_command(text)
    
    async def _interrupt_current_audio(self):
        """中断当前音频播放"""
        if self.current_audio_task and not self.current_audio_task.done():
            self.current_audio_task.cancel()
            print("[音频处理器] 已中断当前音频")
    
    async def _process_command(self, text: str):
        """
        处理指令 - 核心逻辑
        
        分类：
        1. 导航指令（盲道、过马路）
        2. 物品搜索指令
        3. 通用对话
        
        改进：任何新指令都会自动清除搜索状态
        """
        # 先中断当前播放
        await self._interrupt_current_audio()
        
        # 检测是否需要退出搜索模式
        is_search_command = "找" in text or "搜索" in text or "在哪" in text
        is_stop_search = ("停止" in text or "取消" in text or "结束" in text) and ("搜索" in text or "找" in text)
        
        # 如果不是搜索指令，且当前有搜索目标，则自动清除
        if not is_search_command and self.navigation_state.get('search_target'):
            print(f"[音频处理器] 检测到新指令，自动退出搜索模式")
            self.navigation_state['search_target'] = None
            if self.manager:
                await self.manager.broadcast('FINAL', {'text': '[系统] 已退出搜索模式'})
        
        # 明确的停止搜索指令
        if is_stop_search:
            self.navigation_state['search_target'] = None
            if self.manager:
                await self.manager.broadcast('FINAL', {'text': '[系统] 已停止搜索'})
            return
        
        # 1. 导航指令检测
        if "盲道" in text or "导航" in text:
            if "开始" in text or "启动" in text:
                self.navigation_state['blind_path_enabled'] = True
                await self._play_preset_audio('切换到盲道导航')
            elif "停止" in text or "取消" in text:
                self.navigation_state['blind_path_enabled'] = False
                await self._play_preset_audio('导航已被取消')
            return
        
        # 2. 过马路模式
        if "过马路" in text or "红绿灯" in text:
            if "开始" in text or "启动" in text:
                self.navigation_state['traffic_mode_enabled'] = True
                await self._play_preset_audio('过马路模式已启动')
            elif "停止" in text or "取消" in text:
                self.navigation_state['traffic_mode_enabled'] = False
                await self._play_preset_audio('过马路模式已取消')
            return
        
        # 3. 物品搜索指令
        if is_search_command:
            item_name = self._extract_item_name(text)
            if item_name:
                # 翻译成英文
                english_label = await self._translate_to_english(item_name)
                if english_label:
                    self.navigation_state['search_target'] = english_label
                    if self.manager:
                        await self.manager.broadcast('FINAL',
                            {'text': f'[系统] 开始搜索: {item_name} ({english_label})'})
                return
        
        # 4. 通用对话 - 使用Qwen文本+pyttsx3语音
        await self._chat_with_qwen(text)
    
    def _extract_item_name(self, text: str) -> str:
        """提取物品名称"""
        patterns = [
            r"找(?:一下)?(.+?)(?:。|$)",
            r"搜索(.+?)(?:。|$)",
            r"(.+?)在哪",
            r"帮我找(.+?)(?:。|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None
    
    async def _play_preset_audio(self, audio_name: str):
        """播放预录音频"""
        if audio_name in self.audio_cache:
            if self.manager:
                await self.manager.broadcast('AUDIO', {'audio': self.audio_cache[audio_name]})
                await self.manager.broadcast('FINAL', {'text': f'[系统] {audio_name}'})
            print(f"[音频处理器] 播放预录音频: {audio_name}")
        else:
            print(f"[音频处理器] 未找到音频: {audio_name}")
    
    async def _translate_to_english(self, chinese_name: str) -> str:
        """
        将中文物品名翻译成英文YOLO标签
        使用Qwen进行翻译
        """
        try:
            from dashscope import Generation
            
            prompt = (
                "You are a label normalizer. Convert the given Chinese object "
                "description into a short, lowercase English YOLO/vision class name "
                "(1~3 words). If multiple are given, return the single most likely one. "
                "Output ONLY the label, no punctuation."
            )
            
            messages = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': chinese_name}
            ]
            
            response = Generation.call(
                api_key=self.api_key,
                model="qwen-plus",
                messages=messages,
                result_format="message"
            )
            
            if response.status_code == 200:
                english_label = response.output.choices[0].message.content.strip().lower()
                print(f"[翻译] {chinese_name} -> {english_label}")
                return english_label
            else:
                print(f"[翻译] 失败: {response.message}")
                return chinese_name
        
        except Exception as e:
            print(f"[翻译] 异常: {e}")
            return chinese_name
    
    async def _chat_with_qwen(self, user_text: str):
        """
        使用Qwen进行文本对话 + pyttsx3语音合成
        1. 只获取文本回复（不使用Qwen-Omni）
        2. 使用pyttsx3本地合成语音
        3. 优化prompt使回复简洁客观
        """
        try:
            print(f"[Qwen] 开始对话: {user_text}")
            
            # 1. 组装多模态内容
            content_list = []
            
            # 添加图像（如果有当前帧）
            if self.current_frame is not None:
                try:
                    img_b64 = base64.b64encode(self.current_frame).decode('ascii')
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                    print("[Qwen] 已添加图像输入")
                except Exception as e:
                    print(f"[Qwen] 图像编码失败: {e}")
            
            # 添加文本
            content_list.append({"type": "text", "text": user_text})
            
            # 2. 获取实时上下文
            realtime_context = self._get_realtime_context()
            
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
            
            # 4. 调用Qwen VL（只获取文本，不要音频）
            completion = self.oai_client.chat.completions.create(
                model="qwen-vl-max",  # 使用VL模型支持图像
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_list}
                ],
                stream=True
            )
            
            # 5. 处理流式文本响应
            text_buffer = []
            
            for chunk in completion:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # 处理文本增量
                if delta.content:
                    text_buffer.append(delta.content)
                    full_text = "".join(text_buffer)
                    if self.manager:
                        await self.manager.broadcast('PARTIAL', f'[AI] {full_text}')
            
            # 6. 获取完整文本
            final_text = "".join(text_buffer)
            if final_text and self.manager:
                await self.manager.broadcast('FINAL', f'[AI] {final_text}')
            
            print(f"[Qwen] 文本回复: {final_text}")
            
            # 5. 使用pyttsx3合成语音
            if final_text:
                await self._synthesize_speech(final_text)
            
        except Exception as e:
            print(f"[Qwen] 错误: {e}")
            import traceback
            traceback.print_exc()
            
            if self.manager:
                await self.manager.broadcast('FINAL', f'对话出错: {str(e)}')
    
    async def _synthesize_speech(self, text: str):
        """
        使用阿里云TTS服务合成语音并广播
        替代pyttsx3，解决Docker环境兼容性问题
        """
        try:
            print(f"[TTS] 使用阿里云TTS合成: {text}")
            
            # 使用阿里云TTS服务
            audio_data = await self._dashscope_tts(text)
            
            if audio_data:
                await self._broadcast_tts_audio(audio_data)
                print(f"[TTS] ✓ 语音合成并广播完成，大小: {len(audio_data)} 字节")
            else:
                print(f"[TTS] ✗ 语音合成失败，未收到音频数据")
            
        except Exception as e:
            print(f"[TTS] ✗ 合成任务失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def _dashscope_tts(self, text: str) -> bytes:
        """
        使用阿里云DashScope TTS服务将文本转换为WAV音频
        支持中文，音质更好，无需本地音频设备
        """
        try:
            from dashscope.audio.tts import SpeechSynthesizer
            
            # 调用阿里云TTS服务
            result = SpeechSynthesizer.call(
                model='sambert-zhiqi-v1',  # 中文女声
                text=text,
                sample_rate=16000,
                format='wav',
                api_key=self.api_key
            )
            
            # 获取音频数据
            if result.get_audio_data():
                audio_data = result.get_audio_data()
                print(f"[TTS] 阿里云TTS合成成功，大小: {len(audio_data)} 字节")
                return audio_data
            else:
                print(f"[TTS] 阿里云TTS返回空数据")
                return b''
            
        except Exception as e:
            print(f"[TTS] 阿里云TTS调用失败: {e}")
            import traceback
            traceback.print_exc()
            return b''
    
    async def _broadcast_tts_audio(self, wav_data: bytes):
        """
        广播TTS生成的音频
        直接发送完整WAV文件（与预录音频保持一致）
        """
        try:
            # 检查Manager状态
            if not self.manager:
                print("[TTS] 错误: Manager未设置，无法广播音频")
                return
            
            print(f"[TTS] 准备广播音频，大小: {len(wav_data)} 字节")
            print(f"[TTS] Manager状态: 活跃连接数 = {len(self.manager.active_connections)}")
            
            # 编码为Base64
            wav_b64 = base64.b64encode(wav_data).decode('utf-8')
            print(f"[TTS] Base64编码完成，长度: {len(wav_b64)}")
            
            # 直接发送完整WAV文件（与预录音频格式一致）
            await self.manager.broadcast('AUDIO', {'audio': wav_b64})
            print(f"[TTS] ✓ 音频广播完成")
            
        except Exception as e:
            print(f"[TTS] ✗ 音频广播失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def _broadcast_pcm16_realtime(self, pcm16: bytes):
        """
        按20ms节拍分发PCM数据
        """
        loop = asyncio.get_event_loop()
        next_tick = loop.time()
        off = 0
        
        # 按20ms分片
        BYTES_PER_20MS = 320  # 8kHz * 2字节 * 20ms / 1000
        
        while off < len(pcm16):
            # 取一个20ms的片段
            take = min(BYTES_PER_20MS, len(pcm16) - off)
            piece = pcm16[off:off + take]
            
            # 编码为Base64并广播
            piece_b64 = base64.b64encode(piece).decode('utf-8')
            if self.manager:
                await self.manager.broadcast('AUDIO', {'audio': piece_b64})
            
            # 精确的20ms节拍控制
            next_tick += 0.020
            now = loop.time()
            if now < next_tick:
                await asyncio.sleep(next_tick - now)
            else:
                next_tick = now  # 如果延迟了，重置时间基准
            
            off += take
    
    async def process_button_command(self, command: str):
        """处理前端按钮发送的指令"""
        messages = []   
        print(f"[按钮指令] 收到: {command}")
        
        # 中断当前播放
        await self._interrupt_current_audio()
        
        # 1. 盲道导航指令
        if "导航" in command or "盲道" in command:
            if "开始" in command or "启动" in command:
                self.navigation_state['blind_path_enabled'] = True
                audio_name = '切换到盲道导航'
                status_text = "盲道导航已启动"
            elif "停止" in command or "取消" in command or "结束" in command:
                self.navigation_state['blind_path_enabled'] = False
                audio_name = '导航已被取消'
                status_text = "盲道导航已停止"
            else:
                audio_name = None
                status_text = None
            
            if audio_name and audio_name in self.audio_cache:
                messages.append({
                    'type': 'AUDIO',
                    'data': {'audio': self.audio_cache[audio_name]}
                })
                print(f"[按钮指令] 播放音频: {audio_name}")
            
            if status_text:
                messages.append({
                    'type': 'FINAL',
                    'data': {'text': status_text}
                })
            
            return {'messages': messages, 'navigation_state': self.navigation_state}
        
        # 2. 过马路模式指令
        if "过马路" in command or "红绿灯" in command:
            if "开始" in command or "启动" in command:
                self.navigation_state['traffic_mode_enabled'] = True
                audio_name = '过马路模式已启动'
                status_text = "过马路模式已启动"
            elif "停止" in command or "取消" in command or "结束" in command:
                self.navigation_state['traffic_mode_enabled'] = False
                audio_name = '过马路模式已取消'
                status_text = "过马路模式已停止"
            else:
                audio_name = None
                status_text = None
            
            if audio_name and audio_name in self.audio_cache:
                messages.append({
                    'type': 'AUDIO',
                    'data': {'audio': self.audio_cache[audio_name]}
                })
                print(f"[按钮指令] 播放音频: {audio_name}")
            
            if status_text:
                messages.append({
                    'type': 'FINAL',
                    'data': {'text': status_text}
                })
            
            return {'messages': messages, 'navigation_state': self.navigation_state}
        
        return {'messages': messages}