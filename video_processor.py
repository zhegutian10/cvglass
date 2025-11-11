import base64
from pathlib import Path
import cv2
import numpy as np

class VideoProcessor:
    """
    视频处理器 - 轻量版
    负责传统视觉算法处理（盲道、红绿灯、斑马线）
    物体识别交由Qwen-VL处理
    """
    
    def __init__(self, music_dir="./music"):
        """初始化视频处理器"""
        # 导航状态
        self.navigation_enabled = False
        self.traffic_mode = False
        
        # 状态跟踪（用于避免重复播放）
        self.last_blind_path_state = None  # 上次盲道状态
        self.last_traffic_light_color = None  # 上次红绿灯颜色
        self.last_zebra_distance = None  # 上次斑ma线距离

        # 物品搜索状态
        self.search_target = None
        self.search_template = None  # 用于存储模板图像
        self.orb = None  # ORB特征检测器
        self.bf = None  # BFMatcher匹配器
        self.template_kp = None  # 模板的关键点
        self.template_des = None  # 模板的描述符
        self.search_frame_skip = 0  # 搜索帧跳过计数器
        self.search_skip_interval = 3  # 每3帧处理一次搜索（降低性能影响）
        
        # 语音播报时间控制（避免频繁播报）
        import time
        self.last_audio_time = {
            'blind_detected': 0,      # 已识别盲道
            'blind_not_detected': 0,  # 未识别盲道
            'blind_direction': 0,     # 方向提示
            'traffic_light': 0,       # 红绿灯
            'zebra_crossing': 0,      # 斑马线
            'zebra_not_found': 0,     # 未发现斑马线
            'search_found': 0,        # 找到物体
            'search_not_found': 0     # 未找到物体
        }
        self.audio_interval = 3.0  # 同类语音最小间隔（秒）
        
        # 帧计数器（用于定期打印状态）
        self.frame_count = 0
        
        # 音频播放器（内置）
        self.music_dir = Path(music_dir)
        self.audio_cache = {}
        self._load_audio_files()
        
        print(f"[视频处理器] 初始化完成 - 使用OpenCV传统视觉算法")
        print(f"[视频处理器] OpenCV版本: {cv2.__version__}")
        print(f"[视频处理器] CUDA支持: {cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False}")
    
    def _load_audio_files(self):
        """预加载所有音频文件"""
        if not self.music_dir.exists():
            print(f"[视频处理器] 警告: 音频目录不存在 {self.music_dir}")
            return
        
        # 加载.wav和.WAV文件
        for pattern in ["*.wav", "*.WAV"]:
            for audio_file in self.music_dir.glob(pattern):
                try:
                    with open(audio_file, 'rb') as f:
                        self.audio_cache[audio_file.stem] = base64.b64encode(f.read()).decode('utf-8')
                    print(f"[视频处理器] 已加载音频: {audio_file.name}")
                except Exception as e:
                    print(f"[视频处理器] 加载音频失败 {audio_file.name}: {e}")
    
    def set_search_target(self, target_label: str):
        """设置搜索目标"""
        self.search_target = target_label
        self.search_template = None  # 重置模板
        self.template_kp = None
        self.template_des = None
        self.search_frame_skip = 0
        print(f"[视频处理器] 设置搜索目标: {target_label}")
    
    def clear_search_target(self):
        """清除搜索目标"""
        self.search_target = None
        self.search_template = None
        self.template_kp = None
        self.template_des = None
        self.search_frame_skip = 0
        print(f"[视频处理器] 已清除搜索目标")
    
    async def process_frame(self, frame):
        """
        处理单帧视频 - 轻量版（仅传统视觉算法）
        
        物体识别已移至Qwen-VL，此处只处理：
        - 盲道检测（OpenCV颜色检测）
        - 红绿灯检测（OpenCV颜色检测）
        - 斑马线检测（OpenCV边缘检测）
        
        Returns:
            dict: {
                'blind_path': {...},
                'traffic_light': {...},
                'zebra_crossing': {...},
                'messages': [{'type': 'FINAL'/'AUDIO', 'data': {...}}]
            }
        """
        self.frame_count += 1
        
        result = {
            'blind_path': None,
            'traffic_light': None,
            'zebra_crossing': None,
            'search_result': None,
            'messages': []  # 用于返回给main.py转发的消息
        }
        
        try:
            # 1. 物体搜索（如果有搜索目标）- 使用帧跳过降低性能影响
            if self.search_target:
                self.search_frame_skip += 1
                
                # 每N帧处理一次搜索，减少性能开销
                if self.search_frame_skip >= self.search_skip_interval:
                    self.search_frame_skip = 0
                    result['search_result'] = await self._detect_search_target(frame)
                    
                    # 实时打印搜索状态
                    if result['search_result']['detected']:
                        if result['search_result'].get('is_template'):
                            print(f"[物体搜索] 已捕获目标模板")
                        else:
                            print(f"[物体搜索] 找到目标 - 位置:{result['search_result']['position']}, "
                                  f"置信度:{result['search_result']['confidence']:.2f}, "
                                  f"匹配点:{result['search_result'].get('match_count', 0)}")
                    else:
                        print(f"[物体搜索] 未找到目标")
                else:
                    # 跳过的帧，使用上一次的结果（如果有）
                    result['search_result'] = {'detected': False, 'position': 'none', 'confidence': 0, 'skipped': True}
            
            # 2. 盲道检测（仅在导航模式下）
            if self.navigation_enabled:
                result['blind_path'] = await self._detect_blind_path(frame)
                result['navigation_prompt'] = self._generate_navigation_prompt(
                    result['blind_path']
                )
                
                # 实时打印识别状态
                if result['blind_path']['detected']:
                    print(f"[盲道检测] 已识别盲道 - 方向:{result['blind_path']['direction']}, "
                          f"距离:{result['blind_path']['distance']}, "
                          f"面积:{result['blind_path']['area']:.0f}")
                else:
                    print(f"[盲道检测] 未识别盲道")
            
            # 3. 红绿灯和斑马线检测（仅在过马路模式下）
            if self.traffic_mode:
                result['traffic_light'] = await self._detect_traffic_light(frame)
                result['zebra_crossing'] = await self._detect_zebra_crossing(frame)
                
                # 实时打印识别状态
                if result['traffic_light']['detected']:
                    print(f"[红绿灯检测] 检测到{result['traffic_light']['color']}灯, "
                          f"置信度:{result['traffic_light']['confidence']:.2f}")
                else:
                    print(f"[红绿灯检测] 未检测到红绿灯")
                
                if result['zebra_crossing']['detected']:
                    distance_map = {'far': '远处', 'medium': '正在靠近', 'close': '到达'}
                    distance_text = distance_map.get(result['zebra_crossing']['distance'], '未知')
                    print(f"[斑马线检测] {distance_text}发现斑马线 - 位置:{result['zebra_crossing']['position']}, "
                          f"线条数:{result['zebra_crossing']['line_count']}")
                else:
                    print(f"[斑马线检测] 未发现斑马线")
            
            # 4. 生成消息（用于前端显示和音频播放）
            self._generate_messages(result)
            
            return result
            
        except Exception as e:
            print(f"[视频处理器] 处理帧错误: {e}")
            import traceback
            traceback.print_exc()
            return result
    
    async def _detect_blind_path(self, frame):
        """
        检测盲道
        
        使用颜色检测和边缘检测识别黄色盲道
        
        Returns:
            dict: {detected, direction, distance}
        """
        try:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 黄色范围（盲道通常是黄色）
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # 创建掩码
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 形态学操作去噪
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'detected': False, 'direction': 'none', 'distance': 'far'}
            
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 面积阈值
            if area < 1000:
                return {'detected': False, 'direction': 'none', 'distance': 'far'}
            
            # 计算轮廓中心
            M = cv2.moments(largest_contour)
            if M['m00'] == 0:
                return {'detected': False, 'direction': 'none', 'distance': 'far'}
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # 判断方向
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            
            if cx < frame_width / 3:
                direction = 'left'
            elif cx < frame_width * 2 / 3:
                direction = 'center'
            else:
                direction = 'right'
            
            # 判断距离（根据y坐标和面积）
            if cy > frame_height * 0.7 or area > frame_width * frame_height * 0.3:
                distance = 'close'
            elif cy > frame_height * 0.4:
                distance = 'medium'
            else:
                distance = 'far'
            
            return {
                'detected': True,
                'direction': direction,
                'distance': distance,
                'center': (cx, cy),
                'area': area
            }
            
        except Exception as e:
            print(f"[盲道检测] 错误: {e}")
            return {'detected': False, 'direction': 'none', 'distance': 'far'}
    
    async def _detect_traffic_light(self, frame):
        """
        检测红绿灯
        
        使用颜色检测识别红绿灯状态
        
        Returns:
            dict: {detected, color, position}
        """
        try:
            # 转换到HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 定义颜色范围
            colors = {
                'red': ([0, 100, 100], [10, 255, 255]),
                'yellow': ([20, 100, 100], [30, 255, 255]),
                'green': ([40, 100, 100], [80, 255, 255])
            }
            
            detected_color = None
            max_area = 0
            
            for color_name, (lower, upper) in colors.items():
                lower = np.array(lower)
                upper = np.array(upper)
                
                mask = cv2.inRange(hsv, lower, upper)
                
                # 查找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # 红绿灯通常是圆形且面积适中
                    if 100 < area < 5000 and area > max_area:
                        # 检查圆形度
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.7:  # 接近圆形
                                max_area = area
                                detected_color = color_name
            
            if detected_color:
                return {
                    'detected': True,
                    'color': detected_color,
                    'confidence': min(max_area / 1000, 1.0)
                }
            else:
                return {'detected': False, 'color': 'none', 'confidence': 0}
                
        except Exception as e:
            print(f"[红绿灯检测] 错误: {e}")
            return {'detected': False, 'color': 'none', 'confidence': 0}
    
    async def _detect_zebra_crossing(self, frame):
        """
        检测斑马线 - 平衡版（平衡准确率和召回率）
        
        使用边缘检测和直线检测识别斑马线的平行条纹
        适度的验证条件，既减少误报又保证检测率
        
        Returns:
            dict: {detected, distance, position}
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测（适中的阈值）
        edges = cv2.Canny(blurred, 60, 180)
        
        # 霍夫直线检测（适中的阈值）
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=60,  # 适中阈值
            minLineLength=60,  # 适中最小线长
            maxLineGap=12  # 适中间隙容忍度
        )
        
        # 至少需要6条线才认为可能是斑马线
        if lines is None or len(lines) < 6:
            return {'detected': False, 'distance': 'far', 'position': 'none'}
        
        # 分析直线：斑马线应该有多条平行的横向直线
        horizontal_lines = []
        line_lengths = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线角度
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # 接近水平的直线（角度小于18度）
            if angle < 18 or angle > 162:
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # 只保留足够长的线（至少50像素）
                if line_length > 50:
                    horizontal_lines.append(line[0])
                    line_lengths.append(line_length)
        
        # 至少需要6条符合条件的水平线
        if len(horizontal_lines) < 6:
            return {'detected': False, 'distance': 'far', 'position': 'none'}
        
        # 额外验证：检查线条的间距规律性（放宽条件）
        y_coords = [line[1] for line in horizontal_lines]
        y_coords_sorted = sorted(y_coords)
        
        # 计算相邻线条的间距
        if len(y_coords_sorted) >= 3:
            gaps = [y_coords_sorted[i+1] - y_coords_sorted[i] for i in range(len(y_coords_sorted)-1)]
            # 斑马线的间距应该比较规律，计算间距的标准差
            if len(gaps) > 0:
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                # 放宽间距规律性要求（标准差可以达到平均值的120%）
                if std_gap > avg_gap * 1.2:
                    return {'detected': False, 'distance': 'far', 'position': 'none'}
        
        # 通过所有验证，认为检测到斑马线
        # 计算斑马线位置（取所有线的平均y坐标）
        avg_y = np.mean([line[1] for line in horizontal_lines])
        frame_height = frame.shape[0]
        
        # 判断距离（根据y坐标）
        if avg_y > frame_height * 0.7:
            distance = 'close'
        elif avg_y > frame_height * 0.4:
            distance = 'medium'
        else:
            distance = 'far'
        
        # 判断位置
        avg_x = np.mean([(line[0] + line[2]) / 2 for line in horizontal_lines])
        frame_width = frame.shape[1]
        
        if avg_x < frame_width / 3:
            position = 'left'
        elif avg_x < frame_width * 2 / 3:
            position = 'center'
        else:
            position = 'right'
        
        return {
            'detected': True,
            'distance': distance,
            'position': position,
            'line_count': len(horizontal_lines)
        }
    
    async def _detect_search_target(self, frame):
        """
        使用OpenCV特征匹配搜索目标物体
        
        工作流程：
        1. 如果没有模板，将当前帧中心区域作为模板
        2. 使用ORB特征检测和匹配
        3. 返回物体位置信息
        
        Returns:
            dict: {detected, position, confidence}
        """
        try:
            # 如果没有搜索目标，直接返回
            if not self.search_target:
                return {'detected': False, 'position': 'none', 'confidence': 0}
            
            # 初始化ORB检测器（如果还没有）
            if self.orb is None:
                self.orb = cv2.ORB_create(nfeatures=500)
                self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                print(f"[物体搜索] ORB检测器已初始化")
            
            # 如果没有模板，使用当前帧中心区域作为模板
            if self.search_template is None:
                h, w = frame.shape[:2]
                # 取中心40%区域作为模板
                y1, y2 = int(h * 0.3), int(h * 0.7)
                x1, x2 = int(w * 0.3), int(w * 0.7)
                self.search_template = frame[y1:y2, x1:x2].copy()
                
                # 提取模板特征
                gray_template = cv2.cvtColor(self.search_template, cv2.COLOR_BGR2GRAY)
                self.template_kp, self.template_des = self.orb.detectAndCompute(gray_template, None)
                
                if self.template_des is None or len(self.template_kp) < 10:
                    print(f"[物体搜索] 警告：模板特征点不足，请对准目标物体")
                    self.search_template = None
                    return {'detected': False, 'position': 'none', 'confidence': 0}
                
                print(f"[物体搜索] 已捕获模板，特征点数：{len(self.template_kp)}")
                return {'detected': True, 'position': 'center', 'confidence': 1.0, 'is_template': True}
            
            # 在当前帧中搜索模板
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_kp, frame_des = self.orb.detectAndCompute(gray_frame, None)
            
            if frame_des is None or len(frame_kp) < 10:
                return {'detected': False, 'position': 'none', 'confidence': 0}
            
            # 特征匹配
            matches = self.bf.knnMatch(self.template_des, frame_des, k=2)
            
            # 应用比率测试筛选好的匹配
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # 需要至少10个好的匹配点
            if len(good_matches) < 10:
                return {'detected': False, 'position': 'none', 'confidence': 0}
            
            # 计算匹配物体的中心位置
            matched_points = [frame_kp[m.trainIdx].pt for m in good_matches]
            avg_x = np.mean([pt[0] for pt in matched_points])
            avg_y = np.mean([pt[1] for pt in matched_points])
            
            # 判断位置
            frame_width = frame.shape[1]
            if avg_x < frame_width / 3:
                position = 'left'
            elif avg_x < frame_width * 2 / 3:
                position = 'center'
            else:
                position = 'right'
            
            # 计算置信度（基于匹配点数量）
            confidence = min(len(good_matches) / 50, 1.0)
            
            return {
                'detected': True,
                'position': position,
                'confidence': confidence,
                'match_count': len(good_matches),
                'center': (int(avg_x), int(avg_y))
            }
            
        except Exception as e:
            print(f"[物体搜索] 错误: {e}")
            import traceback
            traceback.print_exc()
            return {'detected': False, 'position': 'none', 'confidence': 0}
    
    def _generate_navigation_prompt(self, blind_path_info):
        """
        生成导航提示 - 增强版
        
        添加障碍物检测和更详细的提示
        """
        if not blind_path_info or not blind_path_info['detected']:
            return None
        
        direction = blind_path_info['direction']
        distance = blind_path_info['distance']
        
        # 根据距离和方向生成提示
        if distance == 'close':
            if direction == 'center':
                return {
                    'text': '已对正，继续前进', 
                    'audio': '已对正.wav'
                }
            elif direction == 'left':
                return {
                    'text': '盲道在左侧，请向右调整', 
                    'audio': '向右.wav'
                }
            else:
                return {
                    'text': '盲道在右侧，请向左调整', 
                    'audio': '向左.wav'
                }
        elif distance == 'medium':
            if direction == 'center':
                return {
                    'text': '继续前进', 
                    'audio': '向前.wav'
                }
            elif direction == 'left':
                return {
                    'text': '盲道偏左，向右调整', 
                    'audio': '向右.wav'
                }
            else:
                return {
                    'text': '盲道偏右，向左调整', 
                    'audio': '向左.wav'
                }
        else:
            return {
                'text': '发现盲道，继续前进',
                'audio': '向前.wav'
            }
    
    def _generate_messages(self, result: dict):
        """
        根据检测结果生成消息
        将消息添加到result['messages']中，供main.py转发
        
        改进：
        1. 避免重复播放相同状态的语音
        2. 添加"未识别"状态的语音提示
        3. 更智能的状态变化检测
        4. 添加时间间隔控制，避免频繁播报
        """
        import time
        current_time = time.time()
        messages = []
        
        # 1. 盲道导航提示
        if self.navigation_enabled:
            blind_path = result.get('blind_path', {})
            
            if blind_path.get('detected'):
                # 检测到盲道
                current_state = f"{blind_path['direction']}_{blind_path['distance']}"
                
                # 首次检测到盲道（从无到有）
                if self.last_blind_path_state is None or self.last_blind_path_state == 'none':
                    # 检查时间间隔
                    if current_time - self.last_audio_time['blind_detected'] >= self.audio_interval:
                        if '已识别盲道' in self.audio_cache:
                            messages.append({
                                'type': 'AUDIO',
                                'data': {'audio': self.audio_cache['已识别盲道']}
                            })
                            messages.append({
                                'type': 'FINAL',
                                'data': {'text': '[导航] 已识别盲道'}
                            })
                            self.last_audio_time['blind_detected'] = current_time
                
                # 只在状态变化时播放方向提示
                if current_state != self.last_blind_path_state:
                    # 检查时间间隔
                    if current_time - self.last_audio_time['blind_direction'] >= self.audio_interval:
                        # 更新状态
                        self.last_blind_path_state = current_state
                        
                        # 播放方向提示
                        if result.get('navigation_prompt'):
                            prompt = result['navigation_prompt']
                            audio_name = prompt['audio'].replace('.wav', '').replace('.WAV', '')
                            
                            if audio_name in self.audio_cache:
                                messages.append({
                                    'type': 'AUDIO',
                                    'data': {'audio': self.audio_cache[audio_name]}
                                })
                            
                            messages.append({
                                'type': 'FINAL',
                                'data': {'text': f"[导航] {prompt['text']}"}
                            })
                            self.last_audio_time['blind_direction'] = current_time
            else:
                # 未检测到盲道
                if self.last_blind_path_state != 'none':
                    # 检查时间间隔
                    if current_time - self.last_audio_time['blind_not_detected'] >= self.audio_interval:
                        self.last_blind_path_state = 'none'
                        
                        # 提示"未识别盲道"
                        if '未识别盲道' in self.audio_cache:
                            messages.append({
                                'type': 'AUDIO',
                                'data': {'audio': self.audio_cache['未识别盲道']}
                            })
                        messages.append({
                            'type': 'FINAL',
                            'data': {'text': '[导航] 未识别盲道'}
                        })
                        self.last_audio_time['blind_not_detected'] = current_time
        
        # 2. 红绿灯检测
        if self.traffic_mode:
            traffic = result.get('traffic_light', {})
            
            if traffic.get('detected'):
                color = traffic['color']
                
                # 只在颜色变化时播放语音
                if color != self.last_traffic_light_color:
                    # 检查时间间隔
                    if current_time - self.last_audio_time['traffic_light'] >= self.audio_interval:
                        self.last_traffic_light_color = color
                        
                        color_map = {'red': '红灯', 'yellow': '黄灯', 'green': '绿灯'}
                        color_text = color_map.get(color, '未知')
                        
                        if color_text in self.audio_cache:
                            messages.append({
                                'type': 'AUDIO',
                                'data': {'audio': self.audio_cache[color_text]}
                            })
                        
                        messages.append({
                            'type': 'FINAL',
                            'data': {'text': f'[过马路] {color_text}'}
                        })
                        self.last_audio_time['traffic_light'] = current_time
            else:
                # 未检测到红绿灯
                if self.last_traffic_light_color is not None:
                    self.last_traffic_light_color = None
        
        # 3. 斑马线检测
        if self.traffic_mode:
            zebra = result.get('zebra_crossing', {})
            
            if zebra.get('detected'):
                distance = zebra['distance']
                
                # 首次检测到斑马线或距离变化时播放语音
                if self.last_zebra_distance != distance:
                    # 检查时间间隔
                    if current_time - self.last_audio_time['zebra_crossing'] >= self.audio_interval:
                        self.last_zebra_distance = distance
                        
                        # 根据距离选择语音
                        if distance == 'far':
                            audio_name = '远处发现斑马线'
                        elif distance == 'medium':
                            audio_name = '正在靠近斑马线'
                        elif distance == 'close':
                            audio_name = '斑马线到了可以过马路'
                        else:
                            audio_name = None
                        
                        if audio_name and audio_name in self.audio_cache:
                            messages.append({
                                'type': 'AUDIO',
                                'data': {'audio': self.audio_cache[audio_name]}
                            })
                            messages.append({
                                'type': 'FINAL',
                                'data': {'text': f'[过马路] {audio_name}'}
                            })
                            self.last_audio_time['zebra_crossing'] = current_time
            else:
                # 未检测到斑马线
                if self.last_zebra_distance != 'none':
                    # 检查时间间隔
                    if current_time - self.last_audio_time['zebra_not_found'] >= self.audio_interval:
                        self.last_zebra_distance = 'none'
                        
                        # 提示"未发现斑马线"
                        if '未发现斑马线' in self.audio_cache:
                            messages.append({
                                'type': 'AUDIO',
                                'data': {'audio': self.audio_cache['未发现斑马线']}
                            })
                        messages.append({
                            'type': 'FINAL',
                            'data': {'text': '[过马路] 未发现斑马线'}
                        })
                        self.last_audio_time['zebra_not_found'] = current_time
        
        # 4. 物体搜索提示
        if self.search_target:
            search = result.get('search_result', {})
            
            if search.get('detected'):
                # 如果是刚捕获的模板，不播报
                if not search.get('is_template'):
                    position = search['position']
                    
                    # 检查时间间隔
                    if current_time - self.last_audio_time['search_found'] >= self.audio_interval:
                        # 根据位置选择语音
                        if position == 'left':
                            audio_name = '在画面左侧'
                        elif position == 'center':
                            audio_name = '在画面中间'
                        elif position == 'right':
                            audio_name = '在画面右侧'
                        else:
                            audio_name = None
                        
                        if audio_name and audio_name in self.audio_cache:
                            messages.append({
                                'type': 'AUDIO',
                                'data': {'audio': self.audio_cache[audio_name]}
                            })
                            messages.append({
                                'type': 'FINAL',
                                'data': {'text': f'[搜索] {audio_name}'}
                            })
                            self.last_audio_time['search_found'] = current_time
            else:
                # 未找到目标
                if current_time - self.last_audio_time['search_not_found'] >= self.audio_interval * 2:  # 搜索失败提示间隔更长
                    # 可以添加"未找到"的语音提示（如果有对应音频文件）
                    self.last_audio_time['search_not_found'] = current_time
        
        result['messages'] = messages