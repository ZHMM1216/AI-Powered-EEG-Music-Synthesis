"""
===================================================================================
音乐生成模块 (Music Generation Module) 
===================================================================================

项目背景：
    本模块是脑电助眠系统的音乐生成部分，负责根据用户的放松状态自动生成
    适合的助眠音乐。使用 Suno API，通过文本提示词 (Prompt) 生成高质量的
    音乐片段。

核心功能：
    1. 调用 Suno API 生成音乐
    2. 根据文本 Prompt 生成音乐
    3. 下载并保存生成的音频文件
    4. 与脑电预测系统集成

技术栈：
    - Suno API: 专业的 AI 音乐生成服务
    - requests: HTTP 请求库
    - 支持异步任务查询和高质量音乐生成

Suno API 使用说明：
    1. 注册账号：访问 https://suno.ai 或相关API服务商
    2. 获取API密钥
    3. 设置环境变量或直接传入密钥
    4. 调用接口生成音乐

===================================================================================
"""

# ==========================================
# 库导入说明
# ==========================================
import requests          # HTTP 请求库，用于调用 Suno API
import os                # 操作系统接口，用于文件和目录操作
import uuid              # 通用唯一识别码生成器，用于创建唯一的文件名
import time              # 时间模块，用于轮询任务状态
import json              # JSON 处理库

class MusicGenerator:
    """
    音乐生成器类：封装 Suno API 的调用和音乐生成功能
    
    设计理念：
        将复杂的音乐生成流程封装成简单易用的接口，使得其他模块
        （如脑电预测器）可以轻松调用音乐生成功能。
    
    主要功能：
        1. API 初始化：配置 Suno API 密钥和端点
        2. 音乐生成：根据文本 Prompt 生成音乐（异步任务）
        3. 文件管理：自动下载和保存生成的音频文件
    
    使用示例：
        >>> generator = MusicGenerator(api_key='your-api-key')
        >>> audio_path = generator.generate_music("relaxing piano music")
        >>> print(f"音乐已保存到: {audio_path}")
    """
    
    def __init__(self, api_key=None, api_base_url=None, callback_url=None):
        """
        初始化音乐生成器
        
        参数：
            api_key (str): Suno API 密钥
                          获取方式: 访问 https://sunoapi.org/api-key
                          如果不提供，会从环境变量 SUNO_API_KEY 读取
            
            api_base_url (str): Suno API 端点地址
                               官方端点: https://api.sunoapi.org
                               如果不提供，会从环境变量 SUNO_API_BASE_URL 读取
            
            callback_url (str): 可选的回调 URL
                               任务完成时 Suno 会向此 URL 发送回调通知
                               如果不提供，将使用轮询方式查询任务状态
        
        工作流程：
            1. 加载 API 配置
            2. 验证 API 密钥
            3. 打印初始化信息
        """
        # 步骤 1: 加载 API 配置
        # 优先使用传入的参数，否则从环境变量读取
        self.api_key = api_key or os.getenv('SUNO_API_KEY')
        self.api_base_url = api_base_url or os.getenv('SUNO_API_BASE_URL', 'https://api.sunoapi.org')
        self.callback_url = callback_url or os.getenv('SUNO_CALLBACK_URL') or 'http://localhost/callback'
        
        # 步骤 2: 验证 API 密钥
        if not self.api_key:
            raise ValueError(
                "未找到 Suno API 密钥！\n"
            )
        
        # 步骤 3: 设置请求头
        # 根据官方文档 https://docs.sunoapi.org 使用 Authorization: Bearer 认证
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        

    def generate_music(self, prompt, output_folder='./generated_music', model='V5', emotion_state='Unknown'):
        """
        根据文本提示词调用 Suno API 生成音乐并保存为音频文件
        
        参数：
            prompt (str): 音乐生成的文本提示词（最多 500 字符）
            output_folder (str): 输出文件夹路径，默认 './generated_music'
            model (str): Suno 模型版本，默认 'V5'
            emotion_state (str): 情绪状态，用于文件命名（Nervous/Alert/Relaxed）
        
        返回：
            list: 包含两首音乐信息的列表，每首音乐包含：
                  {
                      'path': 文件路径,
                      'title': 歌曲标题,
                      'duration': 时长（秒）,
                      'size': 文件大小（KB）,
                      'emotion': 情绪状态,
                      'timestamp': 生成时间戳
                  }
        """
        from datetime import datetime
        
        # 步骤 1: 创建带时间戳的输出文件夹
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dated_folder = os.path.join(output_folder, f"{emotion_state}_{timestamp}")
        if not os.path.exists(dated_folder):
            os.makedirs(dated_folder)
            print(f"创建输出文件夹: {dated_folder}")
        
        # 步骤 2: 验证 prompt 长度
        if len(prompt) > 500:
            print(f"警告：Prompt 长度 ({len(prompt)}) 超过 500 字符，将被截断")
            prompt = prompt[:500]
        
        # 步骤 3: 打印生成信息
        print(f"\n" + "=" * 70)
        print(f"正在为提示词生成音乐: '{prompt}'")
        print(f"使用模型: {model}")
        print(f"情绪状态: {emotion_state}")
        print(f"=" * 70)
        
        # 步骤 4: 构建 API 请求数据
        request_data = {
            "customMode": False,
            "instrumental": True,
            "prompt": prompt,
            "model": model,
            "callBackUrl": self.callback_url
        }
        
        try:
            # 步骤 5: 调用 API 创建生成任务
            print("正在提交音乐生成请求到 Suno API...")
            
            if '/api/v1' in self.api_base_url:
                create_url = f"{self.api_base_url}/generate"
            else:
                create_url = f"{self.api_base_url}/api/v1/generate"
            
            print(f"  - 请求 URL: {create_url}")
            print(f"  - 请求数据: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
            
            response = requests.post(
                create_url,
                headers=self.headers,
                json=request_data,
                timeout=30
            )
            
            print(f"  - 响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                raise Exception(f"API 请求失败 (状态码: {response.status_code})\n响应内容: {response.text}")
            
            result = response.json()
            
            if result.get('code') != 200:
                error_msg = result.get('msg', '未知错误')
                raise Exception(f"API 返回错误: {error_msg}")
            
            task_id = result.get('data', {}).get('taskId')
            if not task_id:
                raise Exception(f"无法提取任务 ID: {result}")
            
            print(f"✓ 任务创建成功 (ID: {task_id})")
            print(f"正在等待音乐生成完成...")
            
            # 步骤 6: 轮询获取两首音乐
            songs = self._wait_for_all_songs(task_id, max_wait_time=300)
            
            # 步骤 7: 下载并保存两首音乐
            music_list = []
            for idx, song in enumerate(songs, 1):
                audio_url = song.get('audioUrl') or song.get('streamAudioUrl')
                if not audio_url:
                    print(f"  警告：第 {idx} 首歌曲无法获取 URL，跳过")
                    continue
                
                print(f"\n正在下载第 {idx} 首音乐...")
                
                audio_response = requests.get(audio_url, timeout=60)
                if audio_response.status_code != 200:
                    print(f"  警告：第 {idx} 首音乐下载失败")
                    continue
                
                # 构建文件名：情绪_时间戳_序号.mp3
                song_title = song.get('title', f'Track_{idx}')
                safe_title = "".join(c for c in song_title if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
                filename = f"{emotion_state}_{timestamp}_{idx}_{safe_title}.mp3"
                file_path = os.path.join(dated_folder, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(audio_response.content)
                
                file_size = len(audio_response.content) / 1024
                duration = song.get('duration', 0)
                
                print(f"✓ 第 {idx} 首音乐保存成功")
                print(f"  - 标题: {song_title}")
                print(f"  - 文件: {filename}")
                print(f"  - 大小: {file_size:.2f} KB")
                print(f"  - 时长: {duration:.1f} 秒")
                
                music_list.append({
                    'path': file_path,
                    'title': song_title,
                    'duration': duration,
                    'size': round(file_size, 2),
                    'emotion': emotion_state,
                    'timestamp': timestamp,
                    'prompt': prompt
                })
            
            print(f"\n" + "=" * 70)
            print(f"✓ 音乐生成完成！共生成 {len(music_list)} 首音乐")
            print(f"  - 保存位置: {dated_folder}")
            print(f"=" * 70 + "\n")
            
            return music_list
            
        except Exception as e:
            print(f"\n" + "=" * 70)
            print(f"✗ 错误：音乐生成失败！")
            print(f"错误信息: {str(e)}")
            print(f"=" * 70 + "\n")
            raise
    
    def _wait_for_all_songs(self, task_id, max_wait_time=300, poll_interval=10):
        """
        轮询 Suno API 直到两首歌曲都生成完成
        
        返回：
            list: 包含两首歌曲信息的列表
        """
        if '/api/v1' in self.api_base_url:
            query_url = f"{self.api_base_url}/generate/record-info"
        else:
            query_url = f"{self.api_base_url}/api/v1/generate/record-info"
        
        start_time = time.time()
        attempts = 0
        
        print(f"\n开始轮询任务状态（任务 ID: {task_id}）...")
        
        while True:
            attempts += 1
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_time:
                raise Exception(f"任务超时！已等待 {elapsed:.1f} 秒")
            
            try:
                response = requests.get(
                    query_url,
                    headers=self.headers,
                    params={"taskId": task_id},
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"  [尝试 {attempts}] 查询失败，{poll_interval}秒后重试...")
                    time.sleep(poll_interval)
                    continue
                
                result = response.json()
                
                if result.get('code') != 200:
                    print(f"  [尝试 {attempts}] API 返回错误，{poll_interval}秒后重试...")
                    time.sleep(poll_interval)
                    continue
                
                data_wrapper = result.get('data', {})
                task_status = data_wrapper.get('status', 'PENDING')
                
                print(f"  [尝试 {attempts}] 任务状态: {task_status} (已等待 {elapsed:.1f}秒)")
                
                if task_status == 'FAILED':
                    error_msg = data_wrapper.get('errorMessage', '未知错误')
                    raise Exception(f"音乐生成失败: {error_msg}")
                
                if task_status in ['PENDING', 'GENERATING', 'TEXT_SUCCESS', 'FIRST_SUCCESS']:
                    time.sleep(poll_interval)
                    continue
                
                if task_status == 'SUCCESS':
                    response_data = data_wrapper.get('response', {})
                    songs = response_data.get('sunoData', [])
                    
                    if not songs or len(songs) == 0:
                        print(f"  数据为空，{poll_interval}秒后重试...")
                        time.sleep(poll_interval)
                        continue
                    
                    # 检查是否所有歌曲都有可用的 URL
                    ready_songs = [s for s in songs if s.get('audioUrl') or s.get('streamAudioUrl')]
                    
                    if len(ready_songs) >= 2:
                        print(f"✓ 两首歌曲均已就绪！总耗时: {elapsed:.1f} 秒")
                        return ready_songs[:2]
                    elif len(ready_songs) >= 1:
                        print(f"  第一首已就绪，等待第二首...")
                        time.sleep(poll_interval)
                        continue
                    else:
                        print(f"  音频还在生成中，{poll_interval}秒后重试...")
                        time.sleep(poll_interval)
                        continue
                
                time.sleep(poll_interval)
                continue
                    
            except requests.exceptions.RequestException as e:
                print(f"  网络错误: {e}，{poll_interval}秒后重试...")
                time.sleep(poll_interval)
                continue
    
    def _wait_for_completion(self, task_id, max_wait_time=300, poll_interval=10):
        """
        轮询 Suno API 直到任务完成
        
        参数：
            task_id (str): 任务 ID
            max_wait_time (int): 最大等待时间（秒），默认 300 秒（5 分钟）
            poll_interval (int): 轮询间隔（秒），默认 10 秒
                                建议 5-10 秒，避免请求过频繁
        
        返回：
            str: 生成的音频文件 URL（MP3 格式）
                注意：Suno API 返回 2 首歌曲，本方法返回第一首的 audio_url
        
        工作流程：
            1. 每隔 poll_interval 秒查询一次任务状态
            2. 使用官方的任务查询接口
            3. 解析返回的歌曲列表，提取第一首的 audio_url
            4. 如果超时，抛出异常
        
        API 文档：
            https://docs.sunoapi.org/suno-api/get-music-generation-details
        """
        # 官方任务查询端点
        # 根据官方文档：https://docs.sunoapi.org
        # 查询端点: /api/v1/generate/record-info?taskId=xxx
        if '/api/v1' in self.api_base_url:
            query_url = f"{self.api_base_url}/generate/record-info"
        else:
            query_url = f"{self.api_base_url}/api/v1/generate/record-info"
        
        start_time = time.time()
        attempts = 0
        
        print(f"\n开始轮询任务状态（任务 ID: {task_id}）...")
        
        while True:
            attempts += 1
            elapsed = time.time() - start_time
            
            # 检查是否超时
            if elapsed > max_wait_time:
                raise Exception(
                    f"任务超时！已等待 {elapsed:.1f} 秒\n"
                    f"建议：\n"
                    f"  1. 增加 max_wait_time 参数\n"
                    f"  2. 检查任务是否在 Suno 后台仍在处理\n"
                    f"  3. 访问 https://sunoapi.org 查看任务状态"
                )
            
            try:
                # 查询任务状态
                # 参数格式: ?taskId=xxx （根据官方文档）
                response = requests.get(
                    query_url,
                    headers=self.headers,
                    params={"taskId": task_id},
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"  [尝试 {attempts}] 查询失败 (状态码: {response.status_code})，{poll_interval}秒后重试...")
                    time.sleep(poll_interval)
                    continue
                
                result = response.json()
                
                # 检查响应的 code 字段
                if result.get('code') != 200:
                    print(f"  [尝试 {attempts}] API 返回错误: {result.get('msg')}，{poll_interval}秒后重试...")
                    time.sleep(poll_interval)
                    continue
                
                # 提取数据
                # 官方响应格式: {"code": 200, "msg": "success", "data": {"taskId": "...", "status": "...", "response": {"data": [...]}}}
                data_wrapper = result.get('data', {})
                task_status = data_wrapper.get('status', 'PENDING')
                
                print(f"  [尝试 {attempts}] 任务状态: {task_status} (已等待 {elapsed:.1f}秒)")
                
                # 如果任务失败
                if task_status == 'FAILED':
                    error_msg = data_wrapper.get('errorMessage', '未知错误')
                    raise Exception(f"音乐生成失败: {error_msg}")
                
                # 如果任务还在处理中（包括 TEXT_SUCCESS 和 FIRST_SUCCESS 状态）
                # TEXT_SUCCESS: 文本/歌词生成完成，音乐还在生成中
                # FIRST_SUCCESS: 第一首歌完成，第二首还在生成中
                if task_status in ['PENDING', 'GENERATING', 'TEXT_SUCCESS', 'FIRST_SUCCESS']:
                    if task_status == 'TEXT_SUCCESS':
                        print(f"  歌词生成完成，音乐生成中...{poll_interval}秒后重试...")
                    elif task_status == 'FIRST_SUCCESS':
                        print(f"  第一首歌已完成，第二首生成中...{poll_interval}秒后重试...")
                    else:
                        print(f"  任务进行中，{poll_interval}秒后重试...")
                    time.sleep(poll_interval)
                    continue
                
                # 如果任务成功完成
                if task_status == 'SUCCESS':
                    response_data = data_wrapper.get('response', {})
                    # 注意：Suno API 的字段名是 'sunoData' 不是 'data'
                    songs = response_data.get('sunoData', [])
                    
                    # 调试信息：打印完整响应以便诊断
                    if not songs or len(songs) == 0:
                        print(f"  警告：数据为空！")
                        print(f"  完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
                        print(f"  {poll_interval}秒后重试...")
                        time.sleep(poll_interval)
                        continue
                    
                    # 获取第一首歌曲的信息
                    first_song = songs[0]
                    print(f"  找到 {len(songs)} 首歌曲")
                    print(f"  第一首歌曲信息: {json.dumps(first_song, ensure_ascii=False, indent=2)[:500]}...")
                    
                    # 注意：Suno API 使用驼峰命名：audioUrl, streamAudioUrl
                    audio_url = first_song.get('audioUrl')  # 驼峰命名
                    stream_url = first_song.get('streamAudioUrl')  # 驼峰命名
                
                    # 优先使用 stream_url（30-40秒可用），如果不可用则等待 audio_url（2-3分钟）
                    if stream_url:
                        print(f"✓ 流媒体 URL 已就绪！总耗时: {elapsed:.1f} 秒")
                        print(f"  - Stream URL: {stream_url[:60]}...")
                        return stream_url
                    
                    if audio_url:
                        print(f"✓ 完整音频 URL 已就绪！总耗时: {elapsed:.1f} 秒")
                        print(f"  - Audio URL: {audio_url[:60]}...")
                        # 打印歌曲信息
                        if first_song.get('title'):
                            print(f"  - 标题: {first_song.get('title')}")
                        if first_song.get('duration'):
                            print(f"  - 时长: {first_song.get('duration'):.1f} 秒")
                        return audio_url
                    
                    # 如果两个 URL 都没有，继续等待
                    print(f"  音频还在生成中，{poll_interval}秒后重试...")
                    time.sleep(poll_interval)
                    continue
                
                # 如果状态未知，继续等待
                print(f"  未知状态: {task_status}，{poll_interval}秒后重试...")
                time.sleep(poll_interval)
                continue
                    
            except requests.exceptions.RequestException as e:
                print(f"  网络请求错误: {e}，{poll_interval}秒后重试...")
                time.sleep(poll_interval)
                continue
            except Exception as e:
                print(f"  处理响应时出错: {e}，{poll_interval}秒后重试...")
                time.sleep(poll_interval)
                continue

