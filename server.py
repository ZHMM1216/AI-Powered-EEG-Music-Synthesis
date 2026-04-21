"""
===================================================================================
脑电助眠系统 - Web 服务器 (EEG-Music Web Server)
===================================================================================

项目背景：
    本模块是脑电助眠系统的 Web 服务端，提供 HTTP API 接口，允许用户通过
    网页上传 EEG 数据文件，系统自动分析情绪状态并生成对应的助眠音乐。

核心功能：
    1. 接收用户上传的 EEG CSV 文件
    2. 使用 EEGBiFormer 深度学习模型分析情绪状态
    3. 根据预测的情绪（Alert/Nervous/Relaxed）生成音乐 Prompt
    4. 调用 Suno API 生成助眠音乐
    5. 返回生成的音频文件供用户下载

技术架构：
    - 深度学习推理：EEGBiFormer 3分类模型（Alert/Nervous/Relaxed）
    - Web 框架：Flask（轻量级 HTTP 服务）
    - 数据处理：与训练阶段完全一致的预处理流程
    - 音乐生成：Suno API V5

API 端点：
    - GET  /           : 首页（HTML 界面）
    - GET  /health     : 健康检查
    - POST /generate_from_eeg : 上传 EEG 文件并生成音乐
    - GET  /download/<folder>/<filename> : 下载生成的音乐

===================================================================================
"""

# ==========================================
# 库导入说明
# ==========================================
from flask import Flask, request, send_file, jsonify, render_template
# Flask: Web 框架核心
# - Flask: 应用实例
# - request: 处理 HTTP 请求
# - send_file: 发送文件响应
# - jsonify: 返回 JSON 响应
# - render_template: 渲染 HTML 模板

import os                # 操作系统接口，用于文件和目录操作
import pandas as pd      # 数据处理，用于读取 CSV 文件
import numpy as np       # 数值计算，用于统计分析
import torch             # PyTorch 深度学习框架

# 导入自定义模块
from Real_Time_EEG_Predictor import RealTimePredictor      # 脑电信号预测器
from music_module import MusicGenerator      # 音乐生成器

# ==========================================
# Flask 应用初始化
# ==========================================
# 创建 Flask 应用实例
# __name__ 参数帮助 Flask 确定资源位置（如模板、静态文件）
app = Flask(__name__)

# ==========================================
# 配置区域 (Configuration)
# ==========================================
# 这些配置参数可以根据实际部署环境进行调整

# 脑电预测模型路径
# 这是训练好的 EEGBiFormer 模型权重文件
EEG_MODEL_PATH = 'best_model.pth' 

# Suno API 配置（官方规范）
# API 文档: https://docs.sunoapi.org/suno-api/generate-music
# 获取密钥: https://sunoapi.org/api-key

# 尝试从 config.py 导入配置
from config import SUNO_API_KEY as CONFIG_API_KEY
from config import SUNO_API_BASE_URL as CONFIG_BASE_URL
from config import SUNO_CALLBACK_URL as CONFIG_CALLBACK_URL
SUNO_API_KEY = CONFIG_API_KEY
SUNO_API_BASE_URL = CONFIG_BASE_URL
SUNO_CALLBACK_URL = CONFIG_CALLBACK_URL
print("✓ 从 config.py 加载 API 配置")

# 临时文件上传目录
# 用户上传的 EEG 文件会暂时保存在这里，处理完后删除
UPLOAD_FOLDER = './temp_uploads'

# 分类类别名称
# 必须与训练模型时的类别顺序一致：Alert=0, Nervous=1, Relaxed=2
# 文件夹排序后的顺序会决定标签编码
CLASS_NAMES = ['Alert', 'Nervous', 'Relaxed']

# 创建上传文件夹（如果不存在）
# exist_ok=True 表示如果文件夹已存在也不报错
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# 全局模型加载 (Global Model Initialization)
# ==========================================
# 在启动时加载模型，避免每次请求都重新加载
# 这样可以显著提高响应速度

# 步骤 1: 检测可用设备（GPU 或 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n" + "=" * 70)
print(f"正在初始化模型 (Device: {device})...")
print("=" * 70)

# 步骤 2: 初始化模型
# 2.1 初始化脑电预测器
# serial_port=None 表示不使用串口实时数据，只处理文件
print("\n[1/2] 正在加载脑电预测模型...")
eeg_predictor = RealTimePredictor(
    model_path=EEG_MODEL_PATH,      # 模型权重文件路径
    class_names=CLASS_NAMES,        # 类别名称列表
    serial_port=None                # 不使用串口（文件模式）
)
print("✓ 脑电预测模型加载成功")

# 2.2 初始化音乐生成器（Suno API 版本）
print("\n[2/2] 正在初始化 Suno API 音乐生成器...")

# 检查 API 密钥是否配置
if not SUNO_API_KEY:
    print("⚠ 警告：未找到 SUNO_API_KEY")
    print("音乐生成功能将不可用，但服务器会继续运行")
    music_gen = None
else:
    music_gen = MusicGenerator(
        api_key=SUNO_API_KEY,              # Suno API 密钥
        api_base_url=SUNO_API_BASE_URL,    # API 端点
        callback_url=SUNO_CALLBACK_URL     # 回调 URL
    )
    print("✓ Suno API 音乐生成器初始化成功")

# 2.3 打印成功信息
print("\n" + "=" * 70)
print("系统初始化完成，等待请求")
print("=" * 70 + "\n")



# ==========================================
# 核心业务逻辑函数
# ==========================================

def analyze_eeg_file(filepath):
    """
    分析 EEG 文件并生成音乐 Prompt
    
    功能流程：
        1. 读取 CSV 格式的 EEG 数据
        2. 使用滑动窗口逐段分析，每个窗口调用深度学习模型预测
        3. 计算所有窗口的平均概率分布
        4. 选择平均概率最高的情绪作为最终结果
        5. 根据最终情绪生成对应的音乐 Prompt
    
    参数：
        filepath (str): EEG CSV 文件的完整路径
    
    返回：
        tuple: (result_dict, error)
            - result_dict (dict): 分析结果字典，包含：
                - 'emotion': 最终预测的情绪类别（Alert/Nervous/Relaxed）
                - 'prompt': 音乐生成 Prompt
                - 'avg_probs': 平均概率分布 {'Alert': 0.xx, 'Nervous': 0.xx, 'Relaxed': 0.xx}
                - 'window_count': 分析的窗口数量
                - 'emotion_distribution': 各情绪在窗口中的统计分布（用于调试）
            - error (str): 错误信息，如果成功则为 None
    
    CSV 文件格式要求：
        - 第一行：表头（会被跳过）
        - 第 1 列：时间戳（会被忽略）
        - 第 2-17 列：16 个 EEG 通道的数据
        - 采样率：202 Hz
    
    决策逻辑：
        使用平均概率法（而非简单的窗口计数投票）来确保预测准确性：
        
        例如，假设有77个窗口：
        - Alert 出现 33 次，平均概率 39.45%
        - Nervous 出现 32 次，平均概率 42.59%
        - Relaxed 出现 12 次，平均概率 17.95%
        
        最终选择：Nervous（因为平均概率最高）
        
        这种方法更能反映模型的真实置信度，避免因窗口计数接近而导致的误判。
    """
    # 步骤 1: 读取 CSV 文件
    try:
        data = pd.read_csv(filepath, skiprows=1, usecols=range(1, 17)).values
    except Exception as e:
        return None, f"文件读取错误: {str(e)}"

    # 步骤 2: 检查模型是否已加载
    if eeg_predictor is None:
        return None, "服务端模型未加载，请检查服务器日志"

    # 步骤 3: 初始化预测器的缓冲区
    eeg_predictor._initialize_buffer()
    
    # 步骤 4: 准备存储变量
    emotions_list = []  # 存储每个窗口预测的情绪
    probs_list = []     # 存储每个窗口的概率分布
    
    window_size = eeg_predictor.window_size
    stride = eeg_predictor.stride

    print(f"正在分析文件: {len(data)} 个数据点 ({len(data)/202:.1f} 秒)...")

    # 步骤 5: 滑动窗口分析
    for i, frame in enumerate(data):
        eeg_predictor.buffer.append(frame)
        
        if i >= window_size and (i - window_size) % stride == 0:
            # 执行预测
            result = eeg_predictor.predict_full_cycle()
            
            # 记录结果
            emotions_list.append(result['emotion'])
            probs_list.append(result['probs'])

    # 步骤 6: 检查是否有足够的数据
    if not emotions_list:
        return None, "数据太短，无法进行有效预测 (需要至少 404 个数据点，约 2 秒)"

    # 步骤 7: 计算平均概率分布
    avg_probs = {}
    for class_name in eeg_predictor.class_names:
        probs_for_class = [p[class_name] for p in probs_list]
        avg_probs[class_name] = np.mean(probs_for_class)
    
    # 步骤 8: 基于平均概率确定最终情绪
    # 选择平均概率最高的情绪作为最终预测结果
    final_emotion = max(avg_probs, key=avg_probs.get)
    
    # 同时统计窗口分布（用于调试和分析）
    from collections import Counter
    emotion_counts = Counter(emotions_list)
    
    # 步骤 9: 生成最终的音乐 Prompt
    final_prompt = eeg_predictor.decision_engine.generate_prompt_from_emotion(final_emotion)
    
    # 步骤 10: 构建返回结果
    result_dict = {
        'emotion': final_emotion,
        'prompt': final_prompt,
        'avg_probs': avg_probs,
        'window_count': len(emotions_list),
        'emotion_distribution': dict(emotion_counts)
    }
    
    # 打印分析摘要
    print(f"分析完成！")
    print(f"  - 分析窗口数: {len(emotions_list)}")
    print(f"  - 情绪分布: {dict(emotion_counts)}")
    print(f"  - 最终情绪: {final_emotion}")
    print(f"  - 平均概率: {avg_probs}")
    
    return result_dict, None

# ==========================================
# API 路由定义 (API Routes)
# ==========================================

# ========== 路由 1: 首页 ==========
@app.route('/', methods=['GET'])
def index():
    """
    首页路由：渲染 HTML 用户界面
    
    HTTP 方法：GET
    URL：http://your-server:6006/
    
    功能：
        返回一个 HTML 页面，用户可以通过网页上传 EEG 文件
    
    返回：
        HTML 页面（templates/index.html）
    
    使用示例：
        在浏览器中访问：http://localhost:6006/
    """
    # render_template() 会在 templates/ 文件夹中查找 index.html
    print(f"收到首页请求: {request.remote_addr}")
    return render_template('index.html')


# ========== 路由 2: 健康检查 ==========
@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查端点：用于监控服务器状态
    
    HTTP 方法：GET
    URL：http://your-server:6006/health
    
    功能：
        返回服务器运行状态，用于负载均衡器或监控系统
    
    返回：
        JSON: {"status": "ok", "message": "..."}
    
    使用示例：
        curl http://localhost:6006/health
    """
    return jsonify({
        "status": "ok", 
        "message": "EEG-Music Server is running",
        "models_loaded": eeg_predictor is not None and music_gen is not None
    })


# ========== 路由 3: 主要业务逻辑 ==========
@app.route('/generate_from_eeg', methods=['POST'])
def process_eeg_flow():
    """
    核心 API：上传 EEG 文件并生成音乐
    
    HTTP 方法：POST
    URL：http://your-server:5000/generate_from_eeg
    Content-Type：multipart/form-data
    
    请求参数：
        file (File): EEG CSV 文件（必须包含16个通道，采样率202Hz）
    
    返回格式：
        成功（200）：
        {
            "success": true,
            "analysis": {
                "emotion": "Nervous",  // 预测的情绪类别
                "emotion_label": "焦虑/紧张",  // 中文标签
                "emotion_color": "#ff0080",  // 前端显示颜色
                "probabilities": {  // 概率分布
                    "Alert": 0.3945,
                    "Nervous": 0.4259,
                    "Relaxed": 0.1795
                },
                "prompt": "...",  // 音乐生成 Prompt
                "window_count": 77,  // 分析的窗口数
                "emotion_distribution": {  // 窗口统计
                    "Alert": 33,
                    "Nervous": 32,
                    "Relaxed": 12
                }
            },
            "music": [  // 生成的音乐列表
                {
                    "id": 1,
                    "title": "Track 1",
                    "download_url": "...",
                    ...
                }
            ]
        }
        
        失败（400/500）：
        {
            "error": "错误信息"
        }
    """
    # 打印请求日志
    print("\n" + "="*70)
    print("⚡ 收到 /generate_from_eeg 请求")
    print("="*70)
    print(f"请求方法: {request.method}")
    print(f"Content-Type: {request.content_type}")
    print(f"请求来源: {request.remote_addr}")
    print(f"请求文件字段: {list(request.files.keys())}")
    print(f"请求表单字段: {list(request.form.keys())}")
    print("="*70 + "\n")
    
    # 步骤 1: 检查请求中是否包含文件
    if 'file' not in request.files:
        print("❌ 错误：请求中没有 'file' 字段")
        return jsonify({"error": "没有上传文件，请在请求中包含 'file' 字段"}), 400
    
    # 步骤 2: 获取上传的文件对象
    file = request.files['file']
    
    # 步骤 3: 检查文件名是否为空
    if file.filename == '':
        return jsonify({"error": "文件名为空，请选择有效的文件"}), 400

    # 步骤 4: 保存上传的文件到临时目录
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"文件已保存: {filepath}")

    try:
        # 步骤 5: 分析 EEG 文件（新格式：返回字典）
        analysis_result, error = analyze_eeg_file(filepath)
        
        # 步骤 6: 检查分析是否成功
        if error:
            return jsonify({"error": error}), 500
        
        # 步骤 7: 从分析结果中提取信息
        emotion = analysis_result['emotion']  # 模型预测的情绪
        prompt = analysis_result['prompt']
        avg_probs = analysis_result['avg_probs']
        
        # 步骤 8: 根据情绪确定显示标签和颜色
        emotion_labels = {
            'Nervous': 'Anxious / Tense',
            'Alert': 'Alert / Focused',
            'Relaxed': 'Deep Relaxation'
        }
        emotion_colors = {
            'Nervous': '#ff0080',
            'Alert': '#00f2fe',
            'Relaxed': '#00ff88'
        }
        
        emotion_label = emotion_labels.get(emotion, 'Unknown')
        emotion_color = emotion_colors.get(emotion, '#ffffff')
        
        print(f"分析完成! 情绪: {emotion} ({emotion_label})")
        print(f"  - 概率分布: {avg_probs}")
        print(f"  - 生成 Prompt: {prompt}")

        # 步骤 9: 检查音乐生成器是否可用
        if music_gen:
            # 生成音乐（返回两首音乐的列表）
            music_list = music_gen.generate_music(
                prompt=prompt,
                emotion_state=emotion  # 使用模型预测的情绪
            )
            
            # 构建响应数据
            response_data = {
                "success": True,
                "analysis": {
                    "emotion": emotion,  # 模型预测的情绪类别
                    "emotion_label": emotion_label,  # 中文标签
                    "emotion_color": emotion_color,  # 配色
                    "probabilities": {k: round(v, 4) for k, v in avg_probs.items()},  # 概率分布
                    "prompt": prompt,  # 音乐生成 Prompt
                    # 统计信息
                    "window_count": analysis_result['window_count'],
                    "emotion_distribution": analysis_result['emotion_distribution']
                },
                "music": []
            }
            
            # 添加每首音乐的信息
            for idx, music in enumerate(music_list):
                music_info = {
                    "id": idx + 1,
                    "title": music.get('title', f'Track {idx + 1}'),
                    "duration": music.get('duration', 0),
                    "size": music.get('size', 0),
                    "download_url": f"/download/{os.path.basename(os.path.dirname(music['path']))}/{os.path.basename(music['path'])}",
                    "emotion": music.get('emotion', emotion),
                    "timestamp": music.get('timestamp', '')
                }
                response_data["music"].append(music_info)
            
            return jsonify(response_data)
        else:
            return jsonify({"error": "MusicGen 模型未加载，无法生成音乐"}), 500

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500
    
    finally:
        # 清理临时文件
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"临时文件已删除: {filepath}")



# ========== 路由 4: 音频文件下载 ==========
@app.route('/download/<folder>/<filename>', methods=['GET'])
def download_music(folder, filename):
    """
    音频文件下载端点
    
    参数：
        folder: 音乐文件夹名称（格式：情绪_时间戳）
        filename: 音频文件名
    
    返回：
        音频文件（MP3 格式）
    """
    music_folder = './generated_music'
    file_path = os.path.join(music_folder, folder, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 404
    
    return send_file(
        file_path,
        mimetype="audio/mpeg",
        as_attachment=True,
        download_name=filename
    )

# ==========================================
# 服务器启动入口
# ==========================================
if __name__ == '__main__':
    """
    启动 Flask 开发服务器
    
    参数说明：
        host='0.0.0.0': 监听所有网络接口
                       - 0.0.0.0: 允许外部访问
                       - 127.0.0.1: 只允许本地访问
        
        port=6006: 监听端口号
                  - 可以改为任何未被占用的端口
                  - 常用端口：5000, 8000, 8080, 6006
    
    访问方式：
        - 本地访问：http://localhost:6006
        - 局域网访问：http://your-ip:6006
        - 公网访问：需要配置防火墙和端口转发
    
    生产环境部署：
        不要使用 Flask 自带的开发服务器！
        推荐使用：
        - Gunicorn: gunicorn -w 4 -b 0.0.0.0:6006 server:app
        - uWSGI: uwsgi --http 0.0.0.0:6006 --wsgi-file server.py --callable app
        - Nginx + Gunicorn: 反向代理 + 负载均衡
    
    调试模式：
        开发时可以启用调试模式（自动重载、详细错误信息）：
        app.run(host='0.0.0.0', port=6006, debug=True)
    """
    # 从 config.py 读取服务器配置
    try:
        from config import SERVER_HOST, SERVER_PORT, SERVER_DEBUG
        host = SERVER_HOST
        port = SERVER_PORT
        debug = SERVER_DEBUG
        print(f"✓ 使用 config.py 中的服务器配置")
    except ImportError:
        # 如果没有 config.py，使用默认值
        host = '0.0.0.0'
        port = 5000
        debug = False
        print(f"✓ 使用默认服务器配置")
    
    print("\n" + "="*70)
    print("启动 EEG-Music Web Server")
    print("="*70)
    print(f"访问地址: http://localhost:{port}")
    print(f"监听地址: {host}:{port}")
    print("="*70)
    print("提示：按 Ctrl+C 停止服务器\n")
    
    # 启动服务器
    app.run(
        host=host,   # 从配置读取
        port=port,   # 从配置读取
        debug=debug  # 从配置读取
    )
