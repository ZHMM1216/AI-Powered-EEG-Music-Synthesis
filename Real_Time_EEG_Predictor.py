"""
===================================================================================
脑电助眠系统 - 实时预测器 (Real-Time EEG Predictor)
===================================================================================

项目背景：
    本系统通过实时监测用户的脑电信号（EEG），使用深度学习模型智能判断用户的情绪状态
    （Alert/Nervous/Relaxed），并自动生成适合当前状态的助眠音乐提示词（Prompt），
    用于驱动 Suno API 等音乐生成模型。

系统架构：
    采用端到端的深度学习架构，完全依赖训练好的 EEGBiFormer 模型进行情绪识别。
    
    核心组件：
    1. 数据预处理：带通滤波 + Z-score标准化（与训练阶段完全一致）
    2. 深度学习推理：EEGBiFormer 3分类模型（Alert/Nervous/Relaxed）
    3. 决策引擎：将预测的情绪映射到对应的音乐生成 Prompt

核心流程：
    原始EEG数据 -> 滤波 + 标准化 -> 深度学习推理 -> 情绪分类 -> 音乐Prompt生成

===================================================================================
"""

# ==========================================
# 库导入说明
# ==========================================
import torch              # PyTorch 深度学习框架，用于加载和运行神经网络模型
import torch.nn as nn     # PyTorch 神经网络模块
import numpy as np        # 数值计算库，用于数组操作和数学运算
import pandas as pd       # 数据处理库，用于读取 CSV 格式的 EEG 数据
from serial import Serial # 串口通信库，用于实时接收硬件设备的 EEG 数据（可选）
from scipy.signal import butter, filtfilt, welch  # 信号处理工具：滤波器设计、零相位滤波、功率谱估计
from collections import deque, Counter  # deque: 高效的双端队列，用于滑动窗口缓冲区



# ==========================================
# 1. 信号处理工具函数
# ==========================================

def bandpass_filter(data, sampling_rate, lowcut, highcut):
    """
    巴特沃斯带通滤波器 (Butterworth Bandpass Filter)
    
    功能：
        去除 EEG 信号中的低频漂移（如头部移动）和高频噪声（如肌电干扰），
        只保留有效的脑电频段（通常为 0.5-40Hz）。
    
    参数：
        data (np.ndarray): 原始 EEG 数据，形状为 (时间点数, 通道数)，例如 (404, 16)
        sampling_rate (int): 采样率，单位 Hz，本项目中为 202 Hz
        lowcut (float): 截止频率下限，单位 Hz，例如 0.5 Hz
        highcut (float): 截止频率上限，单位 Hz，例如 40 Hz
    
    返回：
        np.ndarray: 滤波后的数据，形状与输入相同
    
    原理：
        1. 计算奈奎斯特频率 (Nyquist Frequency)：采样率的一半，是能处理的最高频率
        2. 将截止频率归一化到 [0, 1] 区间
        3. 使用 butter() 设计 4 阶巴特沃斯滤波器的系数 (b, a)
        4. 使用 filtfilt() 进行零相位滤波（前向+后向滤波，避免相位失真）
    
    注意：
        - axis=0 表示沿时间轴滤波，不对通道间混淆
        - 使用 float64 精度以提高数值稳定性
    """
    # 步骤 1: 计算奈奎斯特频率
    nyq = 0.5 * sampling_rate  # 例如：202 / 2 = 101 Hz
    
    # 步骤 2: 归一化截止频率
    low = lowcut / nyq   # 例如：0.5 / 101 ≈ 0.00495
    high = highcut / nyq # 例如：40 / 101 ≈ 0.396
    
    # 步骤 3: 设计滤波器系数
    # order=4: 4阶滤波器，阶数越高，过渡带越陡峭，但计算量越大
    # btype='band': 带通滤波器（保留 low 到 high 之间的频率）
    b, a = butter(4, [low, high], btype='band')
    
    # 步骤 4: 零相位滤波
    # filtfilt 会先正向滤波一次，再反向滤波一次，消除相位延迟
    # axis=0: 沿第 0 维（时间轴）滤波
    return filtfilt(b, a, data.astype(np.float64), axis=0)


# ==========================================
# 2. 决策引擎 (Decision Layer)
# ==========================================

class DecisionEngine:
    """
    决策引擎：将深度学习模型预测的情绪映射到音乐生成提示词
    
    设计理念：
        直接使用训练好的深度学习模型进行情绪分类，无需额外的物理特征融合。
        模型经过专门训练，能够准确识别三种情绪状态。
    
    三分类情绪标签：
        - Nervous（焦虑/紧张）：需要更强效的降噪和冥想引导音乐
        - Alert（警觉/专注）：正常工作状态，需要平缓过渡音乐
        - Relaxed（放松）：已进入放松状态，需要深度睡眠音乐
    """
    
    def __init__(self, class_names=['Alert', 'Nervous', 'Relaxed']):
        """
        初始化决策引擎，定义音乐生成提示词模板
        
        参数：
            class_names (list): 类别名称列表，必须与模型训练时的顺序一致
                               默认：['Alert', 'Nervous', 'Relaxed']（按字母排序）
        
        Prompt 设计原则：
            - 使用纯音乐术语，避免医疗/治疗相关词汇
            - 描述具体的乐器、节奏、氛围
            - 适配不同情绪状态的音乐需求
        """
        self.class_names = class_names
        
        # 针对三类情绪优化的 Suno Prompt
        self.prompts = {
            # 状态：焦虑/紧张 (Nervous) → 策略：温暖治愈的钢琴音乐，稳定情绪
            'Nervous': "gentle piano melody with warm strings, soft female humming, healing and comforting, 65 bpm, emotional ballad, studio ghibli style, cinematic and tender",
            
            # 状态：警觉/专注 (Alert) → 策略：轻松愉悦的轻爵士，帮助过渡放松
            'Alert': "smooth jazz piano trio, soft brushed drums, mellow upright bass, cafe ambience, 70 bpm, relaxing afternoon vibe, bossa nova influence, warm and pleasant",
            
            # 状态：深度放松 (Relaxed) → 策略：温暖舒适的轻音乐
            'Relaxed': "soft acoustic guitar and piano duet, gentle melody, warm strings, light jazz brushes, 55 bpm, cozy and heartwarming, lo-fi bedroom pop"
        }
    def get_emotion_from_model(self, probs):
        """
        从模型概率分布获取情绪分类
        
        参数：
            probs (np.ndarray or list): 模型输出的概率分布，形状 (num_classes,)
                                       例如：[0.2, 0.1, 0.7] 表示 [Alert, Nervous, Relaxed]
        
        返回：
            str: 情绪类别名称（'Alert', 'Nervous', 或 'Relaxed'）
        
        说明：
            找到概率最大的类别作为预测结果。
            这是最直接和准确的方法，完全依赖深度学习模型的判断。
        """
        # 找到概率最大的类别索引
        pred_idx = np.argmax(probs)
        # 返回对应的类别名称
        return self.class_names[pred_idx]
    
    def generate_prompt_from_emotion(self, emotion):
        """
        根据情绪类别生成音乐 Prompt
        
        参数：
            emotion (str): 情绪类别，必须是 'Alert', 'Nervous', 或 'Relaxed'
        
        返回：
            str: 对应的音乐生成 Prompt
        """
        return self.prompts.get(emotion, self.prompts['Alert'])

# ==========================================
# 3. 深度学习模型定义 (Perception Layer)
# ==========================================

class EEGBiFormer(nn.Module):
    """
    EEGBiFormer: 基于 CNN + Transformer 的脑电信号分类模型
    
    架构设计：
        1. 输入归一化层 (BatchNorm1d)：稳定训练过程
        2. 特征提取器 (CNN)：3 层卷积网络，逐步提取时域特征
        3. Transformer 编码器：捕捉长距离时间依赖关系
        4. 分类头 (MLP)：将特征映射到类别概率
    
    输入：
        - 形状：(batch_size, time_points, channels)，例如 (1, 404, 16)
        - 含义：1 个样本，404 个时间点，16 个 EEG 通道
    
    输出：
        - 形状：(batch_size, num_classes)，例如 (1, 3)
        - 含义：每个类别的 logits（未归一化的分数），对应 [Alert, Nervous, Relaxed]
    
    注意：
        此模型定义必须与训练时（main.py）的定义完全一致，否则无法加载权重！
    """
    
    def __init__(self, num_classes, in_channels=16, dim=256):
        """
        初始化模型结构
        
        参数：
            num_classes (int): 分类类别数，本项目中为 3（Alert, Nervous, Relaxed）
            in_channels (int): 输入通道数，即 EEG 电极数量，默认 16
            dim (int): Transformer 的隐藏维度，默认 256
        """
        super().__init__()
        
        # 层 1: 输入归一化
        # 对每个通道独立进行 Batch Normalization，加速收敛
        self.input_norm = nn.BatchNorm1d(in_channels)
        
        # 层 2: 特征提取器（3 层 CNN）
        # 每层包含：卷积 -> 批归一化 -> ReLU 激活
        # kernel_size=9: 卷积核大小，捕捉局部时间模式
        # padding=4: 保持时间维度不变（9//2 = 4）
        self.feature_extractor = nn.Sequential(
            # 第 1 层：16 -> 64 通道
            nn.Conv1d(in_channels, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # 第 2 层：64 -> 128 通道
            nn.Conv1d(64, 128, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 第 3 层：128 -> 256 通道
            nn.Conv1d(128, dim, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )
        
        # 层 3: Transformer 编码器
        # 用于捕捉长距离的时间依赖关系（例如：脑电节律的周期性）
        # d_model=256: 输入特征维度
        # nhead=8: 多头注意力的头数
        # dim_feedforward=1024: 前馈网络的隐藏层维度（通常是 d_model 的 4 倍）
        # dropout=0.2: 防止过拟合
        # num_layers=2: 堆叠 2 层 Transformer 编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=8, 
                dim_feedforward=dim*4, 
                dropout=0.2, 
                batch_first=True  # 输入形状为 (batch, seq, feature)
            ), 
            num_layers=2
        )
        
        # 层 4: 分类头（两层全连接网络）
        # 256 -> 128 -> num_classes
        self.head = nn.Sequential(
            nn.Linear(dim, dim//2),      # 256 -> 128
            nn.ReLU(),
            nn.Dropout(0.2),             # 防止过拟合
            nn.Linear(dim//2, num_classes)  # 128 -> 2
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数：
            x (torch.Tensor): 输入数据，形状 (batch, time, channels)，例如 (1, 404, 16)
        
        返回：
            torch.Tensor: 输出 logits，形状 (batch, num_classes)，例如 (1, 2)
        
        数据流：
            输入 (1, 404, 16) 
            -> 转置 (1, 16, 404) 
            -> BatchNorm 
            -> CNN (1, 256, 404) 
            -> 转置 (1, 404, 256) 
            -> Transformer (1, 404, 256) 
            -> 平均池化 (1, 256) 
            -> 分类头 (1, 2)
        """
        # 步骤 1: 数值安全处理（防止 NaN 和 Inf）
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=-20.0, max=20.0)
        
        # 步骤 2: 转置以适配 Conv1d（需要 (batch, channels, time)）
        x = x.transpose(1, 2)  # (1, 404, 16) -> (1, 16, 404)
        
        # 步骤 3: 输入归一化
        x = self.input_norm(x)
        
        # 步骤 4: CNN 特征提取
        x = self.feature_extractor(x)  # (1, 16, 404) -> (1, 256, 404)
        
        # 步骤 5: 转置回 (batch, time, features) 以适配 Transformer
        x = x.transpose(1, 2)  # (1, 256, 404) -> (1, 404, 256)
        
        # 步骤 6: Transformer 编码
        x = self.transformer(x)  # (1, 404, 256) -> (1, 404, 256)
        
        # 步骤 7: 时间维度平均池化（将所有时间点的特征取平均）
        x = x.mean(dim=1)  # (1, 404, 256) -> (1, 256)
        
        # 步骤 8: 分类头输出
        return self.head(x)  # (1, 256) -> (1, 2)

# ==========================================
# 4. 实时预测器 (System Integration)
# ==========================================

class RealTimePredictor:
    def __init__(self, model_path, class_names, serial_port=None, window_size=404, stride=202, sampling_rate=202):
        """
        实时脑电预测器
        
        参数：
            model_path (str): 模型权重文件路径
            class_names (list): 类别名称列表，必须与训练时的顺序一致
                               例如：['Alert', 'Nervous', 'Relaxed']
            serial_port (str): 串口号（可选），用于实时硬件接入
            window_size (int): 滑动窗口大小，默认 404 个点（约 2 秒）
            stride (int): 滑动步长，默认 202 个点（约 1 秒）
            sampling_rate (int): 采样率，默认 202 Hz
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = self.load_model(model_path)
        
        # 初始化决策引擎（传入类别名称）
        self.decision_engine = DecisionEngine(class_names=class_names)
        
        # 串口设置
        self.ser = None
        if serial_port:
            try: self.ser = Serial(port=serial_port, baudrate=3000000); print(f"Open serial: {serial_port}")
            except Exception as e: print(f"Serial Error: {e}")
            
        # 数据缓冲区
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.buffer = deque(maxlen=self.window_size)
        self._initialize_buffer()

    def _initialize_buffer(self):
        self.buffer.clear()
        [self.buffer.append(np.zeros(16)) for _ in range(self.window_size)]

    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        model = EEGBiFormer(num_classes=self.num_classes).to(self.device)
        try: 
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            return model
        except FileNotFoundError: print("Model not found!"); exit()

    def preprocess_window(self, window_data):
        # 1. 滤波 (0.5-40Hz，与训练一致)
        processed_data = bandpass_filter(window_data, sampling_rate=self.sampling_rate, lowcut=0.5, highcut=40)
        # 2. 标准化
        for i in range(processed_data.shape[1]):
            mean, std = np.mean(processed_data[:, i]), np.std(processed_data[:, i])
            processed_data[:, i] = (processed_data[:, i] - mean) / std if std > 1e-8 else 0.0
        # 3. 裁剪
        np.clip(processed_data, -20.0, 20.0, out=processed_data)
        return processed_data.astype(np.float32)

    def predict_full_cycle(self):
        """
        执行完整的预测周期：深度学习推理 -> 情绪分类 -> 生成 Prompt
        
        返回：
            dict: 包含以下键值的字典
                - 'emotion': str, 模型预测的情绪类别（'Alert', 'Nervous', 或 'Relaxed'）
                - 'prompt': str, 对应的音乐生成 Prompt
                - 'probs': dict, 所有类别的概率分布，例如 {'Alert': 0.2, 'Nervous': 0.1, 'Relaxed': 0.7}
        
        决策逻辑：
            完全依赖深度学习模型的预测结果：
            - 将预处理后的 EEG 数据输入模型
            - 获取每个类别的概率分布
            - 选择概率最高的类别作为最终预测结果
            - 根据预测的情绪生成对应的音乐 Prompt
        
        示例返回值：
            {
                'emotion': 'Relaxed',
                'prompt': 'deep sleep meditation, theta waves, cosmic drone, 40 bpm, very slow, minimalist',
                'probs': {'Alert': 0.15, 'Nervous': 0.05, 'Relaxed': 0.80}
            }
        """
        # 1. 获取数据并预处理
        raw_window = np.array(self.buffer)  # Shape: (404, 16)
        processed_window = self.preprocess_window(raw_window)
        
        # 2. 深度学习推理
        input_tensor = torch.from_numpy(processed_window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)  # Shape: (1, 3)
            probs_np = probs[0].cpu().numpy()  # 转为 numpy 数组
            
            # 3. 获取情绪分类（选择概率最大的类别）
            emotion = self.decision_engine.get_emotion_from_model(probs_np)
            
            # 4. 构建概率分布字典
            probs_dict = {name: float(prob) for name, prob in zip(self.class_names, probs_np)}
        
        # 5. 生成音乐 Prompt
        prompt = self.decision_engine.generate_prompt_from_emotion(emotion)
        
        # 6. 构建返回结果
        result = {
            'emotion': emotion,      # 模型预测的情绪类别
            'prompt': prompt,        # 音乐生成 Prompt
            'probs': probs_dict     # 所有类别的概率分布
        }
        
        return result

