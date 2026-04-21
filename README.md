# 🧠 基于脑电情绪识别的音乐干预系统（CNN+Transformer 版 · 含前端交互）

## 📌 项目简介

本项目是一个**端到端的脑电（EEG）情绪识别与音乐干预系统**。系统通过深度学习模型（CNN + Transformer 架构）对用户的 16 通道 EEG 信号进行实时情绪分类，识别出三种情绪状态（**Alert / Nervous / Relaxed**），并根据识别结果自动调用 Suno API 生成个性化的助眠/放松音乐，实现"脑电采集 → 情绪识别 → 音乐生成"的完整闭环。

项目提供基于 Flask 的 Web 前端界面，用户可通过浏览器上传 EEG 数据文件，系统自动完成分析与音乐生成。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户浏览器（前端界面）                      │
│                   templates/index.html                          │
│              上传 EEG CSV 文件 / 播放与下载音乐                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST /generate_from_eeg
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Flask Web 服务器                             │
│                       server.py                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ EEG 文件分析      │→│ 实时预测器        │→│ 音乐生成模块  │  │
│  │ (滑动窗口分析)    │  │ Real_Time_EEG_   │  │ music_module │  │
│  │                  │  │ Predictor.py     │  │ .py          │  │
│  └──────────────────┘  └────────┬─────────┘  └──────┬───────┘  │
│                                 │                    │          │
│                    ┌────────────▼───────────┐        │          │
│                    │  EEGBiFormer 模型       │        │          │
│                    │  best_model.pth        │        │          │
│                    │  (CNN+Transformer)     │        │          │
│                    └────────────────────────┘        │          │
│                                                      ▼          │
│                                              ┌──────────────┐   │
│                                              │  Suno API    │   │
│                                              │  (云端音乐   │   │
│                                              │   生成服务)  │   │
│                                              └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目文件结构与说明

```
EEG-py-CNN+Transformer-有前端交互/
│
├── config.py                        # 全局配置文件
├── main.py                          # 模型训练脚本
├── Real_Time_EEG_Predictor.py       # 实时 EEG 预测器（模型推理 + 决策引擎）
├── music_module.py                  # 音乐生成模块（Suno API 封装）
├── server.py                        # Flask Web 服务器（核心入口）
├── EEG_wave_change_analysis.py      # EEG 脑电波形分析工具（α/β波）
├── 1.0EEGdata-preprocess.ipynb      # EEG 数据预处理教学 Notebook
├── best_model.pth                   # 训练好的最佳模型权重文件
├── requirements.txt                 # Python 依赖包列表
├── training_curves.png              # 训练过程曲线图（损失/准确率）
├── confusion_matrix.png             # 验证集混淆矩阵图
├── intervention_comparison.png      # 音乐干预前后脑电波对比图
├── templates/
│   └── index.html                   # 前端 Web 界面（HTML/CSS/JS）
└── dataset/                         # 数据集文件夹（需自行准备）
    └── train/
        ├── Alert/                   # Alert（警觉）类 EEG 数据
        │   └── *.csv
        ├── Nervous/                 # Nervous（焦虑）类 EEG 数据
        │   └── *.csv
        └── Relaxed/                 # Relaxed（放松）类 EEG 数据
            └── *.csv
```

---

## 📄 各文件详细说明

### 1. `config.py` — 全局配置文件

存放项目的核心配置参数，修改配置时只需编辑此文件。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `SUNO_API_KEY` | Suno API 密钥（用于音乐生成） | 需自行填写 |
| `SUNO_API_BASE_URL` | Suno API 端点地址 | `https://api.sunoapi.org` |
| `SUNO_CALLBACK_URL` | 回调 URL（留空则使用轮询模式） | `""` |
| `GENERATED_MUSIC_FOLDER` | 生成音乐的保存路径 | `./generated_music` |
| `MODEL_PATH` | 模型权重文件路径 | `./best_model.pth` |
| `SERVER_HOST` | 服务器监听地址 | `0.0.0.0` |
| `SERVER_PORT` | 服务器端口号 | `5000` |
| `SERVER_DEBUG` | 是否开启调试模式 | `False` |

---

### 2. `main.py` — 模型训练脚本

**功能**：从头训练 EEGBiFormer 深度学习模型，包含完整的数据加载、预处理、模型定义、训练和评估流程。

**核心组件**：

- **`bandpass_filter()`**：巴特沃斯带通滤波器（0.5-40Hz），去除 EEG 信号中的低频漂移和高频噪声。
- **`EEGDataset` 类**：自定义 PyTorch 数据集，负责：
  - 从 `dataset/train/` 目录按类别文件夹加载 CSV 文件
  - 对每个文件执行预处理流程：带通滤波 → Z-score 标准化 → 裁剪异常值
  - 使用滑动窗口（窗口大小 404 点 ≈ 2秒，步长 202 点 ≈ 1秒）切分样本
- **`EEGBiFormer` 类**：CNN + Transformer 混合模型
  - 3 层 1D CNN 提取局部时域特征（16→64→128→256 通道）
  - 2 层 Transformer 编码器捕捉长距离时间依赖
  - 全局平均池化 + 全连接分类头
- **`train_model()`**：训练函数，包含：
  - 类别权重平衡（处理数据不均衡）
  - AdamW 优化器 + ReduceLROnPlateau 学习率调度
  - 梯度裁剪（max_norm=1.0）
  - 早停机制（patience=15）
  - 自动保存最佳模型到 `best_model.pth`
- **`plot_curves()`**：绘制训练/验证的损失和准确率曲线
- **`plot_confusion_matrix()`**：绘制验证集的混淆矩阵

**数据集格式要求**：
- CSV 文件，第 1 行为表头（跳过）
- 第 1 列为时间戳（忽略），第 2-17 列为 16 通道 EEG 数据
- 采样率：202 Hz
- 数据按情绪类别放入对应的子文件夹（`Alert/`、`Nervous/`、`Relaxed/`）

**运行方式**：
```bash
python main.py
```

---

### 3. `Real_Time_EEG_Predictor.py` — 实时 EEG 预测器

**功能**：加载训练好的模型，对 EEG 数据进行实时情绪预测，并生成对应的音乐提示词。

**核心组件**：

- **`bandpass_filter()`**：与训练阶段一致的带通滤波函数
- **`DecisionEngine` 类**：决策引擎，负责：
  - 将模型输出的概率分布转换为情绪类别（选择最大概率对应的类别）
  - 为每种情绪生成专门设计的音乐 Prompt：
    - **Nervous**（焦虑）→ 温暖治愈的钢琴音乐，65 BPM
    - **Alert**（警觉）→ 轻松的爵士乐，70 BPM
    - **Relaxed**（放松）→ 温暖舒适的轻音乐，55 BPM
- **`EEGBiFormer` 类**：与 `main.py` 中完全一致的模型定义（用于加载权重）
- **`RealTimePredictor` 类**：实时预测器，核心类：
  - 自动检测 GPU/CPU 并加载模型
  - 使用滑动窗口缓冲区管理 EEG 数据流
  - `preprocess_window()`：对单个窗口执行预处理（滤波+标准化+裁剪）
  - `predict_full_cycle()`：执行完整预测周期（预处理→推理→情绪分类→Prompt生成）
  - 支持串口实时数据接入（可选）

---

### 4. `music_module.py` — 音乐生成模块

**功能**：封装 Suno API 的调用逻辑，根据文本 Prompt 生成高质量的 AI 音乐。

**核心组件**：

- **`MusicGenerator` 类**：
  - `generate_music(prompt, emotion_state)`：提交音乐生成任务，每次生成 2 首音乐
    - 自动创建带时间戳的输出文件夹
    - Prompt 长度限制 500 字符
    - 使用 Suno V5 模型生成纯器乐音乐
  - `_wait_for_all_songs(task_id)`：轮询等待两首音乐都生成完成
    - 轮询间隔 10 秒，最大等待 300 秒
    - 处理 PENDING → GENERATING → TEXT_SUCCESS → FIRST_SUCCESS → SUCCESS 状态流转
  - 自动下载 MP3 音频并保存到本地

**生成的文件结构**：
```
generated_music/
└── Nervous_20260213_143000/
    ├── Nervous_20260213_143000_1_TrackTitle.mp3
    └── Nervous_20260213_143000_2_TrackTitle.mp3
```

---

### 5. `server.py` — Flask Web 服务器（核心入口）

**功能**：系统的 Web 服务端，提供 HTTP API，是前端和后端的桥梁。

**API 端点**：

| 方法 | 路由 | 功能 |
|------|------|------|
| GET | `/` | 返回前端 HTML 界面 |
| GET | `/health` | 服务器健康检查 |
| POST | `/generate_from_eeg` | 上传 EEG 文件 → 分析情绪 → 生成音乐 |
| GET | `/download/<folder>/<filename>` | 下载生成的 MP3 音乐文件 |

**核心业务流程（`/generate_from_eeg`）**：

1. 接收用户上传的 EEG CSV 文件
2. 使用滑动窗口逐段分析，每个窗口通过模型预测情绪
3. 计算所有窗口的**平均概率分布**（而非简单投票），选择概率最高的情绪
4. 根据最终情绪生成音乐 Prompt
5. 调用 Suno API 生成 2 首助眠音乐
6. 返回分析结果和音乐下载链接

**返回数据示例**：
```json
{
  "success": true,
  "analysis": {
    "emotion": "Nervous",
    "emotion_label": "Anxious / Tense",
    "emotion_color": "#ff0080",
    "probabilities": {"Alert": 0.3945, "Nervous": 0.4259, "Relaxed": 0.1795},
    "prompt": "gentle piano melody with warm strings...",
    "window_count": 77,
    "emotion_distribution": {"Alert": 33, "Nervous": 32, "Relaxed": 12}
  },
  "music": [
    {"id": 1, "title": "Track 1", "download_url": "/download/..."}
  ]
}
```

**启动方式**：
```bash
python server.py
```
启动后访问 `http://localhost:5000` 进入 Web 界面。

---

### 6. `EEG_wave_change_analysis.py` — 脑电波形分析工具

**功能**：分析 EEG 数据中 α 波（8-13Hz）和 β 波（13-30Hz）随时间的变化，用于观察音乐干预前后脑电波的变化趋势。

**核心功能**：

- **`load_eeg_data()`**：加载 16 通道 EEG CSV 数据
- **`preprocess_eeg()`**：带通滤波 + Z-score 标准化
- **`calculate_band_power_timeseries()`**：基于 Welch 方法计算指定频段的能量时间序列
- **`plot_alpha_beta_changes()`**：绘制单个文件的 α/β 波能量随时间变化的图表
- **`compare_before_after()`**：**对比分析**干预前后的 α/β 波变化
  - 计算变化率百分比
  - 自动给出结论（如"α 波增强，β 波下降 → 放松效果显著"）

**可配置参数**：
- `SAMPLING_RATE`：采样率（默认 202 Hz）
- `WINDOW_SIZE`：滑动窗口大小（默认 4 秒）
- `OVERLAP`：窗口重叠比例（默认 0.75）
- `CHANNELS`：分析的通道（`'all'`、单通道索引、或通道索引列表）

**运行方式**：
```bash
python EEG_wave_change_analysis.py
```

---

### 7. `1.0EEGdata-preprocess.ipynb` — EEG 数据预处理教学 Notebook

**功能**：交互式教学笔记本，讲解 EEG 数据预处理的基础知识。

**内容涵盖**：
- **第一部分：Python 数值处理基础**
  - 读取 CSV 格式的 EEG 数据文件
  - NumPy 数组的基本操作
  - Matplotlib 绘制时域波形
- **第二部分：EEG 信号处理入门**
  - 带通滤波的原理与应用
  - FFT 频谱分析方法
  - 观察 α/β 等脑电频段特征
  - 理解项目中使用的完整预处理流程

---

### 8. `best_model.pth` — 训练好的模型权重文件

已训练好的 EEGBiFormer 模型权重（约 7.6MB），可直接加载用于情绪预测。

**模型参数**：
- 输入：(batch_size, 404, 16) — 404 个时间点 × 16 个 EEG 通道
- 输出：(batch_size, 3) — 三种情绪类别的 logits
- 架构：3 层 CNN + 2 层 Transformer Encoder + MLP 分类头

---

### 9. `templates/index.html` — 前端 Web 界面

**功能**：现代化的 Web 用户界面，采用深色主题 + 玻璃态设计风格。

**用户操作流程**：
1. 打开浏览器访问系统首页
2. 上传 EEG CSV 数据文件
3. 系统自动分析情绪状态并展示概率分布
4. 系统自动生成 2 首个性化音乐
5. 在线播放或下载生成的音乐

---

### 10. 图片文件

| 文件名 | 说明 |
|--------|------|
| `training_curves.png` | 模型训练过程中的损失和准确率曲线（4 子图：训练损失、验证损失、训练准确率、验证准确率） |
| `confusion_matrix.png` | 最佳模型在验证集上的混淆矩阵，显示各类别的分类效果 |
| `intervention_comparison.png` | 音乐干预前后 α 波和 β 波的对比分析图 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+（推荐使用 Conda 管理环境）
- CUDA（可选，有 GPU 可加速训练和推理）

### 1. 创建虚拟环境

```bash
conda create -n eeg_music_env python=3.10
conda activate eeg_music_env
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖说明**：

| 依赖包 | 用途 |
|--------|------|
| `Flask` | Web 框架，提供 HTTP 服务 |
| `torch` | PyTorch 深度学习框架 |
| `pandas` / `numpy` | 数据处理与数值计算 |
| `scipy` | 信号处理（带通滤波、频谱分析） |
| `scikit-learn` | 标签编码、混淆矩阵可视化 |
| `matplotlib` | 数据可视化与绘图 |
| `requests` | HTTP 请求（调用 Suno API） |
| `tqdm` | 训练进度条显示 |
| `pyserial` | 串口通信（实时 EEG 硬件数据采集，可选） |

### 3. 配置 API 密钥

编辑 `config.py`，填写你的 Suno API 密钥：

```python
SUNO_API_KEY = "你的API密钥"
```

> API 密钥获取地址：https://sunoapi.org/api-key

### 4. 启动 Web 服务

```bash
python server.py
```

启动成功后会显示：
```
✓ 脑电预测模型加载成功
✓ Suno API 音乐生成器初始化成功
启动 EEG-Music Web Server
访问地址: http://localhost:5000
```

在浏览器中打开 `http://localhost:5000` 即可使用。

### 5. 使用流程

1. 准备一个 EEG 数据的 CSV 文件（16 通道，采样率 202Hz）
2. 在 Web 界面点击上传文件
3. 等待系统分析情绪并生成音乐（约 1-5 分钟）
4. 在页面上查看情绪分析结果，在线播放或下载音乐

---

## 🔬 模型训练（可选）

如果你想重新训练模型：

### 1. 准备数据集

将 EEG 数据按以下目录结构组织：
```
dataset/
└── train/
    ├── Alert/
    │   ├── file1.csv
    │   └── file2.csv
    ├── Nervous/
    │   ├── file1.csv
    │   └── file2.csv
    └── Relaxed/
        ├── file1.csv
        └── file2.csv
```

**CSV 文件格式**：
- 第 1 行：表头（会被跳过）
- 第 1 列：时间戳（会被忽略）
- 第 2-17 列：16 个 EEG 通道数据
- 采样率：202 Hz

### 2. 执行训练

```bash
python main.py
```

训练过程会：
- 自动将数据按 80%/20% 划分为训练集和验证集
- 使用带通滤波（0.5-40Hz）和 Z-score 标准化进行预处理
- 使用滑动窗口（404 点，步长 202 点）切分样本
- 训练最多 100 个 epoch（早停 patience=15）
- 自动保存最佳模型到 `best_model.pth`
- 生成 `training_curves.png` 和 `confusion_matrix.png`

---

## 📊 脑电波分析（可选）

使用 `EEG_wave_change_analysis.py` 分析音乐干预前后的脑电波变化：

```bash
python EEG_wave_change_analysis.py
```

修改脚本末尾的文件路径来指定干预前后的 EEG 数据文件：
```python
file_before = "./dataset/train/Alert/xxx.csv"    # 干预前
file_after = "./dataset/train/Relaxed/xxx.csv"   # 干预后
```

该脚本会：
- 计算 α 波（8-13Hz）和 β 波（13-30Hz）的能量随时间变化趋势
- 生成对比图并自动计算变化率
- 给出干预效果结论

---

## 🔑 三种情绪状态说明

| 情绪 | 英文 | 含义 | 音乐干预策略 |
|------|------|------|-------------|
| 警觉 | Alert | 大脑处于警觉/专注状态 | 轻松爵士乐，帮助平缓过渡到放松 |
| 焦虑 | Nervous | 大脑处于焦虑/紧张状态 | 温暖治愈的钢琴旋律，稳定情绪 |
| 放松 | Relaxed | 大脑已进入放松状态 | 温暖舒适的轻音乐，深度助眠 |

---

## ⚙️ 技术栈

| 技术 | 用途 |
|------|------|
| **PyTorch** | 深度学习模型训练与推理 |
| **CNN (1D Conv)** | 提取 EEG 信号的局部时域特征 |
| **Transformer Encoder** | 捕捉长距离时间依赖关系 |
| **Flask** | 轻量级 Web 服务框架 |
| **Suno API (V5)** | AI 音乐生成（云端服务） |
| **SciPy** | 数字信号处理（带通滤波、功率谱分析） |
| **HTML/CSS/JS** | 前端用户界面 |

---

## 📝 注意事项

1. **API 密钥安全**：请勿将包含真实 API 密钥的 `config.py` 上传到公开仓库。
2. **模型一致性**：`Real_Time_EEG_Predictor.py` 中的模型定义必须与 `main.py` 中的完全一致，否则无法加载权重。
3. **采样率**：本项目的 EEG 采样率固定为 **202 Hz**，使用不同采样率的数据需要重新调整参数。
4. **Suno API 费用**：Suno API 为付费服务，每次调用会生成 2 首音乐，请注意用量和费用。
5. **GPU 加速**：推荐使用 CUDA GPU 进行训练；推理阶段 CPU 也能正常运行。
6. **生产部署**：请勿在生产环境中使用 Flask 自带的开发服务器，建议使用 Gunicorn + Nginx。
