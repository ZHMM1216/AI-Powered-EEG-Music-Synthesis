# 导入 torch 库，这是 PyTorch 的核心库
import torch
# 导入 torch.nn 模块，其中包含构建神经网络所需的各种类和函数，化名为 nn
import torch.nn as nn
# 从 torch.utils.data 模块导入 Dataset 和 DataLoader 类，用于处理和加载数据
from torch.utils.data import Dataset, DataLoader

# 从 scikit-learn 库中导入 LabelEncoder 类，用于将文本标签转换为数字
from sklearn.preprocessing import LabelEncoder
# 导入 pandas 库，用于数据处理和 CSV 文件操作，化名为 pd
import pandas as pd
# 导入 numpy 库，用于进行高效的数值计算，化名为 np
import numpy as np
# 导入 os 库，用于与操作系统进行交互，如文件路径操作
import os
# 从 tqdm 库中导入 tqdm 函数，用于在循环中显示进度条
from tqdm import tqdm
# 从 scipy.signal 模块导入 butter 和 filtfilt 函数，用于设计和应用数字滤波器
from scipy.signal import butter, filtfilt
# 导入 matplotlib.pyplot 模块，用于数据可视化和绘图，化名为 plt
import matplotlib.pyplot as plt
# 从 scikit-learn.metrics 模块导入 confusion_matrix 和 ConfusionMatrixDisplay，用于计算和可视化混淆矩阵
from sklearn.metrics import  ConfusionMatrixDisplay # 新增：导入混淆矩阵相关库
# 从 torch.optim.lr_scheduler 模块导入 ReduceLROnPlateau，用于根据验证指标动态调整学习率
from torch.optim.lr_scheduler import ReduceLROnPlateau # 在文件顶部或函数内部导入



# 定义一个带通滤波器函数
def bandpass_filter(data, sampling_rate, lowcut, highcut):
    # 计算奈奎斯特频率，即采样率的一半
    nyq = 0.5 * sampling_rate
    # 将截止频率从赫兹（Hz）单位归一化到奈奎斯特频率
    low = lowcut / nyq
    # 将截止频率从赫兹（Hz）单位归一化到奈奎斯特频率
    high = highcut / nyq
    # 设计一个4阶的巴特沃斯（Butterworth）带通滤波器，返回滤波器的分子（b）和分母（a）系数
    b, a = butter(4, [low, high], btype='band')
    # 使用 filtfilt 函数对数据进行零相位滤波，axis=0 表示沿着列（每个通道）进行滤波
    y = filtfilt(b, a, data, axis=0)
    # 返回滤波后的数据
    return y


# 定义一个继承自 torch.utils.data.Dataset 的自定义数据集类 EEGDataset
class EEGDataset(Dataset):
    # __init__ 是类的构造函数，在创建类的实例时被调用
    # root_folder: 数据集的根目录；window_size: 滑动窗口的大小；stride: 窗口滑动的步长；sampling_rate: 采样率
    def __init__(self, root_folder, window_size=404, stride=202, sampling_rate=202):
        # 初始化用于存储处理后数据和标签的列表
        self.data, self.labels = [], []
        # 获取根目录下所有子文件夹（即类别名称），并排序以确保一致性
        self.classes = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
        # 使用 scikit-learn 的 LabelEncoder 将文本类别名称（如 "class1", "class2"）转换为数字（如 0, 1）
        self.label_encoder = LabelEncoder().fit(self.classes)
        # 存储脑电信号的采样率
        self.sampling_rate = sampling_rate

        # 遍历每一个类别文件夹
        # enumerate 同时提供索引 (label_idx) 和值 (label)
        for label_idx, label in enumerate(self.classes):
            # 构建当前类别的完整路径
            class_path = os.path.join(root_folder, label)
            # 遍历类别文件夹中的每一个文件
            for file in os.listdir(class_path):
                # 只处理 .csv 文件
                if file.endswith('.csv'):
                    # 构建当前文件的完整路径
                    file_path = os.path.join(class_path, file)
                    # 使用 try-except 块来捕获和处理在文件处理过程中可能发生的错误，增强代码的健壮性
                    try:
                        # --- 数据处理流程 ---
                        
                        # 步骤 1: 加载数据
                        # 使用 pandas 读取 CSV 文件，跳过第一行（通常是表头），只读取第1到16列的EEG数据，并将其转换为 float64 类型的 numpy 数组
                        full_data = pd.read_csv(file_path, skiprows=1, usecols=range(1, 17)).values.astype(np.float64)

                        # 步骤 2: 滤波
                        # 修正后的代码行
                        # 对加载的数据应用带通滤波器，保留0.5Hz到40Hz之间的信号
                        full_data = bandpass_filter(full_data, sampling_rate=self.sampling_rate, lowcut=0.5, highcut=40)

                        # 步骤 3: Z-score 标准化 (带安全检查)
                        # 遍历数据的每一列（即每一个EEG通道）
                        for i in range(full_data.shape[1]):
                            # 计算当前列的平均值
                            mean = np.mean(full_data[:, i]) 
                            # 计算当前列的标准差
                            std = np.std(full_data[:, i])
                            # 检查标准差是否大于一个很小的值（1e-8），以避免除以零的错误
                            if std > 1e-8:
                                # 如果标准差有效，则进行 Z-score 标准化：(值 - 平均值) / 标准差
                                full_data[:, i] = (full_data[:, i] - mean) / std
                            # 如果标准差接近于零（说明该通道数据没有变化），则将所有值设为0
                            else:
                                full_data[:, i] = 0.0

                        # 步骤 4: 激进的裁剪 (在窗口化之前)
                        # 使用 np.clip 将数据中的所有值限制在 -20.0 到 20.0 的范围内，以去除极端异常值
                        np.clip(full_data, -20.0, 20.0, out=full_data)
                        
                        # 步骤 5: 滑动窗口
                        # 初始化滑动窗口的起始位置
                        start = 0
                        # 当窗口的结束位置不超过数据总长度时，循环继续
                        while start + window_size <= len(full_data):
                            # 从 full_data 中切片，提取一个窗口的数据
                            window = full_data[start : start + window_size]
                            # 将提取的窗口数据添加到 self.data 列表中
                            self.data.append(window)
                            # 将当前窗口对应的数字标签添加到 self.labels 列表中
                            self.labels.append(label_idx)
                            # 将窗口的起始位置向前移动一个步长（stride）
                            start += stride
                            
                    # 如果在 try 块中发生任何异常
                    except Exception as e:
                        # 打印错误信息，包括出错的文件路径和具体的异常内容
                        print(f"!!! ERROR processing file {file_path}: {e}")

        # 步骤 6: 最终转换和清理
        # 将存储窗口数据的列表转换为 float32 类型的 numpy 数组，以节省内存并符合模型输入要求
        self.data = np.array(self.data, dtype=np.float32)
        # 将存储标签的列表转换为 int64 类型的 numpy 数组
        self.labels = np.array(self.labels, dtype=np.int64)

        # 检查处理后的数据中是否包含任何 NaN (非数字) 或 Inf (无穷大) 的值
        if np.isnan(self.data).any() or np.isinf(self.data).any():
            # 如果存在，打印警告信息
            print("\n!!! WARNING: NaNs or Infs found AFTER processing. Cleaning them now.")
            # 使用 np.nan_to_num 函数将所有 NaN 和 Inf 值替换为 0.0，确保数据纯净
            self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)
        
    # 定义 __len__ 方法，它返回数据集中样本的总数
    def __len__(self):
        # 返回 self.data 列表的长度
        return len(self.data)

    # 定义 __getitem__ 方法，它根据给定的索引 (idx) 获取单个数据样本及其标签
    def __getitem__(self, idx):
        # 将索引为 idx 的 numpy 数组数据转换为 PyTorch 张量，并返回它和对应的标签张量
        return torch.from_numpy(self.data[idx]), torch.from_numpy(np.array(self.labels[idx]))

# --- MODEL CLASS WITH STABLE INITIALIZATION ---
# 定义一个名为 EEGBiFormer 的神经网络模型类，它继承自 nn.Module
class EEGBiFormer(nn.Module):
    # 类的构造函数，定义模型的结构
    def __init__(self, num_classes, in_channels=16, dim=256):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 定义一个批量归一化层，用于对输入的16个通道进行归一化
        self.input_norm = nn.BatchNorm1d(in_channels)
        # 定义一个序列模块（nn.Sequential），作为特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一个一维卷积层：输入通道16，输出通道64，卷积核大小9，填充4，无偏置
            nn.Conv1d(in_channels, 64, kernel_size=9, padding=4, bias=False),
            # 批量归一化层，对64个通道进行归一化
            nn.BatchNorm1d(64),
            # ReLU 激活函数
            nn.ReLU(),
            # 第二个一维卷积层：输入通道64，输出通道128
            nn.Conv1d(64, 128, kernel_size=9, padding=4, bias=False),
            # 批量归一化层，对128个通道进行归一化
            nn.BatchNorm1d(128),
            # ReLU 激活函数
            nn.ReLU(),
            # 第三个一维卷积层：输入通道128，输出通道 dim (默认为256)
            nn.Conv1d(128, dim, kernel_size=9, padding=4, bias=False),
            # 批量归一化层，对 dim 个通道进行归一化
            nn.BatchNorm1d(dim),
            # ReLU 激活函数
            nn.ReLU()
        )
        # 定义一个 Transformer 编码器模块
        self.transformer = nn.TransformerEncoder(
            # Transformer 编码器层：模型维度(d_model)为dim，8个注意力头，前馈网络维度为dim*4，dropout率为0.2，批次维度优先
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim*4, dropout=0.2, batch_first=True),
            # 编码器层数
            num_layers=2
        )
        # 定义一个序列模块，作为分类头
        self.head = nn.Sequential(
            # 第一个全连接层：输入维度dim，输出维度dim//2
            nn.Linear(dim, dim//2),
            # ReLU 激活函数
            nn.ReLU(),
            # Dropout 层，防止过拟合
            nn.Dropout(0.2),
            # 第二个全连接层：输入维度dim//2，输出维度为类别数
            nn.Linear(dim//2, num_classes)
        )
        # 对模型的所有模块应用自定义的权重初始化方法
        self.apply(self._initialize_weights)
        # 打印信息，表示权重已初始化
        print("Model weights initialized with custom stable method.")

    # 定义一个内部方法，用于自定义权重初始化
    def _initialize_weights(self, m):
        # 如果模块是一维卷积层
        if isinstance(m, nn.Conv1d):
            # 使用 Kaiming 正态分布初始化权重，适用于 ReLU 激活函数
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 如果模块是批量归一化层
        elif isinstance(m, nn.BatchNorm1d):
            # 将其权重初始化为1
            nn.init.constant_(m.weight, 1)
            # 将其偏置初始化为0
            nn.init.constant_(m.bias, 0)
        # 如果模块是全连接层
        elif isinstance(m, nn.Linear):
            # 使用均值为0，标准差为0.01的正态分布初始化权重
            nn.init.normal_(m.weight, 0, 0.01)
            # 如果存在偏置项，则将其初始化为0
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    # 定义模型的前向传播逻辑
    def forward(self, x):
        # 将输入张量中的 NaN 和 Inf 值替换为0，增强数值稳定性
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # 将输入张量的值裁剪到[-20.0, 20.0]范围内，防止极端值影响
        x = torch.clamp(x, min=-20.0, max=20.0)
        # 交换第1和第2个维度，以匹配 nn.Conv1d 的输入格式 (N, C, L)
        x = x.transpose(1, 2)
        # 应用输入批量归一化
        x = self.input_norm(x)
        # 通过特征提取器
        x = self.feature_extractor(x)
        # 再次交换维度，以匹配 Transformer 的输入格式 (N, L, C)
        x = x.transpose(1, 2)
        # 通过 Transformer 编码器
        x = self.transformer(x)
        # 对序列维度（时间步）取平均值，实现全局平均池化
        x = x.mean(dim=1)
        # 通过分类头得到最终的输出
        return self.head(x)

# --- PLOTTING FUNCTIONS ---

# 定义一个函数，用于绘制训练和验证过程中的损失和准确率曲线
def plot_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history):
    # 获取训练的总轮数
    num_epochs = len(train_loss_history)
    # 创建一个从1到总轮数的整数序列，作为x轴
    epochs = range(1, num_epochs + 1)
    # 创建一个大小为15x10的画布
    plt.figure(figsize=(15, 10))
    # 创建一个2x2的子图网格，并选择第一个子图 (左上)
    plt.subplot(2, 2, 1)
    # 绘制训练损失曲线
    plt.plot(epochs, train_loss_history, label='Training Loss', color='blue')
    # 设置子图标题
    plt.title('Training Loss')
    # 设置x轴标签
    plt.xlabel('Epoch')
    # 设置y轴标签
    plt.ylabel('Loss')
    # 显示图例
    plt.legend()
    # 显示网格线
    plt.grid(True)

    # 选择第二个子图 (右上)
    plt.subplot(2, 2, 2)
    # 绘制验证损失曲线
    plt.plot(epochs, val_loss_history, label='Validation Loss', color='red')
    # 设置子图标题
    plt.title('Validation Loss')
    # 设置x轴标签
    plt.xlabel('Epoch')
    # 设置y轴标签
    plt.ylabel('Loss')
    # 显示图例
    plt.legend()
    # 显示网格线
    plt.grid(True)

    # 选择第三个子图 (左下)
    plt.subplot(2, 2, 3)
    # 绘制训练准确率曲线
    plt.plot(epochs, train_acc_history, label='Training Accuracy', color='green')
    # 设置子图标题
    plt.title('Training Accuracy')
    # 设置x轴标签
    plt.xlabel('Epoch')
    # 设置y轴标签
    plt.ylabel('Accuracy')
    # 显示图例
    plt.legend()
    # 显示网格线
    plt.grid(True)

    # 选择第四个子图 (右下)
    plt.subplot(2, 2, 4)
    # 绘制验证准确率曲线
    plt.plot(epochs, val_acc_history, label='Validation Accuracy', color='orange')
    # 设置子图标题
    plt.title('Validation Accuracy')
    # 设置x轴标签
    plt.xlabel('Epoch')
    # 设置y轴标签
    plt.ylabel('Accuracy')
    # 显示图例
    plt.legend()
    # 显示网格线
    plt.grid(True)

    # 自动调整子图参数，使之填充整个图像区域
    plt.tight_layout()
    # 将绘制的图像保存为 'training_curves.png' 文件
    plt.savefig('training_curves.png')
    # 打印成功保存的信息
    print("\n成功将训练曲线图保存为 'training_curves.png'")

# 定义绘制混淆矩阵的函数
def plot_confusion_matrix(model, val_loader, device, class_names):
    """
    加载最佳模型，在验证集上进行预测，并生成/保存混淆矩阵图。
    """
    # 打印提示信息
    print("\nGenerating confusion matrix on validation data with the best model...")
    # 使用 try-except 块来处理可能发生的错误
    try:
        # 加载之前保存的最佳模型权重
        model.load_state_dict(torch.load('best_model.pth'))
        # 将模型移动到指定的设备（CPU或GPU）
        model.to(device)
        # 将模型设置为评估模式，这会关闭 Dropout 等层
        model.eval() 

        # 初始化用于存储所有真实标签和预测标签的列表
        all_labels = []
        all_preds = []

        # 在验证集上收集所有标签和预测
        # 使用 torch.no_grad() 上下文管理器，在该代码块中禁用梯度计算，以节省内存和计算资源
        with torch.no_grad():
            # 遍历验证数据加载器
            for data, labels in val_loader:
                # 将数据和标签移动到指定设备，并确保数据类型正确
                data, labels_tensor = data.to(device).float(), labels.to(device).long()
                # 通过模型进行前向传播，得到输出
                outputs = model(data)
                # 找到输出中概率最大的索引作为预测类别
                preds = outputs.argmax(1)
                
                # 将当前批次的真实标签和预测标签添加到列表中
                all_labels.extend(labels_tensor.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # 使用 scikit-learn 的 ConfusionMatrixDisplay.from_predictions 函数生成并绘制混淆矩阵
        ConfusionMatrixDisplay.from_predictions(
            y_true=all_labels, # 真实标签
            y_pred=all_preds, # 预测标签
            display_labels=class_names, # 坐标轴上显示的类别名称
            cmap=plt.cm.Blues, # 颜色映射方案
            xticks_rotation='horizontal' # x轴标签旋转角度
        )
        
        # 设置混淆矩阵图的标题
        plt.title('Confusion Matrix for Validation Set')
        # 自动调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('confusion_matrix.png')
        # 打印成功保存的信息
        print("Successfully saved confusion matrix to 'confusion_matrix.png'")
    
    # 如果找不到 'best_model.pth' 文件
    except FileNotFoundError:
        # 打印提示信息，并跳过混淆矩阵的生成
        print("Could not find 'best_model.pth'. Skipping confusion matrix generation.")
    # 如果发生其他任何异常
    except Exception as e:
        # 打印错误信息
        print(f"An error occurred during confusion matrix generation: {e}")


# 定义模型训练函数
def train_model(model, train_loader, val_loader, epochs=100, device="cuda"):
    # 获取训练数据集中所有的标签
    train_labels = train_loader.dataset.labels
    # 统计每个类别的样本数量
    class_counts = np.bincount(train_labels, minlength=len(train_loader.dataset.classes))
    # 计算类别权重：权重与类别样本数成反比
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # 对权重进行归一化，使其总和等于类别数
    class_weights = class_weights / class_weights.sum() * len(train_loader.dataset.classes)
    
    # 打印计算出的类别权重
    print(f"Correctly Calculated Class Weights: {class_weights}")

    # 定义交叉熵损失函数，并传入类别权重以处理数据不平衡问题
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # 定义优化器为 AdamW，这是一种带有权重衰减的 Adam 优化器，学习率设置为 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 定义学习率调度器 ---
    # 当'val_loss'停止下降超过2个epoch时，学习率乘以0.05
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.05, patience=2)



    # 初始化用于记录训练和验证过程损失和准确率的列表
    train_loss_history, train_acc_history = [], []
    val_loss_history, val_acc_history = [], []
    
    # 初始化最佳验证准确率
    best_val_acc = 0
    # 初始化用于早停的耐心计数器
    patience_counter = 0
    # 设置早停的耐心值，即连续15个epoch验证准确率没有提升就停止训练
    patience = 15
    # 初始化记录最佳模型是在哪个epoch得到的
    best_epoch = 0

    # 开始训练循环，共进行 epochs 轮
    for epoch in range(epochs):
        # 将模型设置为训练模式
        model.train()
        # 初始化当前epoch的训练损失和训练准确率
        train_loss, train_acc = 0, 0
        # 使用 tqdm 创建一个进度条，用于显示训练进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        # 遍历训练数据加载器中的每一个批次
        for data, labels in pbar:
            # 将数据和标签移动到指定设备，并确保数据类型正确
            data, labels = data.to(device).float(), labels.to(device).long()
            # 模型前向传播，得到输出
            outputs = model(data)
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 检查损失是否为 NaN
            if torch.isnan(loss):
                # 如果是，则打印警告并跳过当前批次
                print(f"\n!!! WARNING: Loss is NaN on batch. Skipping.")
                continue

            # 清空之前的梯度
            optimizer.zero_grad()
            # 反向传播，计算梯度
            loss.backward()
            # 进行梯度裁剪，防止梯度爆炸，将梯度的范数限制在1.0以内
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 更新模型参数
            optimizer.step()

            # 累加当前批次的损失
            train_loss += loss.item()
            # 累加当前批次的准确率
            train_acc += (outputs.argmax(1) == labels).float().mean().item()
            # 在进度条后面显示当前批次的损失值
            pbar.set_postfix(loss=loss.item())

        # 将模型设置为评估模式
        model.eval()
        # 初始化当前epoch的验证损失和验证准确率
        val_loss, val_acc = 0, 0
        # 在 no_grad 上下文中进行验证，以禁用梯度计算
        with torch.no_grad():
            # 遍历验证数据加载器中的每一个批次
            for data, labels in val_loader:
                # 将数据和标签移动到指定设备
                data, labels = data.to(device).float(), labels.to(device).long()
                # 模型前向传播
                outputs = model(data)
                # 检查输出是否包含NaN
                if not torch.isnan(outputs).any():
                    # 计算损失
                    loss = criterion(outputs, labels)
                    # 累加验证损失
                    val_loss += loss.item()
                    # 累加验证准确率
                    val_acc += (outputs.argmax(1) == labels).float().mean().item()
        
        # 计算当前epoch的平均训练损失
        train_loss /= len(train_loader)
        # 计算当前epoch的平均训练准确率
        train_acc /= len(train_loader)
        # 计算当前epoch的平均验证损失（处理分母为0的情况）
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        # 计算当前epoch的平均验证准确率（处理分母为0的情况）
        val_acc /= len(val_loader) if len(val_loader) > 0 else 1

        # 打印当前epoch的训练总结
        print(f"\nEpoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # --- 让调度器根据验证损失调整学习率 ---
        # 调度器根据当前的验证损失来决定是否要降低学习率
        scheduler.step(val_loss)


        # 将当前epoch的各项指标添加到历史记录列表中
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # 早停逻辑：检查当前验证准确率是否超过了历史最佳
        if val_acc > best_val_acc:
            # 如果是，则更新最佳验证准确率
            best_val_acc = val_acc
            # 更新最佳模型所在的epoch
            best_epoch = epoch + 1
            # 保存当前模型的权重参数到 'best_model.pth' 文件
            torch.save(model.state_dict(), 'best_model.pth')
            # 打印保存信息
            print(f"--> New best model saved with accuracy: {best_val_acc:.4f}")
            # 重置耐心计数器
            patience_counter = 0
        # 如果当前验证准确率没有提升
        else:
            # 耐心计数器加一
            patience_counter += 1
            # 如果耐心计数器达到设定的阈值
            if patience_counter >= patience:
                # 打印早停信息并跳出训练循环
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    # 训练结束后，调用函数绘制并保存训练曲线图
    plot_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

    # 打印训练结束的总结信息
    print("\n" + "="*60)
    print("Training Finished Summary")
    print("="*60)
    # 如果 best_epoch 大于 0，说明有模型被保存过
    if best_epoch > 0:
        # 打印最佳模型是在哪个epoch保存的
        print(f"   Best model was saved from Epoch: {best_epoch}")
        # 打印其对应的验证准确率
        print(f"   - Validation Accuracy: {best_val_acc:.4f}")
        # 打印模型保存的路径
        print(f"   - Model weights saved to: 'best_model.pth'")
    # 如果没有模型被保存
    else:
        # 打印提示信息
        print("No model was saved as validation accuracy did not improve during training.")
    print("="*60 + "\n")

    # --- 如果成功保存了模型，则生成混淆矩阵 ---
    # 如果 best_epoch 大于 0，说明有最佳模型
    if best_epoch > 0:
        # 获取类别名称
        class_names = train_loader.dataset.classes
        # 调用函数绘制并保存混淆矩阵
        plot_confusion_matrix(model, val_loader, device, class_names)
    # 如果没有保存最佳模型
    else:
        # 打印提示信息，跳过混淆矩阵的生成
        print("Skipping confusion matrix generation as no best model was saved.")


# 定义主函数
def main():
    # 检查 CUDA 是否可用，如果可用则使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印正在使用的设备
    print(f"Using device: {device}")
    
    # 1. 加载全部数据 (从 dataset/train)
    print("Loading all data from ./dataset/train ...")
    # 这一步会读取所有 CSV 文件
    full_dataset = EEGDataset(root_folder="./dataset/train")

    # 检查总数据量
    if len(full_dataset) == 0:
        print("Dataset is empty. Please check folder paths.")
        return

    # 2. 自动切分验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 使用 random_split 切分
    # generator=torch.Generator().manual_seed(42) 确保每次切分结果一致
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )

    # =============== 关键修正开始 ===============
    # Subset 对象默认没有 labels 和 data 属性，我们需要手动加上
    # 这样后续的代码（打印日志、计算权重）才能正常运行
    
    # 1. 挂载 Labels (用于计算 Class Weights)
    train_dataset.labels = full_dataset.labels[train_dataset.indices]
    val_dataset.labels = full_dataset.labels[val_dataset.indices]
    
    # 2. 挂载 Classes 和 Encoder (用于显示类别名称)
    train_dataset.classes = full_dataset.classes
    val_dataset.classes = full_dataset.classes
    train_dataset.label_encoder = full_dataset.label_encoder
    
    # 3. 挂载 Data (用于 Data Summary 打印日志 - 修复报错的核心)
    train_dataset.data = full_dataset.data[train_dataset.indices]
    val_dataset.data = full_dataset.data[val_dataset.indices]
    
    print(f"Data split completed: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # 3. 打印详细日志 
    print("===================== Data Summary =====================")
    
    # 创建类别映射字典
    class_mapping = dict(zip(full_dataset.label_encoder.classes_, full_dataset.label_encoder.transform(full_dataset.label_encoder.classes_)))
    print(f"Classes found and encoded: {class_mapping}")
    
    print("\n[Train Set]")
    print(f"  - Samples:    {len(train_dataset)}")
    print(f"  - Shape:      {train_dataset.data.shape}") # 之前报错的地方
    print(f"  - Labels:     {np.bincount(train_dataset.labels)}")
    print(f"  - Data Range: Min={train_dataset.data.min():.4f}, Max={train_dataset.data.max():.4f}")

    print("\n[Validation Set]")
    print(f"  - Samples:    {len(val_dataset)}")
    print(f"  - Shape:      {val_dataset.data.shape}")
    print(f"  - Labels:     {np.bincount(val_dataset.labels)}")
    print(f"  - Data Range: Min={val_dataset.data.min():.4f}, Max={val_dataset.data.max():.4f}")
    print("========================================================")

    # 4. 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 获取类别的总数
    num_classes = len(full_dataset.classes)
    
    # 实例化模型
    model = EEGBiFormer(num_classes=num_classes).to(device)
    
    # 开始训练
    train_model(model, train_loader, val_loader, device=device)

# 程序的入口点
# 检查当前脚本是否是作为主程序直接运行
if __name__ == "__main__":
    # 如果是，则调用 main() 函数
    main()