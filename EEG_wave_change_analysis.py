"""
EEG α波和β波时间变化分析脚本
用于观察音乐干预前后α波和β波的变化

使用说明:
1. 在脚本末尾指定要分析的EEG数据文件路径
2. 运行脚本，自动生成α波和β波随时间变化的图表
3. 可以对比多个文件（如干预前、干预后）

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_eeg_data(csv_file):
    """
    加载EEG数据（16通道）
    
    参数:
        csv_file: CSV文件路径
    
    返回:
        eeg_data: numpy数组，形状为(时间点数, 16通道)
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"文件不存在: {csv_file}")
    
    # 读取CSV文件（跳过第一行标题，读取第2-17列的16个通道）
    df = pd.read_csv(csv_file, skiprows=1, usecols=range(1, 17))
    eeg_data = df.values
    
    print(f"✓ 数据加载成功: {os.path.basename(csv_file)}")
    print(f"  形状: {eeg_data.shape} (时间点数={eeg_data.shape[0]}, 通道数={eeg_data.shape[1]})")
    
    return eeg_data


def preprocess_eeg(eeg_data, sampling_rate=202, lowcut=0.5, highcut=40):
    """
    预处理EEG数据：带通滤波 + 标准化
    
    参数:
        eeg_data: 原始EEG数据
        sampling_rate: 采样率(Hz)
        lowcut: 低频截止(Hz)
        highcut: 高频截止(Hz)
    
    返回:
        filtered_data: 预处理后的数据
    """
    # 带通滤波
    nyquist = sampling_rate / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, eeg_data, axis=0)
    
    # Z-score标准化
    mean = np.mean(filtered_data, axis=0, keepdims=True)
    std = np.std(filtered_data, axis=0, keepdims=True)
    filtered_data = (filtered_data - mean) / (std + 1e-8)
    
    return filtered_data


def calculate_band_power_timeseries(eeg_data, sampling_rate=202, 
                                    band_range=(8, 13), 
                                    window_size=4, 
                                    overlap=0.5,
                                    channels='all'):
    """
    计算指定频段能量随时间的变化
    
    参数:
        eeg_data: 预处理后的EEG数据 (时间点数, 通道数)
        sampling_rate: 采样率(Hz)
        band_range: 频段范围，例如 (8, 13) 表示α波
        window_size: 滑动窗口大小(秒)
        overlap: 窗口重叠比例 (0-1)
        channels: 要分析的通道
                 - 'all': 所有16个通道的平均
                 - 整数: 单个通道索引 (0-15)
                 - 列表: 多个通道索引的平均，如 [0, 1, 2]
    
    返回:
        time_points: 时间点数组(秒)
        power_timeseries: 对应时间点的频段能量
    """
    # 选择通道
    if channels == 'all':
        # 使用所有16个通道的平均
        data_to_analyze = np.mean(eeg_data, axis=1)
    elif isinstance(channels, int):
        # 单个通道
        data_to_analyze = eeg_data[:, channels]
    elif isinstance(channels, list):
        # 多个通道的平均
        data_to_analyze = np.mean(eeg_data[:, channels], axis=1)
    else:
        raise ValueError("channels参数必须是'all'、整数或列表")
    
    # 计算窗口参数
    window_samples = int(window_size * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))
    
    # 滑动窗口计算
    time_points = []
    power_timeseries = []
    
    for start in range(0, len(data_to_analyze) - window_samples, step_samples):
        end = start + window_samples
        window_data = data_to_analyze[start:end]
        
        # 计算该窗口的功率谱
        freqs, power = signal.welch(window_data, fs=sampling_rate, nperseg=min(256, window_samples))
        
        # 提取频段能量
        idx_band = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])
        band_power = np.mean(power[idx_band])
        
        # 记录时间点（窗口中心时刻）
        time_point = (start + end) / 2 / sampling_rate
        
        time_points.append(time_point)
        power_timeseries.append(band_power)
    
    return np.array(time_points), np.array(power_timeseries)


def plot_alpha_beta_changes(csv_file, 
                            sampling_rate=202,
                            window_size=4,
                            overlap=0.75,
                            channels='all',
                            output_file=None):
    """
    绘制α波和β波随时间变化的图表
    
    参数:
        csv_file: EEG数据文件路径
        sampling_rate: 采样率(Hz)
        window_size: 滑动窗口大小(秒)
        overlap: 窗口重叠比例
        channels: 要分析的通道 ('all', 整数, 或列表)
        output_file: 输出图片文件路径（可选）
    """
    print(f"\n{'='*70}")
    print(f"分析文件: {os.path.basename(csv_file)}")
    print(f"{'='*70}")
    
    # 1. 加载数据
    eeg_data = load_eeg_data(csv_file)
    
    # 2. 预处理
    print("✓ 预处理中...")
    filtered_data = preprocess_eeg(eeg_data, sampling_rate)
    
    # 3. 计算α波能量时间序列
    print("✓ 计算α波(8-13 Hz)能量变化...")
    time_alpha, power_alpha = calculate_band_power_timeseries(
        filtered_data, 
        sampling_rate=sampling_rate,
        band_range=(8, 13),  # α波
        window_size=window_size,
        overlap=overlap,
        channels=channels
    )
    
    # 4. 计算β波能量时间序列
    print("✓ 计算β波(13-30 Hz)能量变化...")
    time_beta, power_beta = calculate_band_power_timeseries(
        filtered_data, 
        sampling_rate=sampling_rate,
        band_range=(13, 30),  # β波
        window_size=window_size,
        overlap=overlap,
        channels=channels
    )
    
    # 5. 绘图
    print("✓ 绘制图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # α波图
    ax1.plot(time_alpha, power_alpha, 'g-', linewidth=2, label='α波 (8-13 Hz)')
    ax1.fill_between(time_alpha, power_alpha, alpha=0.3, color='green')
    ax1.set_ylabel('α波能量', fontsize=12, fontweight='bold')
    ax1.set_title(f'α波能量随时间变化', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息
    alpha_mean = np.mean(power_alpha)
    alpha_std = np.std(power_alpha)
    ax1.axhline(y=alpha_mean, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7, label=f'平均值={alpha_mean:.4f}')
    ax1.fill_between(time_alpha, alpha_mean - alpha_std, alpha_mean + alpha_std, 
                     color='green', alpha=0.1, label=f'±1标准差')
    ax1.legend(loc='upper right', fontsize=9)
    
    # β波图
    ax2.plot(time_beta, power_beta, 'orange', linewidth=2, label='β波 (13-30 Hz)')
    ax2.fill_between(time_beta, power_beta, alpha=0.3, color='orange')
    ax2.set_xlabel('时间 (秒)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('β波能量', fontsize=12, fontweight='bold')
    ax2.set_title(f'β波能量随时间变化', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息
    beta_mean = np.mean(power_beta)
    beta_std = np.std(power_beta)
    ax2.axhline(y=beta_mean, color='darkorange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'平均值={beta_mean:.4f}')
    ax2.fill_between(time_beta, beta_mean - beta_std, beta_mean + beta_std, 
                     color='orange', alpha=0.1, label=f'±1标准差')
    ax2.legend(loc='upper right', fontsize=9)
    
    # 通道信息
    if channels == 'all':
        channel_info = "所有16通道平均"
    elif isinstance(channels, int):
        channel_info = f"通道CH{channels}"
    else:
        channel_info = f"通道{channels}平均"
    
    plt.suptitle(f'EEG α波和β波时间变化分析\n文件: {os.path.basename(csv_file)} | {channel_info}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # 保存图片
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存: {output_file}")
    
    # 打印统计信息
    print(f"\n{'='*70}")
    print("统计结果:")
    print(f"{'='*70}")
    print(f"α波 (8-13 Hz):")
    print(f"  平均值: {alpha_mean:.6f}")
    print(f"  标准差: {alpha_std:.6f}")
    print(f"  最大值: {np.max(power_alpha):.6f} (时刻: {time_alpha[np.argmax(power_alpha)]:.2f}秒)")
    print(f"  最小值: {np.min(power_alpha):.6f} (时刻: {time_alpha[np.argmin(power_alpha)]:.2f}秒)")
    print(f"\nβ波 (13-30 Hz):")
    print(f"  平均值: {beta_mean:.6f}")
    print(f"  标准差: {beta_std:.6f}")
    print(f"  最大值: {np.max(power_beta):.6f} (时刻: {time_beta[np.argmax(power_beta)]:.2f}秒)")
    print(f"  最小值: {np.min(power_beta):.6f} (时刻: {time_beta[np.argmin(power_beta)]:.2f}秒)")
    print(f"\nα/β比率: {alpha_mean/beta_mean:.4f}")
    print(f"{'='*70}\n")
    
    return fig, (time_alpha, power_alpha), (time_beta, power_beta)


def compare_before_after(file_before, file_after,
                        sampling_rate=202,
                        window_size=4,
                        overlap=0.75,
                        channels='all',
                        output_file=None):
    """
    对比音乐干预前后的α波和β波变化
    
    参数:
        file_before: 干预前的EEG数据文件
        file_after: 干预后的EEG数据文件
        sampling_rate: 采样率
        window_size: 窗口大小
        overlap: 重叠比例
        channels: 分析的通道
        output_file: 输出文件
    """
    print(f"\n{'#'*70}")
    print("# 音乐干预前后对比分析")
    print(f"{'#'*70}\n")
    
    # 分析干预前
    print("【1/2】分析干预前数据...")
    eeg_before = load_eeg_data(file_before)
    filtered_before = preprocess_eeg(eeg_before, sampling_rate)
    
    time_alpha_before, power_alpha_before = calculate_band_power_timeseries(
        filtered_before, sampling_rate, (8, 13), window_size, overlap, channels)
    time_beta_before, power_beta_before = calculate_band_power_timeseries(
        filtered_before, sampling_rate, (13, 30), window_size, overlap, channels)
    
    # 分析干预后
    print("\n【2/2】分析干预后数据...")
    eeg_after = load_eeg_data(file_after)
    filtered_after = preprocess_eeg(eeg_after, sampling_rate)
    
    time_alpha_after, power_alpha_after = calculate_band_power_timeseries(
        filtered_after, sampling_rate, (8, 13), window_size, overlap, channels)
    time_beta_after, power_beta_after = calculate_band_power_timeseries(
        filtered_after, sampling_rate, (13, 30), window_size, overlap, channels)
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
    # α波对比
    axes[0, 0].plot(time_alpha_before, power_alpha_before, 'b-', linewidth=2, label='干预前', alpha=0.7)
    axes[0, 0].fill_between(time_alpha_before, power_alpha_before, alpha=0.2, color='blue')
    axes[0, 0].set_ylabel('α波能量', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('α波 - 干预前', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.mean(power_alpha_before), color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
    
    axes[0, 1].plot(time_alpha_after, power_alpha_after, 'g-', linewidth=2, label='干预后', alpha=0.7)
    axes[0, 1].fill_between(time_alpha_after, power_alpha_after, alpha=0.2, color='green')
    axes[0, 1].set_ylabel('α波能量', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('α波 - 干预后', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=np.mean(power_alpha_after), color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # β波对比
    axes[1, 0].plot(time_beta_before, power_beta_before, 'r-', linewidth=2, label='干预前', alpha=0.7)
    axes[1, 0].fill_between(time_beta_before, power_beta_before, alpha=0.2, color='red')
    axes[1, 0].set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('β波能量', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('β波 - 干预前', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=np.mean(power_beta_before), color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
    
    axes[1, 1].plot(time_beta_after, power_beta_after, 'orange', linewidth=2, label='干预后', alpha=0.7)
    axes[1, 1].fill_between(time_beta_after, power_beta_after, alpha=0.2, color='orange')
    axes[1, 1].set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('β波能量', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('β波 - 干预后', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=np.mean(power_beta_after), color='darkorange', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.suptitle('音乐干预前后 α波和β波对比', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ 对比图已保存: {output_file}")
    
    # 统计对比
    print(f"\n{'='*70}")
    print("干预前后统计对比:")
    print(f"{'='*70}")
    
    alpha_mean_before = np.mean(power_alpha_before)
    alpha_mean_after = np.mean(power_alpha_after)
    alpha_change = ((alpha_mean_after - alpha_mean_before) / alpha_mean_before) * 100
    
    beta_mean_before = np.mean(power_beta_before)
    beta_mean_after = np.mean(power_beta_after)
    beta_change = ((beta_mean_after - beta_mean_before) / beta_mean_before) * 100
    
    print(f"\nα波变化:")
    print(f"  干预前平均: {alpha_mean_before:.6f}")
    print(f"  干预后平均: {alpha_mean_after:.6f}")
    print(f"  变化率: {alpha_change:+.2f}% {'↑ 增强' if alpha_change > 0 else '↓ 减弱'}")
    
    print(f"\nβ波变化:")
    print(f"  干预前平均: {beta_mean_before:.6f}")
    print(f"  干预后平均: {beta_mean_after:.6f}")
    print(f"  变化率: {beta_change:+.2f}% {'↑ 增强' if beta_change > 0 else '↓ 减弱'}")
    
    print(f"\n结论:")
    if alpha_change > 5 and beta_change < -5:
        print(f"  ✓ α波增强 {abs(alpha_change):.1f}%，β波下降 {abs(beta_change):.1f}% → 放松效果显著")
    elif alpha_change > 5:
        print(f"  ✓ α波增强 {abs(alpha_change):.1f}% → 出现放松趋势")
    elif beta_change < -5:
        print(f"  ✓ β波下降 {abs(beta_change):.1f}% → 压力减轻")
    else:
        print(f"  - 变化不明显，可能需要更长时间的音乐干预")
    
    print(f"{'='*70}\n")
    
    return fig


# ============================================================================
# 主程序：在这里指定要分析的EEG数据文件
# ============================================================================

if __name__ == "__main__":
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         EEG α波和β波时间变化分析                                 ║
    ║         用于观察音乐干预前后的脑电波形变化                        ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ========================================================================
    # 配置参数
    # ========================================================================
    
    # 采样率 (Hz)
    SAMPLING_RATE = 202
    
    # 滑动窗口大小（秒）- 控制时间分辨率
    # 较大的值(如5秒)会使曲线更平滑，较小的值(如2秒)会捕捉更多细节
    WINDOW_SIZE = 4
    
    # 窗口重叠比例 (0-1) - 越大曲线越平滑
    OVERLAP = 0.75
    
    # 要分析的通道
    # - 'all': 所有16个通道的平均值
    # - 0-15: 单个通道，如 0 表示CH0
    # - [0,1,2]: 多个通道的平均，如前3个通道
    CHANNELS = 'all'
    
    # ========================================================================
    # 方式1: 分析单个文件
    # ========================================================================
    """
    # 指定要分析的EEG数据文件路径
    eeg_file = "./dataset/test/Stressful_communication.csv"
    
    # 输出文件路径
    output_file = "./alpha_beta_analysis.png"
    
    # 执行分析
    plot_alpha_beta_changes(
        csv_file=eeg_file,
        sampling_rate=SAMPLING_RATE,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        channels=CHANNELS,
        output_file=output_file
    )
    
    plt.show()
    """


    # ========================================================================
    # 方式2: 对比干预前后（如果有两个文件）
    # ========================================================================
    
    # 取消下面的注释来使用对比功能
    file_before = "./dataset/train/Alert/CurveData_3_Shakespere_English_Test.csv"  # 干预前
    file_after = "./dataset/train/Relaxed/CurveData_1_Conversation.csv"          # 干预后
    compare_before_after(
        file_before=file_before,
        file_after=file_after,
        sampling_rate=SAMPLING_RATE,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        channels=CHANNELS,
        output_file="./intervention_comparison.png"
    )
    
    plt.show()

    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  分析完成！                                                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
