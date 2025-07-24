import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 打开数据集文件
dataset_file = h5py.File("data/GOLD_XYZ_OSC.0001_1024.hdf5", "r")

# 定义所有调制方式
base_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                           'FM', 'GMSK', 'OQPSK']

# 选择要可视化的调制方式（24种）
selected_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                               'FM', 'GMSK', 'OQPSK']

# 获取选择的调制方式的索引
selected_classes_id = [base_modulation_classes.index(cls) for cls in selected_modulation_classes]

# 数据集参数
n_snr = 26  # 26个SNR级别：-20dB到+30dB，步长2dB
n_frames_per_snr = 4096  # 每个SNR级别有4096个帧
n_samples_per_frame = 1024  # 每个帧有1024个I/Q样本
n_frames_per_mod = n_snr * n_frames_per_snr  # 每种调制方式的总帧数 = 106496

# 创建主图 - 显示单个高SNR帧的星座图
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.4, wspace=0.3)

# 选择高SNR（30dB）的数据进行可视化
snr_30db_offset = 25 * n_frames_per_snr  # SNR=30dB是第26个SNR级别（索引25）

# 为每种调制方式绘制星座图
for idx, (class_id, mod_name) in enumerate(zip(selected_classes_id, selected_modulation_classes)):
    # 计算子图位置
    row = idx // 4
    col = idx % 4
    ax = fig.add_subplot(gs[row, col])

    # 计算该调制方式在30dB SNR下的起始索引
    mod_start_idx = class_id * n_frames_per_mod
    snr_30db_start_idx = mod_start_idx + snr_30db_offset

    # 选择一个帧进行可视化
    frame_idx = snr_30db_start_idx + 100  # 选择第100个帧

    # 获取I/Q数据
    # X的形状是 (samples, 1024, 2)，其中最后一维是I和Q
    iq_data = dataset_file['X'][frame_idx]
    i_data = iq_data[:, 0]  # I通道
    q_data = iq_data[:, 1]  # Q通道

    # 绘制星座图
    ax.scatter(i_data, q_data, alpha=0.6, s=10, c='blue', edgecolors='none')

    # 设置图表属性
    ax.set_xlabel('I (In-phase)', fontsize=8)
    ax.set_ylabel('Q (Quadrature)', fontsize=8)
    ax.set_title(f'{mod_name}\n(SNR=30dB, 1 frame)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # 设置轴范围
    max_range = max(np.max(np.abs(i_data)), np.max(np.abs(q_data))) * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    # 添加十字线
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

# 设置总标题
fig.suptitle('Constellation Diagrams for 24 Modulation Types (Single Frame at 30dB SNR)', fontsize=16,
             fontweight='bold')

# 保存图片
plt.savefig('constellation_diagrams_24_modulations_single_frame.png', dpi=300, bbox_inches='tight')
plt.show()

# 关闭文件
dataset_file.close()

print("生成多帧叠加的星座图...")

# 重新打开文件
dataset_file = h5py.File("data/GOLD_XYZ_OSC.0001_1024.hdf5", "r")

# 创建第二个图，显示多个帧叠加的星座图
fig2 = plt.figure(figsize=(20, 16))
gs2 = gridspec.GridSpec(6, 4, figure=fig2, hspace=0.4, wspace=0.3)

# 每种调制方式使用多个帧
n_frames_to_plot = 20  # 叠加20个帧

for idx, (class_id, mod_name) in enumerate(zip(selected_classes_id, selected_modulation_classes)):
    row = idx // 4
    col = idx % 4
    ax = fig2.add_subplot(gs2[row, col])

    # 收集多个帧的I/Q数据
    all_i_data = []
    all_q_data = []

    # 计算该调制方式在30dB SNR下的起始索引
    mod_start_idx = class_id * n_frames_per_mod
    snr_30db_start_idx = mod_start_idx + snr_30db_offset

    # 选择多个帧
    for i in range(n_frames_to_plot):
        frame_idx = snr_30db_start_idx + i * 10  # 每隔10个帧取一个
        iq_data = dataset_file['X'][frame_idx]
        all_i_data.extend(iq_data[:, 0])
        all_q_data.extend(iq_data[:, 1])

    all_i_data = np.array(all_i_data)
    all_q_data = np.array(all_q_data)

    # 绘制星座图
    ax.scatter(all_i_data, all_q_data, alpha=0.1, s=1, c='blue', edgecolors='none')

    # 设置图表属性
    ax.set_xlabel('I (In-phase)', fontsize=8)
    ax.set_ylabel('Q (Quadrature)', fontsize=8)
    ax.set_title(f'{mod_name}\n(SNR=30dB, {n_frames_to_plot} frames)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # 设置轴范围
    max_range = max(np.max(np.abs(all_i_data)), np.max(np.abs(all_q_data))) * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    # 添加十字线
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

fig2.suptitle('Constellation Diagrams for 24 Modulation Types (Multiple Frames at 30dB SNR)', fontsize=16,
              fontweight='bold')
plt.savefig('constellation_diagrams_24_modulations_multi_frames.png', dpi=300, bbox_inches='tight')
plt.show()

dataset_file.close()

# 创建第三个图，显示不同SNR下的星座图对比
print("\n生成不同SNR下的星座图对比...")

dataset_file = h5py.File("data/GOLD_XYZ_OSC.0001_1024.hdf5", "r")

# 选择几种代表性的调制方式
demo_modulations = ['BPSK', 'QPSK', '16QAM', '64QAM']
demo_mod_indices = [base_modulation_classes.index(mod) for mod in demo_modulations]
snr_values = [-10, 0, 10, 30]  # 选择几个代表性的SNR值

fig3 = plt.figure(figsize=(16, 16))
gs3 = gridspec.GridSpec(4, 4, figure=fig3, hspace=0.4, wspace=0.3)

for i, (mod_idx, mod_name) in enumerate(zip(demo_mod_indices, demo_modulations)):
    for j, snr in enumerate(snr_values):
        ax = fig3.add_subplot(gs3[i, j])

        # 计算SNR对应的偏移
        snr_idx = (snr + 20) // 2  # 将SNR值转换为索引
        snr_offset = snr_idx * n_frames_per_snr

        # 计算帧索引
        mod_start_idx = mod_idx * n_frames_per_mod
        frame_idx = mod_start_idx + snr_offset + 100

        # 获取I/Q数据
        iq_data = dataset_file['X'][frame_idx]
        i_data = iq_data[:, 0]
        q_data = iq_data[:, 1]

        # 绘制星座图
        ax.scatter(i_data, q_data, alpha=0.6, s=10, c='blue', edgecolors='none')

        # 设置图表属性
        ax.set_xlabel('I', fontsize=8)
        ax.set_ylabel('Q', fontsize=8)
        ax.set_title(f'{mod_name} @ {snr}dB', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # 统一轴范围以便比较
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # 添加十字线
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

fig3.suptitle('Constellation Diagrams at Different SNR Levels', fontsize=16, fontweight='bold')
plt.savefig('constellation_diagrams_snr_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

dataset_file.close()

print("\n星座图已生成完毕！")
print("生成了三个文件：")
print("1. constellation_diagrams_24_modulations_single_frame.png: 每种调制方式在30dB SNR下的单帧星座图")
print("2. constellation_diagrams_24_modulations_multi_frames.png: 每种调制方式在30dB SNR下的多帧叠加星座图")
print("3. constellation_diagrams_snr_comparison.png: 展示不同SNR对星座图影响的对比图")
print(f"\n每个帧包含 {n_samples_per_frame} 个I/Q样本点")