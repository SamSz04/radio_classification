import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 调制方式标签映射
modulation_labels = {
    0: '4ASK',
    1: '8ASK',
    2: '8PSK',
    3: '16PSK',
    4: '16QAM',
    5: '32PSK',
    6: '32QAM',
    7: '64QAM',
    8: '128QAM',
    9: 'BPSK',
    10: 'OOK',
    11: 'QPSK'
}

# 在模块顶层，仅加载一次
SCALER_PATH = 'F:/大三下/小学期实习/Radio/scaler05.pkl'
MODEL_PATH  = 'F:/大三下/小学期实习/Radio/model05.pkl'
# SCALER_PATH = 'D:/RadioData/scaler1m03.pkl'
# MODEL_PATH  = 'D:/RadioData/model1m03.pkl'

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


def data_predict(feat):
    """直接用已加载好的 model 和 scaler"""
    return model.predict(scaler.transform(feat[None, :]))[0]


def test_and_visualize(mat_file_path, true_label=None):
    """
    测试指定文件并可视化结果

    参数:
    mat_file_path: mat文件路径
    true_label: 真实标签（可选，如果提供将显示在图表中）
    """
    test_data_X = []
    result = []

    # 初始化计数器，包含所有12种调制方式
    cnt = [0] * 12

    # 加载.mat文件
    mat_data = loadmat(mat_file_path)

    frames = mat_data['frames']  # shape (L, 2, N)

    L, _, N = frames.shape
    for i in range(N):
        # 取第 i 帧，shape (L,2)
        fi = frames[:, 0, i]
        # 分离实部和虚部，并扁平化
        feat = np.hstack([fi.real.flatten(), fi.imag.flatten()])
        test_data_X.append(feat)

    # # 提取特征
    # for i in range(len(mat_data['R_matrix'][0][0])):
    #     test_data_X.append([
    #         mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
    #         mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
    #         mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
    #         mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag
    #     ])

    # 预测
    print(f"正在预测 {len(test_data_X)} 个样本...")
    for i in range(len(test_data_X)):
        pred = data_predict(test_data_X[i])
        result.append(pred)
        cnt[pred] += 1

        # 每100个样本打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{len(test_data_X)} 个样本")

    # 打印统计结果
    print("\n预测结果统计:")
    for label, count in enumerate(cnt):
        if count > 0:
            print(f"{modulation_labels[label]}: {count} ({count / len(result) * 100:.2f}%)")

    # 可视化
    visualize_results(cnt, result, mat_file_path, true_label)

    return result, cnt


def visualize_results(cnt, result, file_path, true_label=None):
    """可视化预测结果"""
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 柱状图
    ax1 = axes[0, 0]
    labels = [modulation_labels[i] for i in range(12)]
    x_pos = np.arange(len(labels))

    bars = ax1.bar(x_pos, cnt, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('调制方式')
    ax1.set_ylabel('预测数量')
    ax1.set_title('预测结果分布（柱状图）')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    # 在柱状图上添加数值
    for bar, count in zip(bars, cnt):
        if count > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{count}', ha='center', va='bottom')

    # 如果提供了真实标签，标记出来
    if true_label is not None:
        bars[true_label].set_color('red')
        bars[true_label].set_alpha(1.0)
        ax1.text(0.5, 0.95, f'真实标签: {modulation_labels[true_label]}',
                 transform=ax1.transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 饼图（只显示有预测结果的类别）
    ax2 = axes[0, 1]
    non_zero_indices = [i for i, c in enumerate(cnt) if c > 0]
    non_zero_counts = [cnt[i] for i in non_zero_indices]
    non_zero_labels = [modulation_labels[i] for i in non_zero_indices]

    if non_zero_counts:
        wedges, texts, autotexts = ax2.pie(non_zero_counts, labels=non_zero_labels,
                                           autopct='%1.1f%%', startangle=90)
        ax2.set_title('预测结果分布（饼图）')

    # 3. 时间序列图（显示预测结果随样本的变化）
    ax3 = axes[1, 0]
    ax3.plot(result[:min(1000, len(result))], 'b-', linewidth=0.5)  # 最多显示前1000个样本
    ax3.set_xlabel('样本索引')
    ax3.set_ylabel('预测标签')
    ax3.set_title('预测结果时间序列（前1000个样本）')
    ax3.set_yticks(range(12))
    ax3.set_yticklabels(labels)
    ax3.grid(True, alpha=0.3)

    # 4. 热力图（显示预测的置信度分布）
    ax4 = axes[1, 1]
    # 创建一个简单的热力图显示各类别的预测比例
    heatmap_data = np.array(cnt).reshape(1, -1) / len(result)
    im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_yticks([0])
    ax4.set_yticklabels(['预测比例'])
    ax4.set_title('预测比例热力图')

    # 添加数值标注
    for i in range(12):
        text = ax4.text(i, 0, f'{heatmap_data[0, i]:.2f}',
                        ha="center", va="center", color="black" if heatmap_data[0, i] < 0.5 else "white")

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('预测比例')

    # 添加总标题
    fig.suptitle(f'文件: {os.path.basename(file_path)} - 总样本数: {len(result)}', fontsize=16)

    plt.tight_layout()
    plt.show()


def batch_test_visualization(base_path, file_label_list):
    """批量测试多个文件并可视化对比结果"""
    all_results = {}

    for filename, true_label in file_label_list:
        subfolder = filename.replace('_dp.mat', '')
        mat_file = os.path.join(base_path, subfolder, filename)

        if os.path.exists(mat_file):
            print(f"\n测试文件: {filename} (真实标签: {modulation_labels[true_label]})")
            result, cnt = test_and_visualize(mat_file, true_label)
            all_results[filename] = (result, cnt, true_label)
        else:
            print(f"文件不存在: {mat_file}")

    # 创建对比图
    if all_results:
        create_comparison_plot(all_results)


def create_comparison_plot(all_results):
    """创建多个文件的对比图"""
    fig, ax = plt.subplots(figsize=(15, 8))

    files = list(all_results.keys())
    n_files = len(files)
    n_classes = 12

    # 创建混淆矩阵风格的图
    confusion_data = np.zeros((n_files, n_classes))

    for i, (filename, (result, cnt, true_label)) in enumerate(all_results.items()):
        total = len(result)
        for j in range(n_classes):
            confusion_data[i, j] = cnt[j] / total if total > 0 else 0

    im = ax.imshow(confusion_data, cmap='Blues', aspect='auto')

    # 设置标签
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([modulation_labels[i] for i in range(n_classes)], rotation=45, ha='right')
    ax.set_yticks(range(n_files))
    ax.set_yticklabels([f"{os.path.basename(f).replace('_dp.mat', '')}" for f in files])

    # 添加数值标注
    for i in range(n_files):
        for j in range(n_classes):
            text = ax.text(j, i, f'{confusion_data[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if confusion_data[i, j] > 0.5 else "black")

    # 标记真实标签
    for i, (filename, (_, _, true_label)) in enumerate(all_results.items()):
        ax.add_patch(plt.Rectangle((true_label - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=3))

    ax.set_title('多文件预测结果对比（红框表示真实标签）')
    ax.set_xlabel('预测的调制方式')
    ax.set_ylabel('测试文件')

    plt.colorbar(im, ax=ax, label='预测比例')
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 测试单个文件
    mat_file_path = r'D:\RadioData\20250714\1m\8psk\8psk_frames.mat'
    result, cnt = test_and_visualize(mat_file_path, true_label=2)

    # 或者批量测试多个文件
    # base_path = r'D:\RadioData\20250707'
    # file_label_list = [
    #     ('2psk_dp.mat',   0),
    #     ('4ask_dp.mat',   1),
    #     ('8ask_dp.mat',   2),
    #     # ... 其他文件
    # ]
    # batch_test_visualization(base_path, file_label_list)