import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.signal import hilbert
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SignalPreprocessor:
    """信号预处理器"""
    @staticmethod
    def estimate_and_correct_cfo(signal, fs=1.0):
        """
        估计并校正载波频偏
        使用自相关方法估计频偏
        """
        # 计算自相关
        N = len(signal)
        lag = N // 4  # 使用1/4长度的延迟

        # 计算延迟自相关
        r = np.sum(signal[lag:] * np.conj(signal[:-lag]))

        # 估计频偏
        freq_offset = np.angle(r) / (2 * np.pi * lag)

        # 生成校正信号
        n = np.arange(N)
        correction = np.exp(-1j * 2 * np.pi * freq_offset * n)

        # 应用校正
        corrected_signal = signal * correction

        return corrected_signal, freq_offset

    @staticmethod
    def estimate_and_correct_phase(signal, modulation_order=4):
        """
        估计并校正相位偏移
        使用M次方法
        """
        # M次方法
        signal_m = signal ** modulation_order

        # 估计平均相位
        avg_phase = np.angle(np.mean(signal_m)) / modulation_order

        # 校正相位
        corrected_signal = signal * np.exp(-1j * avg_phase)

        return corrected_signal, avg_phase

    @staticmethod
    def normalize_amplitude(signal):
        """幅度归一化"""
        # 计算平均功率
        avg_power = np.mean(np.abs(signal) ** 2)

        # 归一化到单位功率
        if avg_power > 0:
            normalized_signal = signal / np.sqrt(avg_power)
        else:
            normalized_signal = signal

        return normalized_signal

    @staticmethod
    def preprocess_signal(complex_signal):
        """
        完整的信号预处理流程
        """
        # 1. 载波频偏校正
        signal, cfo = SignalPreprocessor.estimate_and_correct_cfo(complex_signal)

        # 2. 相位校正（尝试不同的调制阶数）
        # 对于不同调制方式，可能需要不同的M值
        best_signal = signal
        min_variance = float('inf')

        for m in [2, 4, 8]:  # 尝试BPSK, QPSK, 8PSK
            corrected, _ = SignalPreprocessor.estimate_and_correct_phase(signal, m)
            # 计算星座点的紧密度
            variance = np.var(np.abs(corrected))
            if variance < min_variance:
                min_variance = variance
                best_signal = corrected

        # 3. 幅度归一化
        normalized_signal = SignalPreprocessor.normalize_amplitude(best_signal)

        return normalized_signal


class ModulationDataset(Dataset):
    """改进的数据集类，包含预处理和数据增强"""
    def __init__(self, X, y, transform=None, augment=False, preprocess=True):
        self.X = X  # 复数信号数组
        self.y = torch.LongTensor(y)
        self.transform = transform
        self.augment = augment
        self.preprocess = preprocess
        self.preprocessor = SignalPreprocessor()

    def __len__(self):
        return len(self.y)

    def augment_signal(self, signal):
        """数据增强"""
        # 随机相位旋转
        phase_shift = np.random.uniform(0, 2 * np.pi)
        signal = signal * np.exp(1j * phase_shift)

        # 随机小幅度频偏
        if np.random.rand() > 0.5:
            freq_offset = np.random.normal(0, 0.005)
            n = np.arange(len(signal))
            signal = signal * np.exp(1j * 2 * np.pi * freq_offset * n)

        # 添加高斯噪声
        snr_db = np.random.uniform(15, 30)  # 15-30 dB SNR
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) +
                                            1j * np.random.randn(len(signal)))
        signal = signal + noise

        return signal

    def extract_features(self, complex_signal):
        """从复数信号提取多通道特征"""
        # 预处理
        if self.preprocess:
            complex_signal = self.preprocessor.preprocess_signal(complex_signal)

        # 数据增强
        if self.augment and np.random.rand() > 0.5:
            complex_signal = self.augment_signal(complex_signal)

        # 提取多种特征表示
        # 1. I/Q分量
        I = np.real(complex_signal)
        Q = np.imag(complex_signal)

        # 2. 幅度和相位
        amplitude = np.abs(complex_signal)
        phase = np.angle(complex_signal)

        # 3. 瞬时频率
        phase_unwrapped = np.unwrap(phase)
        inst_freq = np.diff(phase_unwrapped)
        inst_freq = np.pad(inst_freq, (0, 1), mode='edge')

        # 4. 高阶统计量特征（作为额外通道）
        # 归一化二阶矩
        c20 = np.abs(complex_signal) ** 2
        c20 = (c20 - np.mean(c20)) / (np.std(c20) + 1e-8)

        # 归一化四阶矩
        c40 = np.abs(complex_signal) ** 4
        c40 = (c40 - np.mean(c40)) / (np.std(c40) + 1e-8)

        # 组合所有通道 (512, 7)
        features = np.stack([I, Q, amplitude, phase, inst_freq, c20, c40], axis=1)

        return torch.FloatTensor(features)

    def __getitem__(self, idx):
        # 获取复数信号
        complex_signal = self.X[idx]
        label = self.y[idx]

        # 提取特征
        features = self.extract_features(complex_signal)

        if self.transform:
            features = self.transform(features)

        return features, label


class ImprovedCNN1D(nn.Module):
    """改进的1D CNN模型，处理多通道输入"""
    def __init__(self, input_length=512, input_channels=7, num_classes=12):
        super(ImprovedCNN1D, self).__init__()

        # 第一个卷积块 - 处理多通道输入
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )

        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 调整输入维度：从 (batch, length, channels) 到 (batch, channels, length)
        x = x.permute(0, 2, 1)

        # 通过卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 应用注意力
        att = self.attention(x)
        x = x * att

        x = self.conv4(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 通过全连接层
        x = self.fc(x)

        return x

def load_dataset(dataset_path):
    """加载数据集 - 返回复数信号"""
    # 调制类型列表
    modulation_types = [
        '4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
        '32qam', '64qam', '128qam', 'bpsk', 'ook', 'qpsk'
    ]

    # 标签映射
    label_map = {
        '4ask': 0, '8ask': 1, '8psk': 2, '16psk': 3,
        '16qam': 4, '32psk': 5, '32qam': 6, '64qam': 7,
        '128qam': 8, 'bpsk': 9, 'ook': 10, 'qpsk': 11
    }

    all_data = []
    all_labels = []

    print("Loading dataset...")
    for mod_type in modulation_types:
        file_path = os.path.join(dataset_path, f'{mod_type}_seg512.mat')

        if os.path.exists(file_path):
            # 加载数据
            mat_data = loadmat(file_path)

            # 获取数据
            frames = mat_data['frames']  # (512, 2, N)

            # 获取天线0的复数数据
            complex_data = frames[:, 0, :]  # (512, N)

            # 转置得到 (N, 512)
            complex_data = complex_data.T

            n_samples = complex_data.shape[0]
            labels = np.full(n_samples, label_map[mod_type])

            all_data.append(complex_data)
            all_labels.append(labels)

            print(f"  {mod_type}: {n_samples} samples loaded")
        else:
            print(f"  Warning: {file_path} not found")

    # 合并所有数据
    X = np.vstack(all_data)  # (N, 512) 复数数组
    y = np.hstack(all_labels)

    print(f"\nTotal dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    return X, y, modulation_types


def test_model(model_path, test_path, batch_size=64):
    """测试保存的模型"""
    # 1. 加载模型
    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # 获取模型参数
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        input_length = checkpoint.get('input_length', 512)
        num_classes = checkpoint.get('num_classes', 12)
        modulation_names_trained = checkpoint.get('modulation_names', None)
    else:
        # 如果是旧格式，可能直接保存的是state_dict
        model_state = checkpoint
        input_length = 512
        num_classes = 12
        modulation_names_trained = None

    # 创建模型
    model = ImprovedCNN1D(input_length=input_length, num_classes=num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Input length: {input_length}")
    print(f"Number of classes: {num_classes}")

    # 2. 加载测试数据
    X_test, y_test, modulation_names = load_dataset(test_path)

    # 3. 创建数据加载器
    test_dataset = ModulationDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 进行预测
    print("\nRunning predictions...")
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)

            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} samples...")

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # 6. 计算性能指标
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    # 整体准确率
    overall_accuracy = accuracy_score(y_test, all_predictions)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

    # 每个类别的准确率
    print("\nPer-class Accuracy:")
    for i, mod_type in enumerate(modulation_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], all_predictions[mask])
            print(f"  {mod_type:8s}: {class_acc:.4f} ({class_acc * 100:.2f}%) - {np.sum(mask)} samples")

    # 分类报告
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, all_predictions,
                                target_names=modulation_names,
                                digits=4))

    # 7. 混淆矩阵
    cm = confusion_matrix(y_test, all_predictions)

    # 绘制混淆矩阵
    plt.figure(figsize=(14, 12))

    # 计算百分比混淆矩阵
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建标注文本
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)'

    sns.heatmap(cm_percentage, annot=annot_text, fmt='', cmap='Blues',
                xticklabels=modulation_names, yticklabels=modulation_names,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Confusion Matrix (Overall Accuracy: {overall_accuracy:.4f})', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 8. 分析错误分类
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # 找出最容易混淆的类别对
    print("\nMost Confused Pairs:")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # 移除对角线

    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_copy[i, j] > 0:
                confusion_pairs.append((i, j, cm_copy[i, j]))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    for i, (true_idx, pred_idx, count) in enumerate(confusion_pairs[:10]):
        true_class = modulation_names[true_idx]
        pred_class = modulation_names[pred_idx]
        percentage = count / np.sum(y_test == true_idx) * 100
        print(f"  {i + 1}. {true_class} → {pred_class}: {count} samples ({percentage:.1f}%)")

    # 9. 置信度分析
    print("\nConfidence Analysis:")

    # 计算每个预测的置信度
    max_probs = np.max(all_probabilities, axis=1)

    # 正确预测和错误预测的置信度
    correct_mask = all_predictions == y_test
    correct_confidences = max_probs[correct_mask]
    incorrect_confidences = max_probs[~correct_mask]

    print(f"  Average confidence (correct predictions): {np.mean(correct_confidences):.4f}")
    print(f"  Average confidence (incorrect predictions): {np.mean(incorrect_confidences):.4f}")

    # 绘制置信度分布
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 51)
    plt.hist(correct_confidences, bins=bins, alpha=0.5, label='Correct', density=True)
    plt.hist(incorrect_confidences, bins=bins, alpha=0.5, label='Incorrect', density=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Distribution for Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 10. 保存结果
    results = {
        'overall_accuracy': overall_accuracy,
        'predictions': all_predictions,
        'true_labels': y_test,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'modulation_names': modulation_names,
    }

    # 保存结果
    save_path = 'test_results.npz'
    np.savez(save_path, **results)
    print(f"\nTest results saved to: {save_path}")

    return results


def main():
    # 设置路径
    model_path = 'modulation_classifier_improved.pth'  # 或 'best_modulation_model.pth'
    test_path = r'D:\RadioData\testcase0715'

    # 运行测试
    results = test_model(model_path, test_path)

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print(f"Overall Test Accuracy: {results['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()