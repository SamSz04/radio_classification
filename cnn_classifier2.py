import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.signal import hilbert
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

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


class ModulationClassifier:
    """改进的调制分类器"""
    def __init__(self, input_length=512, input_channels=7, num_classes=12):
        self.input_length = input_length
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.device = device

        # 创建模型
        self.model = ImprovedCNN1D(input_length, input_channels, num_classes).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def compile(self, learning_rate=0.001):
        """配置优化器"""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # 使用余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, epochs=100, patience=15):
        """训练模型"""
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 调整学习率
            self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印进度
            print(f'\nEpoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                }, 'best_modulation_model.pth')
                print(f'  --> New best model saved! (Val Acc: {val_acc:.2f}%)')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        # 加载最佳模型
        checkpoint = torch.load('best_modulation_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'\nBest validation accuracy: {checkpoint["best_val_acc"]:.2f}%')

    def plot_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def evaluate(self, test_loader, class_names=None):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = output.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_predictions == all_targets)
        print(f'Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')

        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)

        # 分类报告
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]

        print('\nClassification Report:')
        print(classification_report(all_targets, all_predictions,
                                    target_names=class_names, digits=4))

        # 绘制混淆矩阵
        plt.figure(figsize=(14, 12))

        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 创建标注
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Percentage (%)'})
        plt.title(f'Confusion Matrix (Overall Accuracy: {accuracy:.4f})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

        return accuracy, cm, all_probs


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


def main():
    # 设置参数
    dataset_path = r'D:\RadioData\dataset1'
    batch_size = 64
    epochs = 2
    learning_rate = 0.001

    # 加载数据（复数信号）
    X, y, modulation_names = load_dataset(dataset_path)

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nDataset split:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # 创建数据集（启用预处理和数据增强）
    train_dataset = ModulationDataset(X_train, y_train, augment=True, preprocess=True)
    val_dataset = ModulationDataset(X_val, y_val, augment=False, preprocess=True)
    test_dataset = ModulationDataset(X_test, y_test, augment=False, preprocess=True)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 创建和编译模型
    classifier = ModulationClassifier(input_length=512, input_channels=7, num_classes=12)
    classifier.compile(learning_rate=learning_rate)

    # 打印模型结构
    print("\nModel Architecture:")
    print(classifier.model)
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 训练模型
    print("\nTraining model...")
    classifier.fit(train_loader, val_loader, epochs=epochs, patience=15)

    # 绘制训练历史
    classifier.plot_history()

    # 评估模型
    print("\nEvaluating model on test set...")
    test_accuracy, cm, test_probs = classifier.evaluate(test_loader, class_names=modulation_names)

    # 保存完整模型信息
    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'input_length': classifier.input_length,
        'input_channels': classifier.input_channels,
        'num_classes': classifier.num_classes,
        'modulation_names': modulation_names,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm
    }, 'modulation_classifier_improved.pth')

    print("\nModel saved to 'modulation_classifier_improved.pth'")

    # 测试单个样本
    print("\nTesting single sample prediction:")
    sample_idx = np.random.randint(0, len(X_test))

    # 创建单样本数据集
    single_sample_dataset = ModulationDataset(
        X_test[sample_idx:sample_idx + 1],
        y_test[sample_idx:sample_idx + 1],
        preprocess=True
    )
    single_loader = DataLoader(single_sample_dataset, batch_size=1)

    classifier.model.eval()
    with torch.no_grad():
        for data, target in single_loader:
            data = data.to(device)
            output = classifier.model(data)
            probabilities = torch.softmax(output, dim=1)
            pred_label = output.argmax(dim=1).item()
            confidence = probabilities[0, pred_label].item()

    true_label = y_test[sample_idx]
    print(f"True label: {modulation_names[true_label]}")
    print(f"Predicted label: {modulation_names[pred_label]}")
    print(f"Confidence: {confidence:.3f}")

    # 显示前5个最可能的类别
    top5_probs, top5_indices = torch.topk(probabilities[0], 5)
    print("\nTop 5 predictions:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"  {i + 1}. {modulation_names[idx.item()]}: {prob.item():.3f}")


if __name__ == "__main__":
    main()