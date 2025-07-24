import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
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


class CombinedDataset(Dataset):
    """综合I/Q和相位特征的数据集"""

    def __init__(self, X, y, transform=None):
        """
        X: 原始I/Q数据 (N, 512, 2)
        """
        self.X = torch.FloatTensor(X)  # 保留原始I/Q数据
        self.y = torch.LongTensor(y)
        self.transform = transform

        # 预处理：提取相位特征
        self.phase_features = self._extract_phase_features(X)

    def _extract_phase_features(self, X):
        """从I/Q数据中提取相位特征"""
        N = X.shape[0]
        phase_features = np.zeros((N, 512, 3))  # 3个通道：相位、解缠绕相位、瞬时频率

        for i in range(N):
            # 构造复数信号
            complex_signal = X[i, :, 0] + 1j * X[i, :, 1]

            # 1. 直接相位 [-π, π]
            phase = np.angle(complex_signal)

            # 2. 解缠绕相位
            phase_unwrapped = np.unwrap(phase)

            # 3. 瞬时频率（相位差分）
            inst_freq = np.diff(phase_unwrapped)
            inst_freq = np.pad(inst_freq, (0, 1), mode='edge')

            phase_features[i, :, 0] = phase
            phase_features[i, :, 1] = phase_unwrapped
            phase_features[i, :, 2] = inst_freq

        return torch.FloatTensor(phase_features)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        iq_sample = self.X[idx]  # (512, 2)
        phase_sample = self.phase_features[idx]  # (512, 3)
        label = self.y[idx]

        if self.transform:
            iq_sample = self.transform(iq_sample)
            phase_sample = self.transform(phase_sample)

        return iq_sample, phase_sample, label


class CombinedCNN(nn.Module):
    """综合I/Q卷积和相位特征的CNN模型"""

    def __init__(self, input_length=512, num_classes=12):
        super(CombinedCNN, self).__init__()

        # ===== I/Q特征分支 =====
        # I/Q特征提取层 - 使用1x2卷积核
        self.iq_feature_conv = nn.Conv2d(1, 64, kernel_size=(1, 2), stride=1, padding=0)

        # I/Q分支的卷积层
        self.iq_conv1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.iq_conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.iq_conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # ===== 相位特征分支 =====
        # 相位分支的卷积层
        self.phase_conv1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.phase_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.phase_conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # ===== 特征融合 =====
        # 融合层：I/Q分支256 + 相位分支256 = 512
        self.fusion_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, iq_data, phase_data):
        batch_size = iq_data.size(0)

        # ===== I/Q特征提取分支 =====
        # iq_data: (batch, 512, 2)
        iq_x = iq_data.unsqueeze(1)  # (batch, 1, 512, 2)
        iq_x = self.iq_feature_conv(iq_x)  # (batch, 64, 512, 1)
        iq_x = iq_x.squeeze(-1)  # (batch, 64, 512)

        iq_x = self.iq_conv1(iq_x)
        iq_x = self.iq_conv2(iq_x)
        iq_x = self.iq_conv3(iq_x)  # (batch, 256, 1)
        iq_features = iq_x.view(batch_size, -1)  # (batch, 256)

        # ===== 相位特征提取分支 =====
        # phase_data: (batch, 512, 3) -> (batch, 3, 512)
        phase_x = phase_data.permute(0, 2, 1)

        phase_x = self.phase_conv1(phase_x)
        phase_x = self.phase_conv2(phase_x)
        phase_x = self.phase_conv3(phase_x)  # (batch, 256, 1)
        phase_features = phase_x.view(batch_size, -1)  # (batch, 256)

        # ===== 特征融合 =====
        combined_features = torch.cat([iq_features, phase_features], dim=1)  # (batch, 512)
        output = self.fusion_fc(combined_features)

        return output

    def get_intermediate_features(self, iq_data, phase_data):
        """获取中间层特征用于可视化"""
        batch_size = iq_data.size(0)

        # I/Q分支
        iq_x = iq_data.unsqueeze(1)
        iq_x = self.iq_feature_conv(iq_x)
        iq_x = iq_x.squeeze(-1)

        iq_x = self.iq_conv1(iq_x)
        iq_x = self.iq_conv2(iq_x)
        iq_x = self.iq_conv3(iq_x)
        iq_features = iq_x.view(batch_size, -1)

        # 相位分支
        phase_x = phase_data.permute(0, 2, 1)
        phase_x = self.phase_conv1(phase_x)
        phase_x = self.phase_conv2(phase_x)
        phase_x = self.phase_conv3(phase_x)
        phase_features = phase_x.view(batch_size, -1)

        return iq_features, phase_features


class CombinedClassifier:
    """综合特征分类器"""
    def __init__(self, input_length=512, num_classes=12):
        self.input_length = input_length
        self.num_classes = num_classes
        self.device = device

        self.model = CombinedCNN(input_length, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def compile(self, learning_rate=0.001):
        """配置优化器"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (iq_data, phase_data, target) in enumerate(progress_bar):
            iq_data = iq_data.to(self.device)
            phase_data = phase_data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(iq_data, phase_data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

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
            for iq_data, phase_data, target in tqdm(val_loader, desc="Validation", leave=False):
                iq_data = iq_data.to(self.device)
                phase_data = phase_data.to(self.device)
                target = target.to(self.device)

                output = self.model(iq_data, phase_data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, epochs=100, patience=10):
        """训练模型"""
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_combined_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        self.model.load_state_dict(torch.load('best_combined_model.pth'))

    def plot_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate(self, test_loader, class_names=None):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for iq_data, phase_data, target in tqdm(test_loader, desc="Evaluating"):
                iq_data = iq_data.to(self.device)
                phase_data = phase_data.to(self.device)
                target = target.to(self.device)

                output = self.model(iq_data, phase_data)
                probs = torch.softmax(output, dim=1)
                _, predicted = output.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

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

    def visualize_feature_importance(self, test_loader):
        """可视化两个分支的特征重要性"""
        self.model.eval()

        iq_features_all = []
        phase_features_all = []
        labels_all = []

        with torch.no_grad():
            for iq_data, phase_data, target in test_loader:
                iq_data = iq_data.to(self.device)
                phase_data = phase_data.to(self.device)

                iq_feat, phase_feat = self.model.get_intermediate_features(iq_data, phase_data)

                iq_features_all.append(iq_feat.cpu().numpy())
                phase_features_all.append(phase_feat.cpu().numpy())
                labels_all.extend(target.cpu().numpy())

                if len(labels_all) > 1000:  # 只用前1000个样本
                    break

        iq_features_all = np.vstack(iq_features_all)
        phase_features_all = np.vstack(phase_features_all)

        # 计算特征的方差作为重要性度量
        iq_importance = np.var(iq_features_all, axis=0)
        phase_importance = np.var(phase_features_all, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.bar(range(len(iq_importance)), iq_importance)
        ax1.set_title('I/Q Branch Feature Importance')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Variance')

        ax2.bar(range(len(phase_importance)), phase_importance)
        ax2.set_title('Phase Branch Feature Importance')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Variance')

        plt.tight_layout()
        plt.show()

        print(f"Average I/Q feature importance: {np.mean(iq_importance):.4f}")
        print(f"Average Phase feature importance: {np.mean(phase_importance):.4f}")


def load_dataset(dataset_path):
    """加载数据集"""
    modulation_types = [
        '4ask', '8ask', 'bpsk', '8psk', '16psk', '32psk',
        'qpsk', 'ook', '16qam', '32qam', '64qam', '128qam',
    ]

    label_map = {
        '4ask': 0, '8ask': 1, 'bpsk': 2, '8psk': 3,
        '16psk': 4, '32psk': 5, 'qpsk': 6, 'ook': 7,
        '16qam': 8, '32qam': 9, '64qam': 10, '128qam': 11
    }

    all_data = []
    all_labels = []

    print("Loading dataset...")
    for mod_type in modulation_types:
        file_path = os.path.join(dataset_path, f'{mod_type}_seg512.mat')

        if os.path.exists(file_path):
            mat_data = loadmat(file_path)
            frames = mat_data['frames']  # (512, 2, N)

            if frames.shape[1] != 2:
                raise ValueError(f"Expected 2 antennas, got shape {frames.shape}")

            complex_data = frames[:, 0, :]  # 取天线0
            X_real = np.real(complex_data).T  # (N, 512)
            X_imag = np.imag(complex_data).T  # (N, 512)
            X = np.stack((X_real, X_imag), axis=2)  # (N, 512, 2)

            labels = np.full(X.shape[0], label_map[mod_type])

            all_data.append(X)
            all_labels.append(labels)

            print(f"  {mod_type}: {X.shape[0]} samples loaded")
        else:
            print(f"  Warning: {file_path} not found")

    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    print(f"\nTotal dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    return X, y, modulation_types


def main():
    dataset_path = r'D:\RadioData\dataset1'
    batch_size = 128
    epochs = 10
    learning_rate = 0.0001

    X, y, modulation_names = load_dataset(dataset_path)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"\nDataset split:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # 创建综合特征数据集
    train_dataset = CombinedDataset(X_train, y_train)
    val_dataset = CombinedDataset(X_val, y_val)
    test_dataset = CombinedDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier = CombinedClassifier(input_length=512, num_classes=12)
    classifier.compile(learning_rate=learning_rate)

    print("\nModel Architecture:")
    print(classifier.model)
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\nTraining model...")
    classifier.fit(train_loader, val_loader, epochs=epochs, patience=10)

    classifier.plot_history()

    print("\nEvaluating model on test set...")
    test_accuracy, cm, test_probs = classifier.evaluate(test_loader, class_names=modulation_names)

    # 可视化特征重要性
    print("\nVisualizing feature importance...")
    classifier.visualize_feature_importance(test_loader)

    # 测试testcase0714
    print("\nLoading 0714test data...")
    test_path = r'D:\RadioData\testcase0714'
    X_test0714, y_test0714, _ = load_dataset(test_path)

    print(f"\nTest data shape: {X_test0714.shape}")
    test0714_dataset = CombinedDataset(X_test0714, y_test0714)
    test0714_loader = DataLoader(test0714_dataset, batch_size=batch_size, shuffle=False)

    print("\nEvaluating model on 0714test set...")
    test0714_accuracy, cm, test_probs = classifier.evaluate(test0714_loader, class_names=modulation_names)
    print(f"\n0714Test Accuracy: {test0714_accuracy:.4f}")

    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'model_type': 'combined',
        'input_length': classifier.input_length,
        'num_classes': classifier.num_classes,
        'modulation_names': modulation_names
    }, 'combined_model_complete.pth')

    print("\nModel saved to 'combined_model_complete.pth'")

    # 比较不同特征的贡献
    print("\n" + "=" * 60)
    print("Feature Contribution Analysis")
    print("=" * 60)

    # 测试只使用I/Q特征
    print("\nTesting with I/Q features only...")
    classifier.model.eval()
    correct_iq_only = 0
    total = 0

    with torch.no_grad():
        for iq_data, phase_data, target in test_loader:
            iq_data = iq_data.to(device)
            # 创建全零的相位数据
            phase_data_zeros = torch.zeros_like(phase_data).to(device)
            target = target.to(device)

            output = classifier.model(iq_data, phase_data_zeros)
            _, predicted = output.max(1)
            correct_iq_only += predicted.eq(target).sum().item()
            total += target.size(0)

    iq_only_acc = 100. * correct_iq_only / total

    # 测试只使用相位特征
    print("\nTesting with Phase features only...")
    correct_phase_only = 0
    total = 0

    with torch.no_grad():
        for iq_data, phase_data, target in test_loader:
            # 创建全零的I/Q数据
            iq_data_zeros = torch.zeros_like(iq_data).to(device)
            phase_data = phase_data.to(device)
            target = target.to(device)

            output = classifier.model(iq_data_zeros, phase_data)
            _, predicted = output.max(1)
            correct_phase_only += predicted.eq(target).sum().item()
            total += target.size(0)

    phase_only_acc = 100. * correct_phase_only / total

    print(f"\nFeature Contribution Summary:")
    print(f"Combined (I/Q + Phase): {test_accuracy * 100:.2f}%")
    print(f"I/Q features only: {iq_only_acc:.2f}%")
    print(f"Phase features only: {phase_only_acc:.2f}%")
    print(f"Improvement from combination: {(test_accuracy * 100 - max(iq_only_acc, phase_only_acc)):.2f}%")


if __name__ == "__main__":
    main()