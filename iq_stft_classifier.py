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
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FusionDataset(Dataset):
    """融合I/Q和STFT特征的数据集"""

    def __init__(self, iq_data, stft_data, labels, normalize_stft=True):
        """
        iq_data: (N, 512, 2) - I/Q数据
        stft_data: (N, F, T) - STFT时频图
        labels: (N,) - 标签
        """
        self.iq_data = torch.FloatTensor(iq_data)
        self.labels = torch.LongTensor(labels)

        # STFT数据预处理
        if normalize_stft:
            # 转换到dB域
            stft_data = 20 * np.log10(stft_data + 1e-10)
            # 归一化
            self.stft_mean = np.mean(stft_data)
            self.stft_std = np.std(stft_data)
            stft_data = (stft_data - self.stft_mean) / self.stft_std

        self.stft_data = torch.FloatTensor(stft_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        iq_sample = self.iq_data[idx]  # (512, 2)
        stft_sample = self.stft_data[idx].unsqueeze(0)  # (1, F, T)
        label = self.labels[idx]

        return iq_sample, stft_sample, label


class IQFeatureExtractor(nn.Module):
    """I/Q特征提取器"""

    def __init__(self):
        super(IQFeatureExtractor, self).__init__()

        # I/Q特征提取层 - 使用1x2卷积核
        self.iq_feature_conv = nn.Conv2d(1, 64, kernel_size=(1, 2), stride=1, padding=0)

        # 后续的1D卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.feature_dim = 512

    def forward(self, x):
        # x: (batch, 512, 2)
        batch_size = x.size(0)

        # 添加通道维度: (batch, 1, 512, 2)
        x = x.unsqueeze(1)

        # I/Q特征提取
        x = self.iq_feature_conv(x)  # (batch, 64, 512, 1)
        x = x.squeeze(-1)  # (batch, 64, 512)

        # 后续卷积处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 展平
        x = x.view(batch_size, -1)

        return x


class STFTFeatureExtractor(nn.Module):
    """STFT时频图特征提取器"""

    def __init__(self):
        super(STFTFeatureExtractor, self).__init__()

        # 2D卷积特征提取器
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.feature_dim = 512

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FusionModel(nn.Module):
    """I/Q和STFT特征融合模型"""

    def __init__(self, num_classes=12, fusion_method='concat'):
        super(FusionModel, self).__init__()

        # 特征提取器
        self.iq_extractor = IQFeatureExtractor()
        self.stft_extractor = STFTFeatureExtractor()

        self.fusion_method = fusion_method

        # 特征融合维度
        if fusion_method == 'concat':
            fusion_dim = self.iq_extractor.feature_dim + self.stft_extractor.feature_dim  # 1024
        elif fusion_method == 'add':
            fusion_dim = self.iq_extractor.feature_dim  # 512
            # 确保两个特征维度相同
            assert self.iq_extractor.feature_dim == self.stft_extractor.feature_dim
        elif fusion_method == 'attention':
            fusion_dim = self.iq_extractor.feature_dim  # 512
            # 注意力机制
            self.attention_iq = nn.Sequential(
                nn.Linear(self.iq_extractor.feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            self.attention_stft = nn.Sequential(
                nn.Linear(self.stft_extractor.feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, iq_data, stft_data):
        # 提取特征
        iq_features = self.iq_extractor(iq_data)
        stft_features = self.stft_extractor(stft_data)

        # 特征融合
        if self.fusion_method == 'concat':
            # 简单拼接
            fused_features = torch.cat([iq_features, stft_features], dim=1)

        elif self.fusion_method == 'add':
            # 特征相加
            fused_features = iq_features + stft_features

        elif self.fusion_method == 'attention':
            # 注意力加权融合
            iq_weight = self.attention_iq(iq_features)
            stft_weight = self.attention_stft(stft_features)

            # 归一化权重
            total_weight = iq_weight + stft_weight + 1e-8
            iq_weight = iq_weight / total_weight
            stft_weight = stft_weight / total_weight

            fused_features = iq_weight * iq_features + stft_weight * stft_features

        # 分类
        output = self.classifier(fused_features)

        return output

    def get_feature_weights(self, iq_data, stft_data):
        """获取注意力权重（仅在attention融合模式下）"""
        if self.fusion_method != 'attention':
            return None, None

        iq_features = self.iq_extractor(iq_data)
        stft_features = self.stft_extractor(stft_data)

        iq_weight = self.attention_iq(iq_features)
        stft_weight = self.attention_stft(stft_features)

        total_weight = iq_weight + stft_weight + 1e-8
        iq_weight = iq_weight / total_weight
        stft_weight = stft_weight / total_weight

        return iq_weight.squeeze(), stft_weight.squeeze()


def load_iq_data(dataset_path, modulation_types, max_samples=5000):
    """加载I/Q数据（限制样本数为max_samples）"""
    all_data = []
    all_labels = []

    print("Loading I/Q data...")
    for i, mod_type in enumerate(modulation_types):
        file_path = os.path.join(dataset_path, f'{mod_type}_seg512.mat')

        if os.path.exists(file_path):
            mat_data = loadmat(file_path)
            frames = mat_data['frames']  # (512, 2, N)

            complex_data = frames[:, 0, :]  # 取天线0
            X_real = np.real(complex_data).T  # (N, 512)
            X_imag = np.imag(complex_data).T  # (N, 512)
            X = np.stack((X_real, X_imag), axis=2)  # (N, 512, 2)

            # 限制样本数量
            if X.shape[0] > max_samples:
                X = X[:max_samples]
                labels = np.full(max_samples, i)
            else:
                labels = np.full(X.shape[0], i)

            all_data.append(X)
            all_labels.append(labels)

            print(f"  {mod_type}: {X.shape[0]} samples loaded")

    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    return X, y


def load_stft_data(data_path, modulation_types, max_samples=5000):
    """加载STFT数据（限制样本数为max_samples）"""
    all_data = []

    print("Loading STFT data...")
    for i, mod_type in enumerate(modulation_types):
        file_path = os.path.join(data_path, f'{mod_type}_stft.mat')

        if os.path.exists(file_path):
            mat_data = loadmat(file_path)
            stft_matrix = mat_data['rx_stft']  # (F, T, N)

            # 转换维度: (F, T, N) -> (N, F, T)
            stft_matrix = np.transpose(stft_matrix, (2, 0, 1))

            # 限制样本数量
            if stft_matrix.shape[0] > max_samples:
                stft_matrix = stft_matrix[:max_samples]

            all_data.append(stft_matrix)
            print(f"  {mod_type}: shape {stft_matrix.shape}")

    X = np.vstack(all_data)

    return X



class FusionClassifier:
    """融合特征分类器"""
    def __init__(self, num_classes=12, fusion_method='concat'):
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.device = device

        self.model = FusionModel(num_classes, fusion_method).to(device)
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
        for iq_data, stft_data, target in progress_bar:
            iq_data = iq_data.to(self.device)
            stft_data = stft_data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(iq_data, stft_data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss / (len(progress_bar) + 1),
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
            for iq_data, stft_data, target in tqdm(val_loader, desc="Validation"):
                iq_data = iq_data.to(self.device)
                stft_data = stft_data.to(self.device)
                target = target.to(self.device)

                output = self.model(iq_data, stft_data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, epochs=30, patience=10):
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
                torch.save(self.model.state_dict(), 'best_fusion_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        self.model.load_state_dict(torch.load('best_fusion_model.pth'))

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

        # 如果使用注意力融合，记录权重
        if self.fusion_method == 'attention':
            all_iq_weights = []
            all_stft_weights = []

        with torch.no_grad():
            for iq_data, stft_data, target in tqdm(test_loader, desc="Evaluating"):
                iq_data = iq_data.to(self.device)
                stft_data = stft_data.to(self.device)
                target = target.to(self.device)

                output = self.model(iq_data, stft_data)
                probs = torch.softmax(output, dim=1)
                _, predicted = output.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                if self.fusion_method == 'attention':
                    iq_w, stft_w = self.model.get_feature_weights(iq_data, stft_data)
                    all_iq_weights.extend(iq_w.cpu().numpy())
                    all_stft_weights.extend(stft_w.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        accuracy = np.mean(all_predictions == all_targets)
        print(f'Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')

        # 如果使用注意力融合，显示平均权重
        if self.fusion_method == 'attention':
            avg_iq_weight = np.mean(all_iq_weights)
            avg_stft_weight = np.mean(all_stft_weights)
            print(f'\nAverage feature weights:')
            print(f'  I/Q features: {avg_iq_weight:.3f}')
            print(f'  STFT features: {avg_stft_weight:.3f}')

        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)

        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]

        print('\nClassification Report:')
        print(classification_report(all_targets, all_predictions,
                                    target_names=class_names, digits=4))

        # 绘制混淆矩阵
        plt.figure(figsize=(14, 12))
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Percentage (%)'})
        plt.title(f'Confusion Matrix - Fusion Model (Accuracy: {accuracy:.4f})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

        return accuracy, cm


def main():
    # 配置参数
    iq_data_path = r'D:\RadioData\dataset1'
    stft_data_path = r'D:\RadioData\stft_data'

    modulation_types = [
        '4ask', '8ask', 'bpsk', '8psk', '16psk', '32psk',
        'qpsk', 'ook', '16qam', '32qam', '64qam', '128qam',
    ]

    batch_size = 64
    epochs = 10
    learning_rate = 0.0001
    fusion_method = 'concat'  # 可选: 'concat', 'add', 'attention'

    # 加载数据
    print("Loading datasets...")
    X_iq, y = load_iq_data(iq_data_path, modulation_types)
    X_stft = load_stft_data(stft_data_path, modulation_types)

    # 确保样本数一致
    assert X_iq.shape[0] == X_stft.shape[0], "I/Q and STFT data must have same number of samples"

    print(f"\nDataset shapes:")
    print(f"I/Q data: {X_iq.shape}")
    print(f"STFT data: {X_stft.shape}")
    print(f"Labels: {y.shape}")

    # 划分数据集
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

    X_iq_train, X_iq_val, X_iq_test = X_iq[train_idx], X_iq[val_idx], X_iq[test_idx]
    X_stft_train, X_stft_val, X_stft_test = X_stft[train_idx], X_stft[val_idx], X_stft[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    print(f"\nDataset split:")
    print(f"Training: {len(y_train)} samples")
    print(f"Validation: {len(y_val)} samples")
    print(f"Test: {len(y_test)} samples")

    # 创建数据加载器
    train_dataset = FusionDataset(X_iq_train, X_stft_train, y_train)
    val_dataset = FusionDataset(X_iq_val, X_stft_val, y_val)
    test_dataset = FusionDataset(X_iq_test, X_stft_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建并训练模型
    print(f"\nUsing fusion method: {fusion_method}")
    classifier = FusionClassifier(num_classes=len(modulation_types), fusion_method=fusion_method)
    classifier.compile(learning_rate=learning_rate)

    print("\nModel Architecture:")
    print(classifier.model)
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("\nTraining model...")
    classifier.fit(train_loader, val_loader, epochs=epochs, patience=10)

    classifier.plot_history()

    print("\nEvaluating model on test set...")
    test_accuracy, cm = classifier.evaluate(test_loader, class_names=modulation_types)

    # 测试testcase0714
    print("\n" + "=" * 60)
    print("Testing on testcase0714...")
    print("=" * 60)

    test_iq_path = r'D:\RadioData\testcase0714'
    test_stft_path = r'D:\RadioData\stft_test'

    X_test0714_iq, y_test0714 = load_iq_data(test_iq_path, modulation_types)
    X_test0714_stft = load_stft_data(test_stft_path, modulation_types)

    # 确保测试数据样本数一致
    min_samples = min(X_test0714_iq.shape[0], X_test0714_stft.shape[0])
    X_test0714_iq = X_test0714_iq[:min_samples]
    X_test0714_stft = X_test0714_stft[:min_samples]
    y_test0714 = y_test0714[:min_samples]

    print(f"\nTest data shapes:")
    print(f"I/Q: {X_test0714_iq.shape}")
    print(f"STFT: {X_test0714_stft.shape}")

    test0714_dataset = FusionDataset(X_test0714_iq, X_test0714_stft, y_test0714)
    test0714_loader = DataLoader(test0714_dataset, batch_size=batch_size, shuffle=False)

    print("\nEvaluating model on testcase0714...")
    test0714_accuracy, cm0714 = classifier.evaluate(test0714_loader, class_names=modulation_types)

    # 保存模型
    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'fusion_method': fusion_method,
        'num_classes': classifier.num_classes,
        'modulation_names': modulation_types,
        'test_accuracy': test_accuracy,
        'test0714_accuracy': test0714_accuracy
    }, f'fusion_model_{fusion_method}_complete.pth')

    print(f"\nModel saved to 'fusion_model_{fusion_method}_complete.pth'")

if __name__ == "__main__":
    main()

    # 对比不同融合方法（可选）
    # if fusion_method == 'attention':
    #     print("\n" + "=" * 60)
    #     print("Feature importance analysis with attention fusion")
    #     print("=" * 60)
    #
    #     # 分析不同类别的特征权重
    #     class_iq_weights = {i: [] for i in range(len(modulation_types))}
    #     class_stft_weights = {i: [] for i in range(len(modulation_types))}
    #
    #     classifier.model.eval()
    #     with torch.no_grad():
    #         for iq_data, stft_data, target in test_loader:
    #             iq_data = iq_data.to(device)
    #             stft_data = stft_data.to(device)
    #
    #             iq_w, stft_w = classifier.model.get_feature_weights(iq_data, stft_data)
    #
    #             for i, label in enumerate(target):
    #                 class_iq_weights[label.item()].append(iq_w[i].item())
    #                 class_stft_weights[label.item()].append(stft_w[i].item())
    #
    #     # 计算每个类别的平均权重
    #     print("\nAverage feature weights by class:")
    #     print(f"{'Class':<10} {'I/Q Weight':<12} {'STFT Weight':<12}")
    #     print("-" * 35)
    #
    #     for i,