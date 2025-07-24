import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat, savemat
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class STFTDataset(Dataset):
    """STFT时频图数据集"""

    def __init__(self, stft_data, labels, transform=None, normalize=True):
        """
        stft_data: (N, F, T) - N个样本，F个频率点，T个时间点
        labels: (N,) - 标签
        """
        self.stft_data = stft_data
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.normalize = normalize

        if self.normalize:
            # 转换到dB域并归一化
            self.stft_data = 20 * np.log10(self.stft_data + 1e-10)
            # 全局归一化
            self.mean = np.mean(self.stft_data)
            self.std = np.std(self.stft_data)
            self.stft_data = (self.stft_data - self.mean) / self.std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取时频图
        stft_image = self.stft_data[idx]  # (F, T)

        # 添加通道维度
        stft_image = np.expand_dims(stft_image, axis=0)  # (1, F, T)

        if self.transform:
            stft_image = self.transform(stft_image)

        return torch.FloatTensor(stft_image), self.labels[idx]


class STFTModulationCNN(nn.Module):
    """基于STFT时频图的CNN调制识别模型"""

    def __init__(self, num_classes=12, input_channels=1):
        super(STFTModulationCNN, self).__init__()

        # 卷积特征提取器
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
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

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class STFTResNet(nn.Module):
    """基于ResNet的STFT调制识别模型"""

    def __init__(self, num_classes=12, input_channels=1):
        super(STFTResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def load_stft_data(data_path, modulation_types):
    """加载所有调制类型的STFT数据"""
    all_data = []
    all_labels = []

    print("Loading STFT data...")
    for i, mod_type in enumerate(modulation_types):
        file_path = os.path.join(data_path, f'{mod_type}_stft.mat')

        if os.path.exists(file_path):
            # 加载STFT矩阵
            mat_data = loadmat(file_path)
            stft_matrix = mat_data['rx_stft']  # (F, T, N)

            # 转换维度: (F, T, N) -> (N, F, T)
            stft_matrix = np.transpose(stft_matrix, (2, 0, 1))

            n_samples = stft_matrix.shape[0]
            labels = np.full(n_samples, i)

            all_data.append(stft_matrix)
            all_labels.append(labels)

            print(f"  {mod_type}: {n_samples} samples loaded, shape: {stft_matrix.shape}")
        else:
            print(f"  Warning: {file_path} not found")

    # 合并所有数据
    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    print(f"\nTotal dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    return X, y


def visualize_stft_samples(X, y, modulation_names, n_samples=3):
    """可视化不同调制类型的STFT时频图"""
    n_classes = len(modulation_names)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(12, 2 * n_classes))

    for i in range(n_classes):
        # 找到该类别的样本
        class_indices = np.where(y == i)[0]
        if len(class_indices) == 0:
            continue

        # 随机选择n_samples个样本
        sample_indices = np.random.choice(class_indices,
                                          min(n_samples, len(class_indices)),
                                          replace=False)

        for j, idx in enumerate(sample_indices):
            stft_image = X[idx]

            if n_classes == 1:
                ax = axes[j]
            else:
                ax = axes[i, j] if n_samples > 1 else axes[i]

            # 显示时频图
            im = ax.imshow(stft_image, aspect='auto', origin='lower', cmap='jet')

            if j == 0:
                ax.set_ylabel(modulation_names[i])
            if i == 0:
                ax.set_title(f'Sample {j + 1}')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle('STFT Time-Frequency Representations', y=1.02)
    plt.show()


def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5,
                                                     min_lr=1e-6)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for data, target in train_bar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            train_bar.set_postfix({
                'loss': f'{train_loss / len(train_loader):.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            for data, target in val_bar:
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

                val_bar.set_postfix({
                    'loss': f'{val_loss / len(val_loader):.4f}',
                    'acc': f'{100. * val_correct / val_total:.2f}%'
                })

        # 记录历史
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        # 学习率调整
        scheduler.step(val_loss / len(val_loader))

        print(f'\nEpoch {epoch + 1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_stft_model.pth')

    return history


def evaluate_model(model, test_loader, class_names):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算准确率
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    print(f'Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')

    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)

    # 分类报告
    print('\nClassification Report:')
    print(classification_report(all_targets, all_predictions,
                                target_names=class_names, digits=4))

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return accuracy, cm


def main():
    # 配置参数
    data_path = r'D:\RadioData\stft_data'  # STFT数据路径
    modulation_types = [
        '4ask', '8ask', 'bpsk', '8psk', '16psk',  '32psk',
        'qpsk', 'ook', '16qam', '32qam', '64qam', '128qam',
    ]

    batch_size = 32
    epochs = 5
    learning_rate = 0.0001

    # 加载数据
    X, y = load_stft_data(data_path, modulation_types)

    # 可视化样本
    print("\nVisualizing STFT samples...")
    visualize_stft_samples(X, y, modulation_types, n_samples=3)

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                    random_state=42)

    print(f"\nDataset split:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # 创建数据加载器
    train_dataset = STFTDataset(X_train, y_train)
    val_dataset = STFTDataset(X_val, y_val)
    test_dataset = STFTDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = STFTModulationCNN(num_classes=len(modulation_types)).to(device)
    # model = STFTResNet(num_classes=len(modulation_types)).to(device)  # 使用ResNet

    print(f"\nModel: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 训练模型
    print("\nTraining model...")
    history = train_model(model, train_loader, val_loader, epochs, learning_rate)

    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # 加载最佳模型并评估
    print("\nLoading best model...")
    model.load_state_dict(torch.load('best_stft_model.pth'))

    print("\nEvaluating on test set...")
    test_accuracy, cm = evaluate_model(model, test_loader, modulation_types)

    #==============================================
    # 测试testcase0714
    print("\nLoading 0714test data...")
    test_path = r'D:\RadioData\stft_test'
    X_test0714, y_test0714 = load_stft_data(test_path, modulation_types)

    print(f"\nTest data shape: {X_test0714.shape}")
    test0714_dataset = STFTDataset(X_test0714, y_test0714)
    test0714_loader = DataLoader(test0714_dataset, batch_size=batch_size, shuffle=False)

    print("\nEvaluating model on 0714test set...")
    test0714_accuracy, cm = evaluate_model(model, test0714_loader, modulation_types)
    print(f"\n0714Test Accuracy: {test0714_accuracy:.4f}")


    # 保存完整模型信息
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'num_classes': len(modulation_types),
        'modulation_names': modulation_types,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm
    }, 'stft_classifier_complete.pth')

    print("\nModel saved to 'stft_classifier_complete.pth'")


if __name__ == "__main__":
    main()