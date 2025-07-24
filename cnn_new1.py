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


class ModulationDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class IQConvCNN(nn.Module):
    """使用I/Q卷积特征的CNN模型"""

    def __init__(self, input_length=512, num_classes=12):
        super(IQConvCNN, self).__init__()

        # I/Q特征提取层 - 使用1x2卷积核
        # (batch, in_channels=1, height=512, width=2)
        # 输入: (batch, 1, 512, 2)
        # 输出: (batch, 64, 512, 1)
        self.iq_feature_conv = nn.Conv2d(1, 64, kernel_size=(1, 2), stride=1, padding=0)

        # 后续的1D卷积层
        # 将2D输出reshape为1D: (batch, 64, 512)
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

        # 全连接层
        x = self.fc(x)

        return x

    def visualize_iq_kernels(self):
        """可视化学习到的I/Q卷积核"""
        kernels = self.iq_feature_conv.weight.data.cpu().numpy()

        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        axes = axes.ravel()

        for i in range(64):
            kernel = kernels[i, 0, 0, :]  # 提取1x2卷积核
            axes[i].bar(['I', 'Q'], kernel)
            axes[i].set_title(f'Kernel {i}')
            axes[i].set_ylim(-1, 1)

        plt.tight_layout()
        plt.suptitle('Learned I/Q Convolution Kernels', y=1.02)
        plt.show()


class ModulationClassifier:
    """调制分类器"""

    def __init__(self, model_type='iq_conv', input_length=512, num_classes=12):
        self.model_type = model_type
        self.input_length = input_length
        self.num_classes = num_classes
        self.device = device

        # 创建模型
        self.model = IQConvCNN(input_length, num_classes).to(device)

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
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
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
            for data, target in tqdm(val_loader, desc="Validation", leave=False):
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
                torch.save(self.model.state_dict(), 'best_iq_conv_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        self.model.load_state_dict(torch.load('best_iq_conv_model.pth'))

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
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
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


def load_dataset(dataset_path):
    """加载数据集"""
    modulation_types = [
        '4ask', '8ask', 'bpsk', '8psk', '16psk',  '32psk',
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
    # path1 = r'D:\RadioData\dataset1'
    # path2 = r'D:\RadioData\dataset3'
    # batch_size = 128
    # epochs = 30
    # learning_rate = 0.0001
    #
    # # --- 2. 分别加载两个数据集 ---
    # X1, y1, modulation_names = load_dataset(path1)
    # X2, y2, _ = load_dataset(path2)
    #
    # # --- 3. 合并样本和标签 ---
    # # np.vstack 将样本在第一个维度（N）上拼接
    # X = np.vstack([X1, X2])  # 形状从 (N1,512,2),(N2,512,2) -> (N1+N2,512,2)
    # y = np.hstack([y1, y2])  # 形状从 (N1,), (N2,) -> (N1+N2,)
    #
    # print(f"\nTotal merged dataset shape: {X.shape}, labels shape: {y.shape}")

    dataset_path = r'D:\RadioData\dataset1'
    batch_size = 128
    epochs = 30
    learning_rate = 0.0001

    X, y, modulation_names = load_dataset(dataset_path)

    # --- 4. 划分训练、验证、测试集 ---
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"\nDataset split:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # --- 5. 创建 Dataset 和 DataLoader ---
    train_dataset = ModulationDataset(X_train, y_train)
    val_dataset = ModulationDataset(X_val, y_val)
    test_dataset = ModulationDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier = ModulationClassifier(model_type='iq_conv', input_length=512, num_classes=12)
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

    # 可视化学习到的I/Q卷积核
    print("\nVisualizing learned I/Q convolution kernels...")
    classifier.model.visualize_iq_kernels()

    # 测试testcase0714
    print("\nLoading 0714test data...")
    test_path = r'D:\RadioData\testcase0714'
    X_test0714, y_test0714, _ = load_dataset(test_path)

    print(f"\nTest data shape: {X_test0714.shape}")
    test0714_dataset = ModulationDataset(X_test0714, y_test0714)
    test0714_loader = DataLoader(test0714_dataset, batch_size=batch_size, shuffle=False)

    print("\nEvaluating model on 0714test set...")
    test0714_accuracy, cm, test_probs = classifier.evaluate(test0714_loader, class_names=modulation_names)
    print(f"\n0714Test Accuracy: {test0714_accuracy:.4f}")

    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'model_type': 'iq_conv',
        'input_length': classifier.input_length,
        'num_classes': classifier.num_classes,
        'modulation_names': modulation_names
    }, 'iq_conv_model_complete.pth')

    print("\nModel saved to 'iq_conv_model_complete.pth'")


if __name__ == "__main__":
    main()