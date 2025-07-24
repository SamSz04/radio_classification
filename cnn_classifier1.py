import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
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


class CNN1D(nn.Module):
    """1D CNN模型用于调制识别"""

    def __init__(self, input_length=512, num_classes=10):
        super(CNN1D, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),  # 输入通道数为2 (I和Q)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
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
        x = self.conv4(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 通过全连接层
        x = self.fc(x)

        return x

class ModulationClassifier:
    """调制分类器"""
    def __init__(self, model_type='cnn', input_length=512, num_classes=12):
        self.model_type = model_type
        self.input_length = input_length
        self.num_classes = num_classes
        self.device = device

        # 创建模型
        if model_type == 'cnn':
            self.model = CNN1D(input_length, num_classes).to(device)
        else:
            raise ValueError("model_type must be 'cnn' or 'resnet'")

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

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

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
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 调整学习率
            self.scheduler.step(val_loss)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印进度
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_modulation_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_modulation_model.pth'))

    def plot_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
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

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # 计算指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        accuracy = np.mean(all_predictions == all_targets)
        print(f'Test Accuracy: {accuracy:.4f}')

        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)

        # 分类报告
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]

        print('\nClassification Report:')
        print(classification_report(all_targets, all_predictions, target_names=class_names))

        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        return accuracy, cm


def load_dataset(dataset_path):
    """加载数据集"""
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

    # modulation_types = [
    #     '4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
    #     '32qam', 'bpsk', 'ook', 'qpsk'
    # ]
    #
    # # 标签映射
    # label_map = {
    #     '4ask': 0, '8ask': 1, '8psk': 2, '16psk': 3,
    #     '16qam': 4, '32psk': 5, '32qam': 6,
    #     'bpsk': 7, 'ook': 8, 'qpsk': 9
    # }

    all_data = []
    all_labels = []

    print("Loading dataset...")
    for mod_type in modulation_types:
        file_path = os.path.join(dataset_path, f'{mod_type}_seg512.mat')

        if os.path.exists(file_path):
            # 加载数据
            mat_data = loadmat(file_path)

            # 获取数据
            frames = mat_data['frames']   # (512, 2, N)

            if frames.shape[1] != 2:
                raise ValueError(f"Expected 2 antennas, got shape {frames.shape}")

            # 选择天线0的数据（复数），维度：(512, N)
            complex_data = frames[:, 0, :]  # 取天线0

            # 拆分为实部和虚部 → 得到 (N, 512, 2)
            X_real = np.real(complex_data).T  # (N, 512)
            X_imag = np.imag(complex_data).T  # (N, 512)
            X = np.stack((X_real, X_imag), axis=2)  # (N, 512, 2)

            labels = np.full(X.shape[0], label_map[mod_type])

            all_data.append(X)
            all_labels.append(labels)

            print(f"  {mod_type}: {X.shape[0]} samples loaded")
        else:
            print(f"  Warning: {file_path} not found")

    # 合并所有数据
    X = np.vstack(all_data)     # (total_samples, 512, 2)
    y = np.hstack(all_labels)

    print(f"\nTotal dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    return X, y, modulation_types

def main():
    # 设置参数
    dataset_path = r'D:\RadioData\dataset1'

    batch_size = 128
    epochs = 20
    learning_rate = 0.0001

    # 加载数据
    X, y, modulation_names = load_dataset(dataset_path)

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    print(f"\nDataset split:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # 创建数据加载器
    train_dataset = ModulationDataset(X_train, y_train)
    val_dataset = ModulationDataset(X_val, y_val)
    test_dataset = ModulationDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建和训练模型
    classifier = ModulationClassifier(model_type='cnn', input_length=512, num_classes=12)
    classifier.compile(learning_rate=learning_rate)

    # 打印模型结构
    print("\nModel Architecture:")
    print(classifier.model)
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 训练模型
    print("\nTraining model...")
    classifier.fit(train_loader, val_loader, epochs=epochs, patience=10)

    # 绘制训练历史
    classifier.plot_history()

    # 评估模型
    print("\nEvaluating model on test set...")
    test_accuracy, cm = classifier.evaluate(test_loader, class_names=modulation_names)

    #=============================================

    # 加载测试数据
    print("\nLoading 0714test data...")
    test_path = r'D:\RadioData\testcase0714'
    X_test0714, y_test0714, modulation_names = load_dataset(test_path)

    print(f"\nTest data shape: {X_test0714.shape}")

    test0714_dataset = ModulationDataset(X_test0714, y_test0714)
    test0714_loader = DataLoader(test0714_dataset, batch_size=batch_size, shuffle=False)

    # 评估测试集
    print("\nEvaluating model on 0714test set...")
    test0714_accuracy, cm = classifier.evaluate(test0714_loader, class_names=modulation_names)

    print(f"\n0714Test Accuracy: {test0714_accuracy:.4f}")

    # 保存完整模型
    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'model_type': classifier.model_type,
        'input_length': classifier.input_length,
        'num_classes': classifier.num_classes,
        'modulation_names': modulation_names
    }, 'modulation_classifier_complete.pth')

    print("\nModel saved to 'modulation_classifier64_complete.pth'")

    # 测试单个样本
    print("\nTesting single sample prediction:")
    sample_idx = np.random.randint(0, len(X_test))
    sample = torch.FloatTensor(X_test[sample_idx:sample_idx + 1]).to(device)
    true_label = y_test[sample_idx]

    classifier.model.eval()
    with torch.no_grad():
        output = classifier.model(sample)
        probabilities = torch.softmax(output, dim=1)
        pred_label = output.argmax(dim=1).item()
        confidence = probabilities[0, pred_label].item()

    print(f"True label: {modulation_names[true_label]}")
    print(f"Predicted label: {modulation_names[pred_label]}")
    print(f"Confidence: {confidence:.3f}")


if __name__ == "__main__":
    main()