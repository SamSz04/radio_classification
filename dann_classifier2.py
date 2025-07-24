import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
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

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class GradientReversalFunction(Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_p
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_p=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_p = lambda_p

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_p)


class FeatureExtractor(nn.Module):
    """更简单的特征提取器 - CNN + 单个RNN"""
    def __init__(self, input_channels=2, cnn_feature_dim=512, rnn_feature_dim=256):
        super(FeatureExtractor, self).__init__()

        # CNN分支（同上）
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 只用GRU（通常比LSTM快）
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=rnn_feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.feature_dim = cnn_feature_dim + rnn_feature_dim

    def forward(self, x):
        batch_size = x.size(0)

        # CNN特征
        cnn_features = self.conv_blocks(x.permute(0, 2, 1))
        cnn_features = cnn_features.view(batch_size, -1)

        # GRU特征 - 使用最后的隐藏状态
        _, gru_h = self.gru(x)
        rnn_features = gru_h[-1]  # 取最后一层的输出

        # 拼接
        return torch.cat([cnn_features, rnn_features], dim=1)


class LabelClassifier(nn.Module):
    """标签分类器 - 预测调制类型"""

    def __init__(self, feature_dim=768, num_classes=12):  # 注意特征维度变为768
        super(LabelClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 384),  # 第一层稍微增大
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class DomainDiscriminator(nn.Module):
    """域判别器 - 判断数据来自源域还是目标域"""

    def __init__(self, feature_dim=768):  # 注意特征维度变为768
        super(DomainDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 1)  # 二分类：源域(0)或目标域(1)
        )

    def forward(self, x):
        return self.discriminator(x)


class DANN(nn.Module):
    """完整的DANN模型（CNN + LSTM-GRU）"""

    def __init__(self, input_channels=2, num_classes=12):
        super(DANN, self).__init__()

        # 特征提取器（CNN + LSTM-GRU）
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            cnn_feature_dim=512,
            rnn_feature_dim=256
        )

        # 标签分类器和域判别器使用组合后的特征维度
        total_feature_dim = self.feature_extractor.feature_dim
        self.label_classifier = LabelClassifier(total_feature_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(total_feature_dim)
        self.grl = GradientReversalLayer()

    def forward(self, x, lambda_p=1.0):
        # 提取特征（CNN + LSTM-GRU）
        features = self.feature_extractor(x)

        # 标签分类
        class_output = self.label_classifier(features)

        # 域判别（通过梯度反转层）
        self.grl.lambda_p = lambda_p
        reversed_features = self.grl(features)
        domain_output = self.domain_discriminator(reversed_features)

        return class_output, domain_output, features


class DANNDataset(Dataset):
    """DANN数据集类"""

    def __init__(self, X, y=None, domain_label=0, transform=None):
        """
        X: 复数信号数据
        y: 标签（目标域可能没有标签）
        domain_label: 0表示源域，1表示目标域
        """
        self.X = X
        self.y = y
        self.domain_label = domain_label
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 获取复数信号
        complex_signal = self.X[idx]

        # 提取I/Q特征
        I = np.real(complex_signal)
        Q = np.imag(complex_signal)

        # # 归一化
        # signal_power = np.mean(I ** 2 + Q ** 2)
        # if signal_power > 0:
        #     I = I / np.sqrt(signal_power)
        #     Q = Q / np.sqrt(signal_power)

        # 组合特征
        features = np.stack([I, Q], axis=1)  # (512, 2)
        features = torch.FloatTensor(features)

        # 域标签
        domain = torch.FloatTensor([self.domain_label])

        if self.y is not None:
            label = torch.LongTensor([self.y[idx]])
            return features, label, domain
        else:
            return features, domain


class DANNTrainer:
    """DANN训练器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 损失函数
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()

        # 历史记录
        self.history = {
            'train_class_loss': [], 'train_domain_loss': [],
            'train_class_acc': [], 'train_domain_acc': [],
            'val_class_loss': [], 'val_class_acc': [],
            'target_domain_acc': []
        }

    def train_epoch(self, source_loader, target_loader, epoch, total_epochs):
        self.model.train()

        # 计算lambda_p（渐进增加）
        p = float(epoch) / total_epochs
        lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

        # 统计量
        running_class_loss = 0.0
        running_domain_loss = 0.0
        correct_class = 0
        correct_domain = 0
        total_class = 0
        total_domain = 0

        # 创建迭代器
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        n_batches = max(len(source_loader), len(target_loader))

        progress_bar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}")

        for batch_idx in progress_bar:
            # 获取源域数据
            try:
                source_data, source_labels, source_domains = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_data, source_labels, source_domains = next(source_iter)

            # 获取目标域数据
            try:
                target_data, target_domains = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, target_domains = next(target_iter)

            # 移动到设备
            source_data = source_data.to(self.device)
            source_labels = source_labels.squeeze().to(self.device)
            source_domains = source_domains.to(self.device)
            target_data = target_data.to(self.device)
            target_domains = target_domains.to(self.device)

            # 合并数据
            combined_data = torch.cat([source_data, target_data], dim=0)
            combined_domains = torch.cat([source_domains, target_domains], dim=0)

            # 前向传播
            self.optimizer.zero_grad()

            # 源域：分类损失
            source_class_output, source_domain_output, _ = self.model(source_data, lambda_p)
            class_loss = self.class_criterion(source_class_output, source_labels)

            # 所有数据：域判别损失
            _, combined_domain_output, _ = self.model(combined_data, lambda_p)
            domain_loss = self.domain_criterion(combined_domain_output.squeeze(), combined_domains.squeeze())

            # 总损失
            total_loss = class_loss + domain_loss

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            running_class_loss += class_loss.item()
            running_domain_loss += domain_loss.item()

            # 分类准确率
            _, predicted = source_class_output.max(1)
            correct_class += predicted.eq(source_labels).sum().item()
            total_class += source_labels.size(0)

            # 域判别准确率
            domain_pred = (torch.sigmoid(combined_domain_output) > 0.5).float()
            correct_domain += domain_pred.squeeze().eq(combined_domains.squeeze()).sum().item()
            total_domain += combined_domains.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'λ': f'{lambda_p:.3f}',
                'C_loss': running_class_loss / (batch_idx + 1),
                'D_loss': running_domain_loss / (batch_idx + 1),
                'C_acc': 100. * correct_class / total_class,
                'D_acc': 100. * correct_domain / total_domain
            })

        epoch_class_loss = running_class_loss / n_batches
        epoch_domain_loss = running_domain_loss / n_batches
        epoch_class_acc = 100. * correct_class / total_class
        epoch_domain_acc = 100. * correct_domain / total_domain

        return epoch_class_loss, epoch_domain_loss, epoch_class_acc, epoch_domain_acc, lambda_p

    def validate(self, val_loader):
        """在有标签的验证集上评估"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels, _ in val_loader:
                data = data.to(self.device)
                labels = labels.squeeze().to(self.device)

                # 只需要分类输出
                class_output, _, _ = self.model(data, lambda_p=0)
                loss = self.class_criterion(class_output, labels)

                running_loss += loss.item()
                _, predicted = class_output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def evaluate_target_domain(self, target_loader_with_labels):
        """评估目标域性能（需要有标签）"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels, _ in target_loader_with_labels:
                data = data.to(self.device)
                labels = labels.squeeze().to(self.device)

                class_output, _, _ = self.model(data, lambda_p=0)
                _, predicted = class_output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        target_acc = 100. * correct / total
        return target_acc

    def train(self, source_loader, target_loader, val_loader, target_test_loader=None, epochs=50):
        """训练DANN"""
        best_val_acc = 0

        for epoch in range(epochs):
            # 训练
            class_loss, domain_loss, class_acc, domain_acc, lambda_p = self.train_epoch(
                source_loader, target_loader, epoch, epochs
            )

            # 验证（源域）
            val_loss, val_acc = self.validate(val_loader)

            # 记录历史
            self.history['train_class_loss'].append(class_loss)
            self.history['train_domain_loss'].append(domain_loss)
            self.history['train_class_acc'].append(class_acc)
            self.history['train_domain_acc'].append(domain_acc)
            self.history['val_class_loss'].append(val_loss)
            self.history['val_class_acc'].append(val_acc)

            # 如果有目标域测试集，评估性能
            if target_test_loader is not None:
                target_acc = self.evaluate_target_domain(target_test_loader)
                self.history['target_domain_acc'].append(target_acc)
                target_info = f", Target Acc: {target_acc:.2f}%"
            else:
                target_info = ""

            # 打印进度
            print(f"\nEpoch {epoch + 1}/{epochs} (λ={lambda_p:.3f}):")
            print(f"  Train - Class Loss: {class_loss:.4f}, Class Acc: {class_acc:.2f}%, "
                  f"Domain Loss: {domain_loss:.4f}, Domain Acc: {domain_acc:.2f}%")
            print(f"  Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%{target_info}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_acc': best_val_acc,
                }, 'best_dann_cnnlstm_model.pth')
                print(f"  --> New best model saved! (Val Acc: {val_acc:.2f}%)")

    def plot_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 分类损失
        ax = axes[0, 0]
        ax.plot(self.history['train_class_loss'], label='Train Class Loss')
        ax.plot(self.history['val_class_loss'], label='Val Class Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Classification Loss')
        ax.legend()
        ax.grid(True)

        # 域判别损失
        ax = axes[0, 1]
        ax.plot(self.history['train_domain_loss'], label='Domain Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Domain Discrimination Loss')
        ax.legend()
        ax.grid(True)

        # 分类准确率
        ax = axes[1, 0]
        ax.plot(self.history['train_class_acc'], label='Train Acc')
        ax.plot(self.history['val_class_acc'], label='Val Acc')
        if self.history['target_domain_acc']:
            ax.plot(self.history['target_domain_acc'], label='Target Domain Acc', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Classification Accuracy')
        ax.legend()
        ax.grid(True)

        # 域判别准确率
        ax = axes[1, 1]
        ax.plot(self.history['train_domain_acc'], label='Domain Acc')
        ax.axhline(y=50, color='r', linestyle='--', label='Random (50%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Domain Discrimination Accuracy')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()


def load_source_target_data(source_path, target_path, test_case_path=None):
    """加载源域和目标域数据"""
    modulation_types = [
        '4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
        '32qam', '64qam', '128qam', 'bpsk', 'ook', 'qpsk'
    ]

    # modulation_types = [
    #     '4ask', '8ask', '8psk', '16psk', '16qam',
    #     '32psk', '32qam', 'bpsk', 'ook', 'qpsk'
    # ]

    label_map = {mod: i for i, mod in enumerate(modulation_types)}

    def load_dataset(path, dataset_name):
        all_data = []
        all_labels = []

        print(f"\nLoading {dataset_name} from {path}")

        for mod_type in modulation_types:
            file_path = os.path.join(path, f'{mod_type}_seg512.mat')

            if os.path.exists(file_path):
                mat_data = loadmat(file_path)
                frames = mat_data['frames']  # (512, 2, N)
                complex_data = frames[:, 0, :].T  # (N, 512)

                n_samples = complex_data.shape[0]
                labels = np.full(n_samples, label_map[mod_type])

                all_data.append(complex_data)
                all_labels.append(labels)

                print(f"  {mod_type}: {n_samples} samples")

        X = np.vstack(all_data)
        y = np.hstack(all_labels)

        print(f"Total {dataset_name}: {X.shape}")

        return X, y

    # 加载数据
    X_source, y_source = load_dataset(source_path, "source domain")
    X_target_unlabeled, _ = load_dataset(target_path, "target domain (unlabeled)")

    # 如果有测试集路径，加载带标签的目标域测试集
    if test_case_path:
        X_target_test, y_target_test = load_dataset(test_case_path, "target domain test")
        return X_source, y_source, X_target_unlabeled, X_target_test, y_target_test

    return X_source, y_source, X_target_unlabeled, None, None


def main():
    # 路径设置
    source_path = r'D:\RadioData\dataset1'  # 源域（训练数据）
    target_path = r'D:\RadioData\testcase0714'  # 目标域（无标签）
    test_case_path = r'D:\RadioData\testcase0714'  # 目标域测试集（有标签）

    # 加载数据
    X_source, y_source, X_target_unlabeled, X_target_test, y_target_test = load_source_target_data(
        source_path, target_path, test_case_path
    )

    # 划分源域数据
    X_train, X_val, y_train, y_val = train_test_split(
        X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
    )

    print(f"\nSource domain split:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"Target domain:")
    print(f"  Unlabeled: {X_target_unlabeled.shape}")
    if X_target_test is not None:
        print(f"  Test: {X_target_test.shape}")

    # 创建数据集
    train_dataset = DANNDataset(X_train, y_train, domain_label=0)  # 源域
    val_dataset = DANNDataset(X_val, y_val, domain_label=0)  # 源域验证
    target_dataset = DANNDataset(X_target_unlabeled, domain_label=1)  # 目标域（无标签）

    # 数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    # 如果有目标域测试集
    target_test_loader = None
    if X_target_test is not None:
        target_test_dataset = DANNDataset(X_target_test, y_target_test, domain_label=1)
        target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型和训练器
    model = DANN(input_channels=2, num_classes=12)
    trainer = DANNTrainer(model, device)

    # 打印模型结构
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 训练
    print("\nTraining DANN...")
    trainer.train(
        train_loader,
        target_loader,
        val_loader,
        target_test_loader,
        epochs=20
    )

    # 绘制训练历史
    trainer.plot_history()

    # 评估最终性能
    if target_test_loader is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TARGET DOMAIN TEST SET")
        print("=" * 60)

        # 加载最佳模型
        checkpoint = torch.load('best_dann_cnnlstm_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 评估
        all_predictions = []
        all_targets = []
        all_features = []

        with torch.no_grad():
            for data, labels, _ in tqdm(target_test_loader, desc="Evaluating"):
                data = data.to(device)
                labels = labels.squeeze()

                class_output, _, features = model(data, lambda_p=0)
                _, predicted = class_output.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.numpy())
                all_features.extend(features.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_features = np.array(all_features)

        # 计算准确率
        accuracy = np.mean(all_predictions == all_targets)
        print(f"\nTarget Domain Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # 混淆矩阵
        # modulation_names = ['4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
        #                     '32qam', '64qam', '128qam', 'bpsk', 'ook', 'qpsk']

        modulation_names = ['4ask', '8ask', '8psk', '16psk', '16qam',
                            '32psk', '32qam', 'bpsk', 'ook', 'qpsk']

        cm = confusion_matrix(all_targets, all_predictions)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=modulation_names, yticklabels=modulation_names)
        plt.title(f'Confusion Matrix - Target Domain (Accuracy: {accuracy:.4f})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

        # 分类报告
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions,
                                    target_names=modulation_names, digits=4))

        # 可视化特征分布（t-SNE）
        print("\nVisualizing learned features...")
        from sklearn.manifold import TSNE

        # 对特征进行t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(all_features[:1000])  # 使用前1000个样本

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=all_targets[:1000], cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Learned Features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()