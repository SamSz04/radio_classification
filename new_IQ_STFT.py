import sys
import time
import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.signal import stft
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# Logger类（如果需要）
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 自定义数据集类 - 增强版，包含STFT计算
class RadioMLDataset(Dataset):
    def __init__(self, X, y, compute_stft=True, stft_params=None):
        """
        X: (N, 2, 1024) - I/Q数据
        y: (N, num_classes) - one-hot标签
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.compute_stft = compute_stft

        # STFT参数
        if stft_params is None:
            self.stft_params = {
                'nperseg': 64,
                'noverlap': 48,
                'nfft': 128
            }
        else:
            self.stft_params = stft_params

        # 预计算STFT（可选）
        if self.compute_stft:
            print("Computing STFT features...")
            self.stft_features = self._compute_all_stft()

    def _compute_stft(self, iq_signal):
        """计算单个信号的STFT"""
        # iq_signal: (2, 1024)
        # 构造复数信号
        complex_signal = iq_signal[0] + 1j * iq_signal[1]

        # 计算STFT
        f, t, Zxx = stft(complex_signal.numpy(),
                         nperseg=self.stft_params['nperseg'],
                         noverlap=self.stft_params['noverlap'],
                         nfft=self.stft_params['nfft'])

        # 取幅度谱并转换为dB
        stft_mag = np.abs(Zxx)
        stft_db = 20 * np.log10(stft_mag + 1e-10)

        return torch.FloatTensor(stft_db)

    def _compute_all_stft(self):
        """预计算所有样本的STFT"""
        stft_list = []
        for i in tqdm(range(len(self.X)), desc="Computing STFT"):
            stft_feat = self._compute_stft(self.X[i])
            stft_list.append(stft_feat)
        return torch.stack(stft_list)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        iq_data = self.X[idx]  # (2, 1024)
        label = self.y[idx]

        # 转换为 (1024, 2) 格式，适配我们的模型
        iq_data = iq_data.T  # (1024, 2)

        if self.compute_stft:
            stft_data = self.stft_features[idx].unsqueeze(0)  # 添加通道维度
            return iq_data, stft_data, label
        else:
            return iq_data, label


# ===== 模型定义 =====

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
        # x: (batch, 1024, 2)
        batch_size = x.size(0)

        # 添加通道维度: (batch, 1, 1024, 2)
        x = x.unsqueeze(1)

        # I/Q特征提取
        x = self.iq_feature_conv(x)  # (batch, 64, 1024, 1)
        x = x.squeeze(-1)  # (batch, 64, 1024)

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

    def __init__(self, num_classes=24, fusion_method='attention', use_stft=True):
        super(FusionModel, self).__init__()

        self.use_stft = use_stft
        self.fusion_method = fusion_method

        # 特征提取器
        self.iq_extractor = IQFeatureExtractor()
        if self.use_stft:
            self.stft_extractor = STFTFeatureExtractor()

        # 特征融合维度
        if not self.use_stft:
            fusion_dim = self.iq_extractor.feature_dim
        elif fusion_method == 'concat':
            fusion_dim = self.iq_extractor.feature_dim + self.stft_extractor.feature_dim
        elif fusion_method == 'add':
            fusion_dim = self.iq_extractor.feature_dim
        elif fusion_method == 'attention':
            fusion_dim = self.iq_extractor.feature_dim
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

    def forward(self, iq_data, stft_data=None):
        # 提取I/Q特征
        iq_features = self.iq_extractor(iq_data)

        if not self.use_stft or stft_data is None:
            # 只使用I/Q特征
            return self.classifier(iq_features)

        # 提取STFT特征
        stft_features = self.stft_extractor(stft_data)

        # 特征融合
        if self.fusion_method == 'concat':
            fused_features = torch.cat([iq_features, stft_features], dim=1)
        elif self.fusion_method == 'add':
            fused_features = iq_features + stft_features
        elif self.fusion_method == 'attention':
            iq_weight = self.attention_iq(iq_features)
            stft_weight = self.attention_stft(stft_features)

            total_weight = iq_weight + stft_weight + 1e-8
            iq_weight = iq_weight / total_weight
            stft_weight = stft_weight / total_weight

            fused_features = iq_weight * iq_features + stft_weight * stft_features

        # 分类
        output = self.classifier(fused_features)

        return output


# ===== 主训练脚本 =====

# 时间戳
current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
log_name = f"./logs/{current_time}_fusion.txt"

# 创建logs目录
if not os.path.exists('./logs'):
    os.makedirs('./logs')

sys.stdout = Logger(log_name)

# 加载数据
print("Loading RadioML 2018.01A dataset...")
dataset_file = h5py.File("data/GOLD_XYZ_OSC.0001_1024.hdf5", "r")

base_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                           'FM', 'GMSK', 'OQPSK']

selected_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                               'FM', 'GMSK', 'OQPSK']

selected_classes_id = [base_modulation_classes.index(cls) for cls in selected_modulation_classes]

N_SNR = 4  # from 30 SNR to 22 SNR

# 加载数据
X_data = None
y_data = None

for id in selected_classes_id:
    X_slice = dataset_file['X'][(106496 * (id + 1) - 4096 * N_SNR):106496 * (id + 1)]
    y_slice = dataset_file['Y'][(106496 * (id + 1) - 4096 * N_SNR):106496 * (id + 1)]

    if X_data is not None:
        X_data = np.concatenate([X_data, X_slice], axis=0)
        y_data = np.concatenate([y_data, y_slice], axis=0)
    else:
        X_data = X_slice
        y_data = y_slice

# X_data shape: (N, 1024, 2) -> (N, 2, 1024)
X_data = X_data.transpose(0, 2, 1)

# 处理标签
y_data_df = pd.DataFrame(y_data)
for column in y_data_df.columns:
    if sum(y_data_df[column]) == 0:
        y_data_df = y_data_df.drop(columns=[column])

y_data_df.columns = selected_modulation_classes
y_data = y_data_df.to_numpy()

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
print(f"Train data: {X_train.shape}, Train label: {y_train.shape}")
print(f"Test data: {X_test.shape}, Test label: {y_test.shape}")

# 超参数
batch_size = 64
epochs = 20
lr = 0.001
scheduler_step = 10
scheduler_gamma = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 是否使用STFT特征
USE_STFT = True  # 可以设置为False只使用I/Q特征
FUSION_METHOD = 'attention'  # 可选: 'concat', 'add', 'attention'

# 创建数据集
print(f"Creating datasets (USE_STFT={USE_STFT})...")
train_dataset = RadioMLDataset(X_train, y_train, compute_stft=USE_STFT)
test_dataset = RadioMLDataset(X_test, y_test, compute_stft=USE_STFT)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 创建模型
num_classes = y_train.shape[-1]
model = FusionModel(num_classes=num_classes, fusion_method=FUSION_METHOD, use_stft=USE_STFT).to(device)
print(f"\nModel: Fusion Model (fusion_method={FUSION_METHOD}, use_stft={USE_STFT})")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# 训练历史
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 训练循环
print("\nStarting training...")
best_val_acc = 0.0

for epoch in range(epochs):
    # 训练阶段
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
    for batch_data in train_bar:
        if USE_STFT:
            iq_data, stft_data, labels = batch_data
            iq_data = iq_data.to(device)
            stft_data = stft_data.to(device)
        else:
            iq_data, labels = batch_data
            iq_data = iq_data.to(device)
            stft_data = None

        labels = labels.to(device)

        # 前向传播
        output = model(iq_data, stft_data)
        loss = criterion(output, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        _, label_idx = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == label_idx).sum().item()

        train_bar.set_postfix({'loss': f'{loss.item():.4f}',
                               'acc': f'{100 * correct / total:.2f}%'})

    scheduler.step()
    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        val_bar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs} [Valid]')
        for batch_data in val_bar:
            if USE_STFT:
                iq_data, stft_data, labels = batch_data
                iq_data = iq_data.to(device)
                stft_data = stft_data.to(device)
            else:
                iq_data, labels = batch_data
                iq_data = iq_data.to(device)
                stft_data = None

            labels = labels.to(device)

            output = model(iq_data, stft_data)
            loss = criterion(output, labels)
            val_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            _, label_idx = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == label_idx).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label_idx.cpu().numpy())

            val_bar.set_postfix({'loss': f'{loss.item():.4f}',
                                 'acc': f'{100 * correct / total:.2f}%'})

    val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'\nEpoch [{epoch + 1}/{epochs}]')
    print(f'Train loss: {train_loss:.6f}, Train accuracy: {train_accuracy:.2f}%')
    print(f'Validation loss: {val_loss:.6f}, Validation accuracy: {val_accuracy:.2f}%')

    # 保存最佳模型
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }, f'./logs/{current_time}_best_fusion_model.pth')
        print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig(f'./logs/{current_time}_fusion_training_curves.png')
plt.close()

# 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_modulation_classes,
            yticklabels=selected_modulation_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - Fusion Model ({FUSION_METHOD})')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig(f'./logs/{current_time}_fusion_confusion_matrix.png')
plt.close()

# 打印最终结果
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Model: Fusion Model (fusion_method={FUSION_METHOD}, use_stft={USE_STFT})")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Test Accuracy: {val_accuracy:.2f}%")

# 详细的分类报告
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions,
                            target_names=selected_modulation_classes,
                            digits=4))

# 分析QAM性能
qam_classes = ['16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
qam_indices = [selected_modulation_classes.index(c) for c in qam_classes if c in selected_modulation_classes]

if qam_indices:
    qam_correct = 0
    qam_total = 0
    for i in qam_indices:
        qam_correct += cm[i, i]
        qam_total += cm[i, :].sum()

    qam_accuracy = qam_correct / qam_total * 100 if qam_total > 0 else 0
    print(f"\nQAM Classes Accuracy: {qam_accuracy:.2f}%")
    print(f"Overall Accuracy: {val_accuracy:.2f}%")
    print(f"Improvement on QAM: {qam_accuracy - val_accuracy:.2f}%")

print("\nTraining completed!")
print(f"All results saved in ./logs/ with timestamp {current_time}")