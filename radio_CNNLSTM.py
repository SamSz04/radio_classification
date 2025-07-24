import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 数据加载部分 ====================
dataset_file = h5py.File("data/GOLD_XYZ_OSC.0001_1024.hdf5", "r")

base_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                           'FM', 'GMSK', 'OQPSK']

selected_modulation_classes = ['4ASK', 'BPSK', 'QPSK', '16PSK', '16QAM', 'FM', 'AM-DSB-WC', '32APSK']
# selected_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK',
#                               '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
selected_classes_id = [base_modulation_classes.index(cls) for cls in selected_modulation_classes]

N_SNR = 4  # from 30 SNR to 22 SNR

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

X_data = X_data.reshape(len(X_data), 32, 32, 2)

# 处理标签
y_data_df = pd.DataFrame(y_data)
for column in y_data_df.columns:
    if sum(y_data_df[column]) == 0:
        y_data_df = y_data_df.drop(columns=[column])

y_data_df.columns = selected_modulation_classes

# 转换标签为类别索引
y_labels = np.argmax(y_data_df.values, axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_labels, test_size=0.2, random_state=42
)


# ==================== PyTorch数据集类 ====================
class ModulationDataset(Dataset):
    def __init__(self, X_data, y_data):
        # X_data shape: (N, 32, 32, 2)
        # 分离I和Q通道
        self.i_data = torch.FloatTensor(X_data[:, :, :, 0]).unsqueeze(1)  # (N, 1, 32, 32)
        self.q_data = torch.FloatTensor(X_data[:, :, :, 1]).unsqueeze(1)  # (N, 1, 32, 32)
        self.labels = torch.LongTensor(y_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.i_data[idx], self.q_data[idx], self.labels[idx]


# ==================== 模型定义 ====================
class DualCNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(DualCNNLSTM, self).__init__()

        # CNN分支 - 用于Q输入
        self.cnn_q = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),  # 添加批标准化
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        # CNN分支 - 用于I输入
        self.cnn_i = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),  # 添加批标准化
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        # 计算CNN输出的特征维度
        cnn_output_size = 128 * 4 * 4
        concat_size = cnn_output_size * 2

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=concat_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(256, 32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, num_classes)
        )

    def forward(self, i_input, q_input):
        # 处理Q输入
        q_features = self.cnn_q(q_input)
        q_features = q_features.view(q_features.size(0), -1)

        # 处理I输入
        i_features = self.cnn_i(i_input)
        i_features = i_features.view(i_features.size(0), -1)

        # 拼接特征
        concat_features = torch.cat([q_features, i_features], dim=1)

        # 为LSTM准备输入
        lstm_input = concat_features.unsqueeze(1)

        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        output = self.fc(lstm_out)

        return output


# ==================== 训练相关类 ====================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.001, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=30, learning_rate=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 早停和模型检查点
    early_stopping = EarlyStopping(patience=10, verbose=True)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for i_batch, q_batch, labels in train_pbar:
            i_batch = i_batch.to(device)
            q_batch = q_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(i_batch, q_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({'loss': loss.item(),
                                    'acc': 100. * train_correct / train_total})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            for i_batch, q_batch, labels in val_pbar:
                i_batch = i_batch.to(device)
                q_batch = q_batch.to(device)
                labels = labels.to(device)

                outputs = model(i_batch, q_batch)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix({'loss': loss.item(),
                                      'acc': 100. * val_correct / val_total})

        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100. * val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        print(f'\nEpoch [{epoch + 1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%')
        print('-' * 60)

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 早停检查
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, history


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 创建数据集和数据加载器
    train_dataset = ModulationDataset(X_train, y_train)
    test_dataset = ModulationDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 创建模型
    num_classes = len(selected_modulation_classes)
    model = DualCNNLSTM(num_classes).to(device)

    # 打印模型结构
    print(f"Model structure:\n{model}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    model, history = train_model(model, train_loader, test_loader, epochs=50)

    # ==================== 绘图 ====================
    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.tight_layout()
    plt.show()

    # ==================== 评估和混淆矩阵 ====================
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i_batch, q_batch, labels in test_loader:
            i_batch = i_batch.to(device)
            q_batch = q_batch.to(device)

            outputs = model(i_batch, q_batch)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 显示混淆矩阵
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=selected_modulation_classes)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # 计算测试准确率
    test_accuracy = 100. * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

    # ==================== 保存和加载模型 ====================
    # 保存完整模型
    model_path = 'CNN_LSTM_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'selected_modulation_classes': selected_modulation_classes,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # 加载模型示例
    print("\nLoading saved model...")
    loaded_model = DualCNNLSTM(num_classes).to(device)
    checkpoint = torch.load(model_path)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    # 在测试集上验证加载的模型
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for i_batch, q_batch, labels in test_loader:
            i_batch = i_batch.to(device)
            q_batch = q_batch.to(device)
            labels = labels.to(device)

            outputs = loaded_model(i_batch, q_batch)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    loaded_accuracy = 100. * test_correct / test_total
    print(f"Loaded model accuracy: {loaded_accuracy:.2f}%")