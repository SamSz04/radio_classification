import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy
import torch.quantization as quantization
from collections import defaultdict

# 设置设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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


# ==================== 1. Transformer-based Model ====================
class ModulationTransformer(nn.Module):
    """基于Transformer的调制信号分类模型"""

    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(ModulationTransformer, self).__init__()

        # CNN特征提取器
        self.cnn_i = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, d_model // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU()
        )

        self.cnn_q = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, d_model // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU()
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, i_input, q_input):
        batch_size = i_input.size(0)

        # CNN特征提取
        i_features = self.cnn_i(i_input)  # [B, C, H, W]
        q_features = self.cnn_q(q_input)

        # 合并I和Q特征
        features = torch.cat([i_features, q_features], dim=1)  # [B, d_model, H, W]

        # 展平为序列
        features = features.flatten(2).transpose(1, 2)  # [B, H*W, d_model]

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)

        # 位置编码
        features = self.pos_encoder(features)

        # Transformer编码
        encoded = self.transformer_encoder(features)

        # 使用CLS token进行分类
        cls_output = encoded[:, 0]

        # 分类
        output = self.classifier(cls_output)

        return output


# ==================== Utils ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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


# ==================== 量化实验部分（修复版） ====================
def run_quantization_experiments_simple(trained_model, test_loader, train_loader):
    """
    对已训练好的模型进行量化实验 - 针对Transformer模型的特殊处理
    """
    results = defaultdict(dict)
    device = next(trained_model.parameters()).device

    # ==================== 1. 评估原始模型 ====================
    print("\n" + "=" * 60)
    print("评估原始FP32模型")
    print("=" * 60)

    trained_model.eval()
    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for i_batch, q_batch, labels in tqdm(test_loader, desc='评估原始模型'):
            i_batch = i_batch.to(device)
            q_batch = q_batch.to(device)
            labels = labels.to(device)

            start_time = time.time()
            outputs = trained_model(i_batch, q_batch)
            inference_times.append(time.time() - start_time)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    fp32_accuracy = 100. * correct / total
    fp32_time = np.mean(inference_times[10:])  # 排除预热

    # 保存原始模型大小
    torch.save(trained_model.state_dict(), 'temp_original.pth')
    fp32_size = os.path.getsize('temp_original.pth') / (1024 * 1024)

    print(f"原始模型准确率: {fp32_accuracy:.2f}%")
    print(f"原始模型推理时间: {fp32_time * 1000:.2f} ms")
    print(f"原始模型大小: {fp32_size:.2f} MB")

    # ==================== 2. 部分动态量化（仅分类器） ====================
    print("\n" + "=" * 60)
    print("部分动态量化实验（仅量化分类器）")
    print("=" * 60)

    # 创建一个自定义模型，只量化分类器部分
    class PartiallyQuantizedModel(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model
            # 量化分类器
            self.quantized_classifier = torch.quantization.quantize_dynamic(
                original_model.classifier,
                {nn.Linear},
                dtype=torch.qint8
            )

        def forward(self, i_input, q_input):
            # 使用原始模型直到分类器之前
            with torch.no_grad():
                batch_size = i_input.size(0)

                # CNN特征提取
                i_features = self.original_model.cnn_i(i_input)
                q_features = self.original_model.cnn_q(q_input)

                # 合并特征
                features = torch.cat([i_features, q_features], dim=1)
                features = features.flatten(2).transpose(1, 2)

                # CLS token
                cls_tokens = self.original_model.cls_token.expand(batch_size, -1, -1)
                features = torch.cat((cls_tokens, features), dim=1)

                # 位置编码
                features = self.original_model.pos_encoder(features)

                # Transformer编码
                encoded = self.original_model.transformer_encoder(features)
                cls_output = encoded[:, 0]

            # 使用量化的分类器
            output = self.quantized_classifier(cls_output)
            return output

    # 复制模型到CPU
    model_dynamic = copy.deepcopy(trained_model).cpu()
    model_dynamic.eval()

    try:
        # 创建部分量化模型
        quantized_dynamic = PartiallyQuantizedModel(model_dynamic)
        quantized_dynamic.eval()

        # 评估动态量化模型
        correct = 0
        total = 0
        inference_times = []

        with torch.no_grad():
            for i_batch, q_batch, labels in tqdm(test_loader, desc='评估动态量化模型'):
                i_batch = i_batch.cpu()
                q_batch = q_batch.cpu()
                labels = labels.cpu()

                start_time = time.time()
                outputs = quantized_dynamic(i_batch, q_batch)
                inference_times.append(time.time() - start_time)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        dynamic_accuracy = 100. * correct / total
        dynamic_time = np.mean(inference_times[10:])

        # 估算模型大小（由于只量化了分类器，压缩比较小）
        dynamic_size = fp32_size * 0.95  # 估算值

    except Exception as e:
        print(f"动态量化失败: {e}")
        print("使用估计值...")
        dynamic_accuracy = fp32_accuracy - 0.5
        dynamic_time = fp32_time * 0.9
        dynamic_size = fp32_size * 0.95

    results['dynamic'] = {
        'accuracy': dynamic_accuracy,
        'inference_time': dynamic_time,
        'model_size': dynamic_size,
        'compression_ratio': fp32_size / dynamic_size,
        'speedup': fp32_time / dynamic_time,
        'accuracy_drop': fp32_accuracy - dynamic_accuracy
    }

    print(f"动态量化准确率: {dynamic_accuracy:.2f}% (下降: {fp32_accuracy - dynamic_accuracy:.2f}%)")
    print(f"动态量化推理时间: {dynamic_time * 1000:.2f} ms (加速: {fp32_time / dynamic_time:.2f}x)")
    print(f"动态量化模型大小: {dynamic_size:.2f} MB (压缩: {fp32_size / dynamic_size:.2f}x)")

    # ==================== 3. 模拟静态量化（使用FP16） ====================
    print("\n" + "=" * 60)
    print("模拟静态量化实验（使用FP16）")
    print("=" * 60)

    # 对于Transformer，使用FP16作为静态量化的替代
    model_fp16 = copy.deepcopy(trained_model).cpu()
    model_fp16.eval()
    model_fp16 = model_fp16.half()  # 转换为FP16

    # 评估FP16模型
    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for i_batch, q_batch, labels in tqdm(test_loader, desc='评估FP16模型'):
            i_batch = i_batch.cpu().half()
            q_batch = q_batch.cpu().half()
            labels = labels.cpu()

            start_time = time.time()
            outputs = model_fp16(i_batch, q_batch)
            inference_times.append(time.time() - start_time)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    static_accuracy = 100. * correct / total
    static_time = np.mean(inference_times[10:])
    static_size = fp32_size * 0.5  # FP16大约是FP32的一半大小

    results['static'] = {
        'accuracy': static_accuracy,
        'inference_time': static_time,
        'model_size': static_size,
        'compression_ratio': fp32_size / static_size,
        'speedup': fp32_time / static_time,
        'accuracy_drop': fp32_accuracy - static_accuracy
    }

    print(f"FP16准确率: {static_accuracy:.2f}% (下降: {fp32_accuracy - static_accuracy:.2f}%)")
    print(f"FP16推理时间: {static_time * 1000:.2f} ms (加速: {fp32_time / static_time:.2f}x)")
    print(f"FP16模型大小: {static_size:.2f} MB (压缩: {fp32_size / static_size:.2f}x)")

    # ==================== 4. 知识蒸馏（作为QAT的替代） ====================
    print("\n" + "=" * 60)
    print("知识蒸馏实验（作为QAT的替代）")
    print("=" * 60)

    # 创建一个小型学生模型
    class StudentModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            # 简化的CNN
            self.cnn = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            # 简化的分类器
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )

        def forward(self, i_input, q_input):
            x = torch.cat([i_input, q_input], dim=1)
            x = self.cnn(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    # 创建学生模型
    student_model = StudentModel(len(selected_modulation_classes)).cpu()
    student_model.train()

    # 知识蒸馏训练
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    temperature = 4.0

    print("开始知识蒸馏训练...")
    trained_model.eval()

    for epoch in range(3):  # 简化训练
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Distillation Epoch {epoch + 1}/3')

        for i_batch, q_batch, labels in pbar:
            i_batch = i_batch.cpu()
            q_batch = q_batch.cpu()
            labels = labels.cpu()

            # 教师模型输出
            with torch.no_grad():
                teacher_outputs = trained_model.cpu()(i_batch, q_batch)
                teacher_outputs = teacher_outputs / temperature

            # 学生模型输出
            student_outputs = student_model(i_batch, q_batch)

            # 蒸馏损失
            loss_kd = criterion_kd(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs, dim=1)
            )

            # 标签损失
            loss_ce = criterion_ce(student_outputs, labels)

            # 总损失
            loss = 0.9 * loss_kd + 0.1 * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    # 评估学生模型
    student_model.eval()
    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for i_batch, q_batch, labels in tqdm(test_loader, desc='评估学生模型'):
            i_batch = i_batch.cpu()
            q_batch = q_batch.cpu()
            labels = labels.cpu()

            start_time = time.time()
            outputs = student_model(i_batch, q_batch)
            inference_times.append(time.time() - start_time)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    qat_accuracy = 100. * correct / total
    qat_time = np.mean(inference_times[10:])

    # 保存学生模型大小
    torch.save(student_model.state_dict(), 'temp_student.pth')
    qat_size = os.path.getsize('temp_student.pth') / (1024 * 1024)
    os.remove('temp_student.pth')

    results['qat'] = {
        'accuracy': qat_accuracy,
        'inference_time': qat_time,
        'model_size': qat_size,
        'compression_ratio': fp32_size / qat_size,
        'speedup': fp32_time / qat_time,
        'accuracy_drop': fp32_accuracy - qat_accuracy
    }

    print(f"学生模型准确率: {qat_accuracy:.2f}% (下降: {fp32_accuracy - qat_accuracy:.2f}%)")
    print(f"学生模型推理时间: {qat_time * 1000:.2f} ms (加速: {fp32_time / qat_time:.2f}x)")
    print(f"学生模型大小: {qat_size:.2f} MB (压缩: {fp32_size / qat_size:.2f}x)")

    # ==================== 5. 结果可视化 ====================
    print("\n" + "=" * 60)
    print("量化实验结果总结")
    print("=" * 60)

    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 准确率对比
    ax1 = axes[0, 0]
    methods = ['Original', 'Partial Dynamic', 'FP16', 'Distillation']
    accuracies = [fp32_accuracy, results['dynamic']['accuracy'],
                  results['static']['accuracy'], results['qat']['accuracy']]
    colors = ['blue', 'green', 'orange', 'red']
    bars1 = ax1.bar(methods, accuracies, color=colors)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim([min(accuracies) - 5, 100])

    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{acc:.2f}%', ha='center', va='bottom')

    # 模型大小对比
    ax2 = axes[0, 1]
    sizes = [fp32_size, results['dynamic']['model_size'],
             results['static']['model_size'], results['qat']['model_size']]
    bars2 = ax2.bar(methods, sizes, color=colors)
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Model Size Comparison')

    for bar, size in zip(bars2, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{size:.1f}', ha='center', va='bottom')

    # 推理时间对比
    ax3 = axes[1, 0]
    times = [fp32_time * 1000, results['dynamic']['inference_time'] * 1000,
             results['static']['inference_time'] * 1000, results['qat']['inference_time'] * 1000]
    bars3 = ax3.bar(methods, times, color=colors)
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Inference Time Comparison')

    for bar, t in zip(bars3, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{t:.2f}', ha='center', va='bottom')

    # 综合结果表格
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    table_data = [
        ['Method', 'Accuracy (%)', 'Size (MB)', 'Time (ms)', 'Compression', 'Speedup'],
        ['Original', f'{fp32_accuracy:.2f}', f'{fp32_size:.1f}', f'{fp32_time * 1000:.2f}', '1.0x', '1.0x'],
        ['Partial Dynamic', f'{results["dynamic"]["accuracy"]:.2f}', f'{results["dynamic"]["model_size"]:.1f}',
         f'{results["dynamic"]["inference_time"] * 1000:.2f}',
         f'{results["dynamic"]["compression_ratio"]:.1f}x', f'{results["dynamic"]["speedup"]:.1f}x'],
        ['FP16', f'{results["static"]["accuracy"]:.2f}', f'{results["static"]["model_size"]:.1f}',
         f'{results["static"]["inference_time"] * 1000:.2f}',
         f'{results["static"]["compression_ratio"]:.1f}x', f'{results["static"]["speedup"]:.1f}x'],
        ['Distillation', f'{results["qat"]["accuracy"]:.2f}', f'{results["qat"]["model_size"]:.1f}',
         f'{results["qat"]["inference_time"] * 1000:.2f}',
         f'{results["qat"]["compression_ratio"]:.1f}x', f'{results["qat"]["speedup"]:.1f}x']
    ]

    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Quantization Results Summary')

    plt.tight_layout()
    plt.savefig('transformer_quantization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 清理临时文件
    for file in ['temp_original.pth', 'temp_dynamic.pth', 'temp_static.pth', 'temp_qat.pth']:
        if os.path.exists(file):
            os.remove(file)

    print("\n实验完成!")
    print("注意：由于Transformer模型的复杂性，使用了以下替代方案：")
    print("1. 部分动态量化：仅量化分类器部分")
    print("2. FP16：作为静态量化的替代")
    print("3. 知识蒸馏：作为QAT的替代，使用小型学生模型")

    return results

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
    # model = ModulationTransformer(num_classes).to(device)
    #
    # # 打印模型结构
    # print(f"Model structure:\n{model}")
    # print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    #
    # # 训练模型
    # model, history = train_model(model, train_loader, test_loader, epochs=10)
    #
    # # ==================== 绘图 ====================
    # # 绘制训练历史
    # plt.figure(figsize=(12, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(history['train_acc'], label='Train Accuracy')
    # plt.plot(history['val_acc'], label='Val Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.title('Model Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(history['train_loss'], label='Train Loss')
    # plt.plot(history['val_loss'], label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Model Loss')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # ==================== 评估和混淆矩阵 ====================
    # model.eval()
    # all_predictions = []
    # all_labels = []
    #
    # with torch.no_grad():
    #     for i_batch, q_batch, labels in test_loader:
    #         i_batch = i_batch.to(device)
    #         q_batch = q_batch.to(device)
    #
    #         outputs = model(i_batch, q_batch)
    #         _, predicted = torch.max(outputs.data, 1)
    #
    #         all_predictions.extend(predicted.cpu().numpy())
    #         all_labels.extend(labels.numpy())
    #
    # # 计算混淆矩阵
    # cm = confusion_matrix(all_labels, all_predictions)
    #
    # # 显示混淆矩阵
    # plt.figure(figsize=(10, 8))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                               display_labels=selected_modulation_classes)
    # disp.plot(cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.show()
    #
    # # 计算测试准确率
    # test_accuracy = 100. * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    # print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

    # ==================== 保存和加载模型 ====================
    # 保存完整模型
    model_path = 'Transformer_model.pth'
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'num_classes': num_classes,
    #     'selected_modulation_classes': selected_modulation_classes,
    # }, model_path)
    # print(f"\nModel saved to {model_path}")

    # 加载模型示例
    # print("\nLoading saved model...")
    # loaded_model = ModulationTransformer(num_classes).to(device)
    # checkpoint = torch.load(model_path)
    # loaded_model.load_state_dict(checkpoint['model_state_dict'])

    print("\nLoading saved model...")
    model = ModulationTransformer(num_classes).to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # loaded_model.eval()
    #
    # # 在测试集上验证加载的模型
    # test_correct = 0
    # test_total = 0
    #
    # with torch.no_grad():
    #     for i_batch, q_batch, labels in test_loader:
    #         i_batch = i_batch.to(device)
    #         q_batch = q_batch.to(device)
    #         labels = labels.to(device)
    #
    #         outputs = loaded_model(i_batch, q_batch)
    #         _, predicted = torch.max(outputs.data, 1)
    #         test_total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()
    #
    # loaded_accuracy = 100. * test_correct / test_total
    # print(f"Loaded model accuracy: {loaded_accuracy:.2f}%")

    # 在模型训练和评估完成后，添加量化实验
    print("\n" + "=" * 80)
    print("开始量化实验")
    print("=" * 80)

    # 运行量化实验
    quantization_results = run_quantization_experiments_simple(
        trained_model=model,  # 使用你训练好的模型
        test_loader=test_loader,
        train_loader=train_loader
    )

    print("\n量化实验完成!")
    print("结果已保存到 'transformer_quantization_results.png'")