import sys
import time
import os
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.io import loadmat
from logger import Logger
from model import LSTMGRU
from myDataset import MyDataset

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
log_name = f"./logs/{current_time}.txt"
sys.stdout = Logger(log_name)

# --- 数据加载 & 预处理 ---
dataset_file = h5py.File("data/GOLD_XYZ_OSC.0001_1024.hdf5", "r")

base_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                           'FM', 'GMSK', 'OQPSK']

selected_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                               'FM', 'GMSK', 'OQPSK']
selected_classes_id = [base_modulation_classes.index(c) for c in selected_modulation_classes]
N_SNR = 4  # 从 30 dB 到 22 dB

X_data = None
y_data = None
for idx in selected_classes_id:
    start = 106496 * (idx + 1) - 4096 * N_SNR
    end = 106496 * (idx + 1)
    X_slice = dataset_file['X'][start:end]
    y_slice = dataset_file['Y'][start:end]
    if X_data is None:
        X_data, y_data = X_slice, y_slice
    else:
        X_data = np.concatenate([X_data, X_slice], axis=0)
        y_data = np.concatenate([y_data, y_slice], axis=0)

X_data = X_data.transpose(0, 2, 1)  # (N, 2, 1024)

y_df = pd.DataFrame(y_data)
for col in y_df.columns:
    if y_df[col].sum() == 0:
        y_df.drop(columns=[col], inplace=True)
y_df.columns = selected_modulation_classes
y = y_df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)
print(f"train data: {X_train.shape}   train label: {y_train.shape}")
print(f" test data: {X_test.shape}    test label: {y_test.shape}")

# --- 超参 & 设备 ---
batch_size = 64
epochs = 20
lr = 0.001
sched_step = 10
sched_gamma = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# --- 数据加载器 ---
train_ds = MyDataset(torch.Tensor(X_train).float(), torch.Tensor(y_train).float())
test_ds = MyDataset(torch.Tensor(X_test).float(), torch.Tensor(y_test).float())
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# --- 模型 / 损失 / 优化器 ---
model = LSTMGRU(y_train.shape[-1]).to(device)
print("Model:", model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)

# --- 训练 & 验证循环（带详细进度显示） ---
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_acc = 0.0
best_model_path = f'./logs/{current_time}_best_model.pth'

for epoch in range(1, epochs + 1):
    # --- 训练 ---
    model.train()
    run_loss, run_correct, run_total = 0.0, 0, 0
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]', unit='batch')
    for X_batch, y_batch in train_bar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        # 更新统计
        bs = X_batch.size(0)
        run_loss += loss.item() * bs
        _, p_label = preds.max(dim=1)
        _, t_label = y_batch.max(dim=1)
        run_correct += (p_label == t_label).sum().item()
        run_total += bs

        train_bar.set_postfix({
            'loss': f'{run_loss / run_total:.4f}',
            'acc': f'{100 * run_correct / run_total:.1f}%'
        })
    scheduler.step()
    epoch_train_loss = run_loss / run_total
    epoch_train_acc = 100 * run_correct / run_total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)
    print(f'\nEpoch {epoch}/{epochs} TRAIN → loss: {epoch_train_loss:.4f}, acc: {epoch_train_acc:.2f}%')

    # --- 验证 ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_bar = tqdm(test_loader, desc=f'Epoch {epoch}/{epochs} [Valid]', unit='batch')
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in val_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            all_preds.append(preds.cpu())

            bs = X_batch.size(0)
            val_loss += loss.item() * bs
            _, p_label = preds.max(dim=1)
            _, t_label = y_batch.max(dim=1)
            val_correct += (p_label == t_label).sum().item()
            val_total += bs

            val_bar.set_postfix({
                'loss': f'{val_loss / val_total:.4f}',
                'acc': f'{100 * val_correct / val_total:.1f}%'
            })

    epoch_val_loss = val_loss / val_total
    epoch_val_acc = 100 * val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)
    print(f'Epoch {epoch}/{epochs} VALID → loss: {epoch_val_loss:.4f}, acc: {epoch_val_acc:.2f}%\n')

    # 保存最佳模型
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }, best_model_path)
        print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')

# 合并所有预测结果用于混淆矩阵
all_preds = torch.cat(all_preds, dim=0)

# --- 结果可视化 ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.legend();
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch');
plt.ylabel('Accuracy (%)');
plt.legend();
plt.title('Accuracy Curve')

plt.tight_layout()
plt.savefig(f'./logs/{current_time}_LSTMGRU_curves.png')
plt.close()

cm = confusion_matrix([np.argmax(x) for x in y_test], [np.argmax(x) for x in all_preds.numpy()])
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_modulation_classes,
            yticklabels=selected_modulation_classes)
plt.title('Confusion Matrix - Original Test Set')
plt.xlabel('Predicted');
plt.ylabel('True')
plt.tight_layout()
plt.savefig(f'./logs/{current_time}_LSTMGRU_confmat.png')
plt.close()


# ==================== 新增部分：在自定义测试集上评估 ====================

# def load_custom_testset(test_path):
#     """加载自定义测试集数据"""
#     # 自定义测试集的调制类型
#     custom_modulation_types = ['4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
#                                '32qam', '64qam', '128qam', 'bpsk', 'ook', 'qpsk']
#
#     # 映射到训练集的类别名称（注意大小写）
#     modulation_mapping = {
#         '4ask': '4ASK', '8ask': '8ASK', '8psk': '8PSK', '16psk': '16PSK',
#         '16qam': '16QAM', '32psk': '32PSK', '32qam': '32QAM', '64qam': '64QAM',
#         '128qam': '128QAM', 'bpsk': 'BPSK', 'ook': 'OOK', 'qpsk': 'QPSK'
#     }
#
#     all_data = []
#     all_labels = []
#
#     print(f"\nLoading custom test data from: {test_path}")
#
#     for mod_type in custom_modulation_types:
#         file_path = os.path.join(test_path, f'{mod_type}_seg1024.mat')
#
#         if os.path.exists(file_path):
#             try:
#                 # 加载mat文件
#                 mat_data = loadmat(file_path)
#
#                 # 尝试不同的变量名
#                 if 'frames' in mat_data:
#                     frames = mat_data['frames']  # (1024, 2, N)
#                 else:
#                     # 获取第一个非私有变量
#                     possible_keys = [k for k in mat_data.keys() if not k.startswith('__')]
#                     if possible_keys:
#                         frames = mat_data[possible_keys[0]]
#
#                 # 确保维度正确，并只取天线0的数据
#                 if frames.shape[0] == 1024 and frames.shape[1] >= 1:
#                     # 取天线0的复数数据
#                     complex_data = frames[:, 0, :]  # (1024, N)
#
#                     # 将复数拆分为I和Q
#                     I_data = np.real(complex_data)
#                     Q_data = np.imag(complex_data)
#
#                     # 组合为 (N, 2, 1024) 格式
#                     n_samples = complex_data.shape[1]
#                     data = np.zeros((n_samples, 2, 1024))
#                     data[:, 0, :] = I_data.T  # I通道
#                     data[:, 1, :] = Q_data.T  # Q通道
#
#                     # 创建one-hot标签
#                     mapped_mod = modulation_mapping[mod_type]
#                     if mapped_mod in selected_modulation_classes:
#                         label_idx = selected_modulation_classes.index(mapped_mod)
#                         labels = np.zeros((n_samples, len(selected_modulation_classes)))
#                         labels[:, label_idx] = 1
#
#                         all_data.append(data)
#                         all_labels.append(labels)
#
#                         print(f"  {mod_type}: {n_samples} samples loaded (mapped to {mapped_mod}, index {label_idx})")
#                     else:
#                         print(f"  Warning: {mapped_mod} not in training classes, skipping...")
#                 else:
#                     print(f"  Error: Unexpected data shape in {mod_type}: {frames.shape}")
#
#             except Exception as e:
#                 print(f"  Error loading {mod_type}: {str(e)}")
#         else:
#             print(f"  File not found: {file_path}")
#
#     if len(all_data) == 0:
#         raise ValueError("No custom test data loaded!")
#
#     # 合并所有数据
#     X_custom = np.vstack(all_data)
#     y_custom = np.vstack(all_labels)
#
#     print(f"\nTotal custom test samples: {X_custom.shape}")
#     print(f"Custom test labels shape: {y_custom.shape}")
#
#     return X_custom, y_custom, custom_modulation_types, modulation_mapping
#
#
# def evaluate_on_custom_testset(model, X_custom, y_custom, custom_modulation_types, modulation_mapping, device):
#     """在自定义测试集上评估模型"""
#     print("\n" + "=" * 60)
#     print("EVALUATION ON CUSTOM TEST SET")
#     print("=" * 60)
#
#     # 创建数据加载器
#     custom_dataset = MyDataset(torch.Tensor(X_custom).float(), torch.Tensor(y_custom).float())
#     custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
#
#     # 评估
#     model.eval()
#     all_preds = []
#     all_targets = []
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for X_batch, y_batch in tqdm(custom_loader, desc='Evaluating on custom test set'):
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#
#             preds = model(X_batch)
#
#             _, p_label = preds.max(dim=1)
#             _, t_label = y_batch.max(dim=1)
#
#             correct += (p_label == t_label).sum().item()
#             total += X_batch.size(0)
#
#             all_preds.extend(p_label.cpu().numpy())
#             all_targets.extend(t_label.cpu().numpy())
#
#     accuracy = 100 * correct / total
#     print(f"\nCustom Test Set Accuracy: {accuracy:.2f}%")
#
#     # 获取实际出现在测试集中的类别
#     unique_targets = np.unique(all_targets)
#     unique_preds = np.unique(all_preds)
#     all_unique = np.unique(np.concatenate([unique_targets, unique_preds]))
#
#     # 创建类别名称列表（只包含实际出现的类别）
#     present_classes = []
#     present_indices = []
#     for idx in all_unique:
#         if idx < len(selected_modulation_classes):
#             present_classes.append(selected_modulation_classes[idx])
#             present_indices.append(idx)
#
#     # 详细的分类报告
#     print("\nClassification Report:")
#     print(classification_report(all_targets, all_preds,
#                                 target_names=selected_modulation_classes,
#                                 labels=range(len(selected_modulation_classes))))
#
#     # 混淆矩阵
#     cm = confusion_matrix(all_targets, all_preds, labels=present_indices)
#
#     # 绘制混淆矩阵
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=present_classes,
#                 yticklabels=present_classes)
#     plt.title(f'Confusion Matrix - Custom Test Set (Accuracy: {accuracy:.2f}%)')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.tight_layout()
#     plt.savefig(f'./logs/{current_time}_custom_testset_confmat.png')
#     plt.close()
#
#     # 每个类别的准确率
#     print("\nPer-class Accuracy:")
#     for i, mod_type in enumerate(custom_modulation_types):
#         mapped_mod = modulation_mapping[mod_type]
#         if mapped_mod in selected_modulation_classes:
#             class_idx = selected_modulation_classes.index(mapped_mod)
#             mask = np.array(all_targets) == class_idx
#             if mask.sum() > 0:
#                 class_preds = np.array(all_preds)[mask]
#                 class_acc = (class_preds == class_idx).sum() / mask.sum() * 100
#                 print(f"  {mod_type:8s} ({mapped_mod:8s}): {class_acc:.2f}% ({mask.sum()} samples)")
#
#     return accuracy, cm
#
#
# # 加载最佳模型
# print("\nLoading best model...")
# checkpoint = torch.load(best_model_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['best_val_acc']:.2f}%")
#
# # 在自定义测试集上评估
# custom_test_path = r'D:\RadioData\testcase1024'
# try:
#     X_custom, y_custom, custom_mod_types, mod_mapping = load_custom_testset(custom_test_path)
#     custom_accuracy, custom_cm = evaluate_on_custom_testset(
#         model, X_custom, y_custom, custom_mod_types, mod_mapping, device
#     )
#
#     print("\n" + "=" * 60)
#     print("SUMMARY")
#     print("=" * 60)
#     print(f"Original Test Set Accuracy: {val_accs[-1]:.2f}%")
#     print(f"Custom Test Set Accuracy: {custom_accuracy:.2f}%")
#
# except Exception as e:
#     print(f"\nError evaluating on custom test set: {str(e)}")
#     import traceback
#
#     traceback.print_exc()
#
# print(f"\nAll results saved in ./logs/ with timestamp {current_time}")