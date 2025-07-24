import sys
import time
import os
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from logger import Logger
from model import LSTMGRU
from myDataset import MyDataset

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
log_name = f"./logs/{current_time}.txt"
sys.stdout = Logger(log_name)

dataset_file = h5py.File("data/GOLD_XYZ_OSC.0001_1024.hdf5", "r")

base_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                           'FM', 'GMSK', 'OQPSK']

selected_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                               '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                               '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                               'FM', 'GMSK', 'OQPSK']

# 8 classes
# selected_modulation_classes = ['4ASK', 'BPSK', 'QPSK', '16PSK', '16QAM', 'FM', 'AM-DSB-WC', '32APSK']
# 24 classes
# selected_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
#                                '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
#                                '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
#                                'FM', 'GMSK', 'OQPSK']
# 20 classes
# selected_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK',
#                                '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
#                                'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
#                                'FM', 'GMSK', 'OQPSK']

selected_classes_id = [base_modulation_classes.index(cls) for cls in selected_modulation_classes]

N_SNR = 4 # from 30 SNR to 22 SNR

X_data = None
y_data = None

for id in selected_classes_id:
    X_slice = dataset_file['X'][(106496*(id+1) - 4096*N_SNR):106496 *(id +1)]
    y_slice = dataset_file['Y'][(106496*(id+1) - 4096*N_SNR):106496 *(id +1)]

    if X_data is not None:
        X_data = np.concatenate([X_data, X_slice], axis=0)
        y_data = np.concatenate([y_data, y_slice], axis=0)
    else:
        X_data = X_slice
        y_data = y_slice

X_data = X_data.transpose(0, 2, 1)
# X_data = X_data.reshape(len(X_data), -1, 32, 32)

y_data_df = pd.DataFrame(y_data)
for column in y_data_df.columns:
    if sum(y_data_df[column]) == 0:
        y_data_df = y_data_df.drop(columns=[column])

y_data_df.columns = selected_modulation_classes
y_data = y_data_df.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
print(f"train data: {X_train.shape} train label: {y_train.shape}")
print(f"test data: {X_test.shape} test label: {y_test.shape}")

batch_size = 64
epochs = 20
lr = 0.001
scheduler_step = 10
scheduler_gamma = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataset = MyDataset(torch.Tensor(X_train).float(), torch.Tensor(y_train).float())
test_dataset = MyDataset(torch.Tensor(X_test).float(), torch.Tensor(y_test).float())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = LSTMGRU(y_train.shape[-1]).to(device)
print(f"model is LSTMGRU")

criterion_doa = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in tqdm(range(epochs), desc='Epochs'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_data, batch_label in train_loader:
        batch_data, batch_label = batch_data.to(device), batch_label.to(device)
        output = model(batch_data)
        loss = criterion_doa(output, batch_label)
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        _, labels = torch.max(batch_label.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_label in test_loader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            output = model(batch_data)
            loss = criterion_doa(output, batch_label)
            val_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            _, labels = torch.max(batch_label.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct / total

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'\nEpoch [{epoch + 1}/{epochs}]')
    print(f'Train loss: {train_loss:.6f}, Train accuracy: {train_accuracy:.2f}%')
    print(f'Validation loss: {val_loss:.6f}, Validation accuracy: {val_accuracy:.2f}%')

#损失和准确率曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig(f'./logs/{current_time}_LSTMGRU_training_curves.png')
plt.close()

#混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_modulation_classes,
            yticklabels=selected_modulation_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig(f'./logs/{current_time}_LSTMGRU_confusion_matrix.png')
plt.close()





