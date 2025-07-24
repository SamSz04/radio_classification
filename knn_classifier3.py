import os
import pickle
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 导入KNN 分类器
from sklearn.neighbors import KNeighborsClassifier

base_path = r'D:\RadioData\20250709'
file_label_list = [
    ('2psk_frames.mat',   0),
    ('4ask_frames.mat',   1),
    ('8ask_frames.mat',   2),
    ('8psk_frames.mat',   3),
    ('16psk_frames.mat',  4),
    ('16qam_frames.mat',  5),
    ('32psk_frames.mat',  6),
    ('32qam_frames.mat',  7),
    ('64qam_frames.mat',  8),
    ('128qam_frames.mat', 9),
    ('bpsk_frames.mat',  10),
    ('ook_frames.mat',   11),
    ('qpsk_frames.mat',  12),
]

#### 加载训练集 ####
train_data_X = []
train_data_y = []

for filename, label in file_label_list:
    # 根据文件名自动生成子文件夹名字 e.g. '2psk_frames.mat' → '2psk'
    subfolder = filename.replace('_frames.mat', '')
    # 再拼上子文件夹和文件名
    mat_file = os.path.join(base_path, subfolder, filename)
    mat_data = loadmat(mat_file)
    frames = mat_data['frames']    # shape (L, 2, N)

    L, _, N = frames.shape
    for i in range(N):
        # 取第 i 帧，shape (L,2)
        fi = frames[:, 0, i]
        # 分离实部和虚部，并扁平化
        feat = np.hstack([fi.real.flatten(), fi.imag.flatten()])
        train_data_X.append(feat)
        train_data_y.append(label)

    # for i in range(len(mat_data['R_matrix'][0][0])):
    #     train_data_X.append([mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
    #                          mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
    #                          mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
    #                          mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag])
    #     train_data_y.append(label)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# # 创建并拟合模型
# svm_model = svm.SVC(kernel='rbf')
# svm_model.fit(X_train, y_train)

predictions = clf.predict(X_test)

# 模型评估
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}\n')
print(f'Confusion Matrix:\n{conf_matrix}\n')
print(f'Classification Report:\n{class_report}')


# 假设你已经有 y_test, predictions
cm = confusion_matrix(y_test, predictions)
labels = np.unique(y_train)  # 类别标签列表

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix of KNN Classifier')
plt.colorbar()
plt.xticks(np.arange(len(labels)), labels, rotation=45)
plt.yticks(np.arange(len(labels)), labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')

# 在每个格子里写上数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()


# 使用pickle保存定标器
with open('scaler_single.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# 使用pickle保存模型
with open('model_single.pkl', 'wb') as file:
    pickle.dump(clf, file)


#============================================================

# def data_predict(arr):
#     # 使用pickle加载定标器
#     scaler = pickle.load(open('D:/RadioData/scaler.pkl', 'rb'))
#
#     with open('D:/RadioData/model.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#
#     arr_ = scaler.transform([arr])
#      # 使用加载的模型进行预测
#     predictions = loaded_model.predict(arr_)
#     return predictions[0]
#
# test_data_X = []
# result = []
#
# cnt = [0,0,0,0]
#
# mat_file_path = r'D:\RadioData\8ask_dp.mat'
#
# # 加载.mat文件
# mat_data = loadmat(mat_file_path)
#
# for i in range(len(mat_data['R_matrix'][0][0])):
#     test_data_X.append([mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
#                         mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
#                         mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
#                         mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag])
#
# for i in range(len(test_data_X)):
#     print(data_predict(test_data_X[i]))
#     result.append(data_predict(test_data_X[i]))
#
# for i in range(len(result)):
#     if result[i] == 0:
#         cnt[0] += 1
#     elif result[i] == 1:
#         cnt[1] += 1
#     elif result[i] == 2:
#         cnt[2] += 1
#     elif result[i] == 3:
#         cnt[3] += 1
#
# print(cnt)