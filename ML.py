import pickle
from sklearn import svm
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 导入KNN 分类器
from sklearn.neighbors import KNeighborsClassifier

#### 加载训练集 ####
train_data_X = []
train_data_y = []

# 加载N类型数据

# 替换为您的.mat文件路径
for j in range(1):

    mat_file_path = r'D:\RadioData\2psk_dp.mat'

# 加载.mat文件
    mat_data = loadmat(mat_file_path)
# mat_data是一个字典，其中包含.mat文件中的变量

# print(mat_data['R_matrix'])
# print(len(mat_data['R_matrix'][0][0]))

    for i in range(len(mat_data['R_matrix'][0][0])):
        train_data_X.append([mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
                            mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
                            mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
                            mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag])
        train_data_y.append(0)

# 加载D类型数据
for j in range(1):

    mat_file_path = r'D:\RadioData\8ask_dp.mat'

    # 加载.mat文件
    mat_data = loadmat(mat_file_path)
    # mat_data是一个字典，其中包含.mat文件中的变量

    # print(mat_data['R_matrix'])
    # print(len(mat_data['R_matrix'][0][0]))

    for i in range(len(mat_data['R_matrix'][0][0])):
        train_data_X.append([mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
                            mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
                            mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
                            mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag])
        train_data_y.append(1)

# 加载S类型数据
for j in range(1):

    mat_file_path = r'D:\RadioData\64qam_dp.mat'

    # 加载.mat文件
    mat_data = loadmat(mat_file_path)
    # mat_data是一个字典，其中包含.mat文件中的变量

    # print(mat_data['R_matrix'])
    # print(len(mat_data['R_matrix'][0][0]))

    for i in range(len(mat_data['R_matrix'][0][0])):
        train_data_X.append([mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
                            mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
                            mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
                            mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag])
        train_data_y.append(2)

# 加载M类型数据
for j in range(1):

    mat_file_path = r'D:\RadioData\qpsk_dp.mat'

    # 加载.mat文件
    mat_data = loadmat(mat_file_path)
    # mat_data是一个字典，其中包含.mat文件中的变量

    # print(mat_data['R_matrix'])
    # print(len(mat_data['R_matrix'][0][0]))

    for i in range(len(mat_data['R_matrix'][0][0])):
        train_data_X.append([mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
                            mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
                            mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
                            mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag])
        train_data_y.append(3)

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

# #### 加载训练集 ####
# test_data_X = []
# test_data_y = []
#
# # 加载N类型数据
#
# # 替换为您的.mat文件路径
# for j in range(6):
#
#     mat_file_path = 'D:\\data\\2024-7-27\\N\\data1.mat'
#
#     # 加载.mat文件
#     mat_data = loadmat(mat_file_path)
#     # mat_data是一个字典，其中包含.mat文件中的变量
#
#     # print(mat_data['R_matrix'])
#     # print(len(mat_data['R_matrix'][0][0]))
#
#     for i in range(len(mat_data['R_matrix'][0][0])):
#         test_data_X.append([mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
#                             mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
#                             mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
#                             mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag])
#         test_data_y.append(0)
#
# X_test_ = scaler.transform(test_data_X)
#
# # 使用加载的模型进行预测
# predictions = svm_model.predict(X_test_)
#
# print(predictions)
# print(len(predictions))
# count = 0
# for i in range(len(predictions)):
#     if(predictions[i] == 1):
#         count += 1
#
# print(count)

# 使用pickle保存定标器
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# 使用pickle保存模型
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file)

import pickle
import sklearn

def data_predict(arr):
    # 使用pickle加载定标器
    scaler = pickle.load(open('D:/RadioData/scaler.pkl', 'rb'))

    with open('D:/RadioData/model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    arr_ = scaler.transform([arr])
     # 使用加载的模型进行预测
    predictions = loaded_model.predict(arr_)
    return predictions[0]

test_data_X = []
result = []

cnt = [0,0,0,0]

mat_file_path = r'D:\RadioData\8ask_dp.mat'

# 加载.mat文件
mat_data = loadmat(mat_file_path)

for i in range(len(mat_data['R_matrix'][0][0])):
    test_data_X.append([mat_data['R_matrix'][0][0][i].real, mat_data['R_matrix'][0][0][i].imag,
                        mat_data['R_matrix'][0][1][i].real, mat_data['R_matrix'][0][1][i].imag,
                        mat_data['R_matrix'][1][0][i].real, mat_data['R_matrix'][1][0][i].imag,
                        mat_data['R_matrix'][1][1][i].real, mat_data['R_matrix'][1][1][i].imag])

for i in range(len(test_data_X)):
    print(data_predict(test_data_X[i]))
    result.append(data_predict(test_data_X[i]))

for i in range(len(result)):
    if result[i] == 0:
        cnt[0] += 1
    elif result[i] == 1:
        cnt[1] += 1
    elif result[i] == 2:
        cnt[2] += 1
    elif result[i] == 3:
        cnt[3] += 1

print(cnt)