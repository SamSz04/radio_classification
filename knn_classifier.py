import os
import numpy as np
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# 1. 超参数
data_dir = r"D:\RadioData\20250630\dataset"
modulations = ['2psk', '8ask', '64qam', 'qpsk']
num_classes = len(modulations)
test_size = 0.2
random_state = 42


# 2. 数据加载函数
def load_data(data_dir, modulations):
    """加载并准备数据"""
    X_all, y_all = [], []

    for label, m in enumerate(modulations):
        for suffix in ['data1_dp.mat', 'data2_dp.mat']:
            path = os.path.join(data_dir, f"{m}_{suffix}")
            mat = scipy.io.loadmat(path)
            R = mat['R_matrix']  # shape (2,2,N)
            N = R.shape[2]

            # 展平成 N×8
            feats = np.zeros((N, 8))
            for i in range(N):
                r = R[:, :, i]
                feats[i] = [
                    r[0, 0].real, r[0, 0].imag,
                    r[0, 1].real, r[0, 1].imag,
                    r[1, 0].real, r[1, 0].imag,
                    r[1, 1].real, r[1, 1].imag,
                ]
            X_all.append(feats)
            y_all.append(np.full(N, label))

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    return X, y


# 3. 加载数据
print("加载数据...")
X, y = load_data(data_dir, modulations)
print(f"数据形状: X={X.shape}, y={y.shape}")

# 4. 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 5. 特征标准化（对KNN很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. KNN模型训练与超参数搜索
print("\n开始KNN训练...")

# 方法1：简单KNN
knn_simple = KNeighborsClassifier(n_neighbors=5)
knn_simple.fit(X_train_scaled, y_train)
y_pred_simple = knn_simple.predict(X_test_scaled)
acc_simple = accuracy_score(y_test, y_pred_simple)
print(f"简单KNN (k=5) 准确率: {acc_simple:.4f}")

# 7. 保存模型和标准化器
model_save_path = os.path.join(data_dir, "knn_model.pkl")
scaler_save_path = os.path.join(data_dir, "scaler.pkl")

# 保存训练好的KNN模型
with open(model_save_path, 'wb') as f:
    pickle.dump(knn_simple, f)
print(f"\nKNN模型已保存到: {model_save_path}")

# 保存标准化器
with open(scaler_save_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"标准化器已保存到: {scaler_save_path}")