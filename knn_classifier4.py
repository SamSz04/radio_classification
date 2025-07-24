import os
import pickle
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 1) 调制名称到标签的映射
mod_list = ['4ask','8ask','8psk','16psk',
            '16qam','32psk','32qam','64qam',
            '128qam','bpsk','ook','qpsk']
mod2label = {m:i for i,m in enumerate(mod_list)}

# 2) 目录根
base_path = r'D:\RadioData\20250711'

distances = ['1m','3m','5m']
# distances = ['1m']

train_X, train_y = [], []

# 1) 数据加载+特征提取阶段
for dist in tqdm(distances, desc='距离'):              # 外层距离
    dist_dir = os.path.join(base_path, dist)
    for mod in tqdm(mod_list, desc=f'调制 ({dist})'):  # 每种调制
        mod_dir = os.path.join(dist_dir, mod)
        if not os.path.isdir(mod_dir):
            continue
        mats = [f for f in os.listdir(mod_dir) if f.endswith('_frames.mat')]
        for fn in tqdm(mats, desc=f'{mod} 文件'):     # 每个 _frames.mat
            mat_path = os.path.join(mod_dir, fn)
            data = loadmat(mat_path)
            frames = data['frames']
            L, _, N = frames.shape
            # 只用通道 0
            for i in range(N):
                fi = frames[:,0,i]        # shape (L,)
                feat = np.hstack([fi.real, fi.imag])
                train_X.append(feat)
                train_y.append(mod2label[mod])

# 3) 划分、标准化、训练、评估
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.3, random_state=42)
print(f'训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}')

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# clf = KNeighborsClassifier()
# clf.fit(X_train, y_train)

# 创建并拟合模型
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

# 模型评估
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}\n')
print(f'Confusion Matrix:\n{conf_matrix}\n')
print(f'Classification Report:\n{class_report}')

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

# 保存
with open('scaler_svm.pkl','wb') as f: pickle.dump(scaler,f)
with open('model_svm.pkl','wb') as f: pickle.dump(clf,f)
