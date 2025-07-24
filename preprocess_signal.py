import os
import numpy as np
from scipy.io import savemat

from cnn_classifier2 import load_dataset, SignalPreprocessor  # 替换为实际模块路径

dataset_path = r'D:\RadioData\dataset1'
save_path = r'D:\RadioData\preprocessed_constellation'

X, y, modulation_names = load_dataset(dataset_path)
os.makedirs(save_path, exist_ok=True)

pre = SignalPreprocessor()

for i, mod in enumerate(modulation_names):
    idx = np.where(y == i)[0][:10]  # 每类取前10个样本
    samples = X[idx]  # shape: (10, 512)

    constellations = []
    for sig in samples:
        processed = pre.preprocess_signal(sig)  # 输出为复数 array
        constellations.append(processed)

    constellations = np.array(constellations)  # shape: (10, 512)
    savemat(os.path.join(save_path, f"{mod}_processed.mat"), {
        'mod_type': mod,
        'signals': constellations  # (10, 512) 复数矩阵
    })
