import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CNN1D(nn.Module):
    """1D CNN模型用于调制识别 - 需要与训练时的模型结构完全一致"""
    def __init__(self, input_length=64, num_classes=10):
        super(CNN1D, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 调整输入维度：从 (batch, length, channels) 到 (batch, channels, length)
        x = x.permute(0, 2, 1)

        # 通过卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 通过全连接层
        x = self.fc(x)

        return x


class TestDataset(Dataset):
    """测试数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_test_dataset(test_path):
    """加载测试数据集"""
    # 调制类型列表
    # modulation_types = [
    #     '4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
    #     '32qam', '64qam', '128qam', 'bpsk', 'ook', 'qpsk'
    # ]
    #
    # # 标签映射
    # label_map = {
    #     '4ask': 0, '8ask': 1, '8psk': 2, '16psk': 3,
    #     '16qam': 4, '32psk': 5, '32qam': 6, '64qam': 7,
    #     '128qam': 8, 'bpsk': 9, 'ook': 10, 'qpsk': 11
    # }

    modulation_types = [
        '4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
        '32qam', 'bpsk', 'ook', 'qpsk'
    ]

    # 标签映射
    label_map = {
        '4ask': 0, '8ask': 1, '8psk': 2, '16psk': 3,
        '16qam': 4, '32psk': 5, '32qam': 6,
        'bpsk': 7, 'ook': 8, 'qpsk': 9
    }

    all_data = []
    all_labels = []
    file_info = []  # 记录每个样本来自哪个文件

    print("Loading test dataset...")
    print(f"Test path: {test_path}")

    for mod_type in modulation_types:
        file_path = os.path.join(test_path, f'{mod_type}_seg64.mat')

        if os.path.exists(file_path):
            try:
                # 加载数据
                mat_data = loadmat(file_path)

                # 获取数据
                frames = mat_data['frames']  # (512, 2, N)

                if frames.shape[1] != 2:
                    raise ValueError(f"Expected 2 antennas, got shape {frames.shape}")

                # 选择天线0的数据（复数），维度：(512, N)
                complex_data = frames[:, 0, :]  # 取天线0

                # 拆分为实部和虚部 → 得到 (N, 512, 2)
                X_real = np.real(complex_data).T  # (N, 512)
                X_imag = np.imag(complex_data).T  # (N, 512)
                X = np.stack((X_real, X_imag), axis=2)  # (N, 512, 2)

                labels = np.full(X.shape[0], label_map[mod_type])

                all_data.append(X)
                all_labels.append(labels)

                n_samples = X.shape[0]
                file_info.extend([mod_type] * n_samples)

                print(f"    Loaded {n_samples} samples")

            except Exception as e:
                print(f"  Error loading {file_path}: {str(e)}")
                continue
        else:
            print(f"  Warning: {file_path} not found")

    if len(all_data) == 0:
        raise ValueError("No data loaded! Check your file paths and data format.")

    # 合并所有数据
    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    print(f"\nTotal test dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes in test set: {len(np.unique(y))}")
    print(f"Samples per class:")
    for mod_type, label in label_map.items():
        count = np.sum(y == label)
        if count > 0:
            print(f"  {mod_type}: {count}")

    return X, y, modulation_types, file_info


def test_model(model_path, test_path, batch_size=64):
    """测试保存的模型"""
    # 1. 加载模型
    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # 获取模型参数
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        input_length = checkpoint.get('input_length', 64)
        num_classes = checkpoint.get('num_classes', 12)
        modulation_names_trained = checkpoint.get('modulation_names', None)
    else:
        # 如果是旧格式，可能直接保存的是state_dict
        model_state = checkpoint
        input_length = 64
        num_classes = 12
        modulation_names_trained = None

    # 创建模型
    model = CNN1D(input_length=input_length, num_classes=num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Input length: {input_length}")
    print(f"Number of classes: {num_classes}")

    # 2. 加载测试数据
    X_test, y_test, modulation_names, file_info = load_test_dataset(test_path)

    # 3. 创建数据加载器
    test_dataset = TestDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 进行预测
    print("\nRunning predictions...")
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)

            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} samples...")

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # 6. 计算性能指标
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    # 整体准确率
    overall_accuracy = accuracy_score(y_test, all_predictions)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

    # 每个类别的准确率
    print("\nPer-class Accuracy:")
    for i, mod_type in enumerate(modulation_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], all_predictions[mask])
            print(f"  {mod_type:8s}: {class_acc:.4f} ({class_acc * 100:.2f}%) - {np.sum(mask)} samples")

    # 分类报告
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, all_predictions,
                                target_names=modulation_names,
                                digits=4))

    # 7. 混淆矩阵
    cm = confusion_matrix(y_test, all_predictions)

    # 绘制混淆矩阵
    plt.figure(figsize=(14, 12))

    # 计算百分比混淆矩阵
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建标注文本
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)'

    sns.heatmap(cm_percentage, annot=annot_text, fmt='', cmap='Blues',
                xticklabels=modulation_names, yticklabels=modulation_names,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Confusion Matrix (Overall Accuracy: {overall_accuracy:.4f})', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 8. 分析错误分类
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # 找出最容易混淆的类别对
    print("\nMost Confused Pairs:")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # 移除对角线

    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_copy[i, j] > 0:
                confusion_pairs.append((i, j, cm_copy[i, j]))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    for i, (true_idx, pred_idx, count) in enumerate(confusion_pairs[:10]):
        true_class = modulation_names[true_idx]
        pred_class = modulation_names[pred_idx]
        percentage = count / np.sum(y_test == true_idx) * 100
        print(f"  {i + 1}. {true_class} → {pred_class}: {count} samples ({percentage:.1f}%)")

    # 9. 置信度分析
    print("\nConfidence Analysis:")

    # 计算每个预测的置信度
    max_probs = np.max(all_probabilities, axis=1)

    # 正确预测和错误预测的置信度
    correct_mask = all_predictions == y_test
    correct_confidences = max_probs[correct_mask]
    incorrect_confidences = max_probs[~correct_mask]

    print(f"  Average confidence (correct predictions): {np.mean(correct_confidences):.4f}")
    print(f"  Average confidence (incorrect predictions): {np.mean(incorrect_confidences):.4f}")

    # 绘制置信度分布
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 51)
    plt.hist(correct_confidences, bins=bins, alpha=0.5, label='Correct', density=True)
    plt.hist(incorrect_confidences, bins=bins, alpha=0.5, label='Incorrect', density=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Distribution for Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 10. 保存结果
    results = {
        'overall_accuracy': overall_accuracy,
        'predictions': all_predictions,
        'true_labels': y_test,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'modulation_names': modulation_names,
        'file_info': file_info
    }

    # 保存结果
    # save_path = 'test_results.npz'
    # np.savez(save_path, **results)
    # print(f"\nTest results saved to: {save_path}")

    return results


def main():
    # 设置路径
    model_path = 'modulation_classifier_complete.pth'  # 或 'best_modulation_model.pth'
    test_path = r'D:\RadioData\testcase64'

    # 运行测试
    results = test_model(model_path, test_path)

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print(f"Overall Test Accuracy: {results['overall_accuracy']:.4f}")


if __name__ == "__main__":
    main()