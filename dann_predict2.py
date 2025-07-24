import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from scipy.io import loadmat
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ===== 复制DANN模型定义 =====
class GradientReversalFunction(Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_p
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_p=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_p = lambda_p

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_p)


class FeatureExtractor(nn.Module):
    """更简单的特征提取器 - CNN + 单个RNN"""
    def __init__(self, input_channels=2, cnn_feature_dim=512, rnn_feature_dim=256):
        super(FeatureExtractor, self).__init__()

        # CNN分支（同上）
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 只用GRU（通常比LSTM快）
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=rnn_feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.feature_dim = cnn_feature_dim + rnn_feature_dim

    def forward(self, x):
        batch_size = x.size(0)

        # CNN特征
        cnn_features = self.conv_blocks(x.permute(0, 2, 1))
        cnn_features = cnn_features.view(batch_size, -1)

        # GRU特征 - 使用最后的隐藏状态
        _, gru_h = self.gru(x)
        rnn_features = gru_h[-1]  # 取最后一层的输出

        # 拼接
        return torch.cat([cnn_features, rnn_features], dim=1)


class LabelClassifier(nn.Module):
    """标签分类器"""

    def __init__(self, feature_dim=768, num_classes=12):
        super(LabelClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class DomainDiscriminator(nn.Module):
    """域判别器"""

    def __init__(self, feature_dim=768):
        super(DomainDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 1)  # 二分类：源域(0)或目标域(1)
        )

    def forward(self, x):
        return self.discriminator(x)


class DANN(nn.Module):
    """完整的DANN模型（CNN + LSTM-GRU）"""

    def __init__(self, input_channels=2, num_classes=12):
        super(DANN, self).__init__()

        # 特征提取器（CNN + LSTM-GRU）
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            cnn_feature_dim=512,
            rnn_feature_dim=256
        )

        # 标签分类器和域判别器使用组合后的特征维度
        total_feature_dim = self.feature_extractor.feature_dim
        self.label_classifier = LabelClassifier(total_feature_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(total_feature_dim)
        self.grl = GradientReversalLayer()

    def forward(self, x, lambda_p=1.0):
        # 提取特征（CNN + LSTM-GRU）
        features = self.feature_extractor(x)

        # 标签分类
        class_output = self.label_classifier(features)

        # 域判别（通过梯度反转层）
        self.grl.lambda_p = lambda_p
        reversed_features = self.grl(features)
        domain_output = self.domain_discriminator(reversed_features)

        return class_output, domain_output, features


class TestDataset(Dataset):
    """测试数据集类"""

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 获取复数信号
        complex_signal = self.X[idx]

        # 提取I/Q特征
        I = np.real(complex_signal)
        Q = np.imag(complex_signal)

        # # 归一化
        # signal_power = np.mean(I ** 2 + Q ** 2)
        # if signal_power > 0:
        #     I = I / np.sqrt(signal_power)
        #     Q = Q / np.sqrt(signal_power)

        # 组合特征
        features = np.stack([I, Q], axis=1)  # (512, 2)
        features = torch.FloatTensor(features)

        if self.y is not None:
            label = torch.LongTensor([self.y[idx]])
            return features, label
        else:
            return features


def load_test_data(test_path):
    """加载测试数据"""
    # modulation_types = [
    #     '4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
    #     '32qam', 'bpsk', 'ook', 'qpsk'
    # ]

    modulation_types = [
        '4ask', '8ask', '8psk', '16psk', '16qam', '32psk',
        '32qam', '64qam', '128qam', 'bpsk', 'ook', 'qpsk'
    ]

    label_map = {mod: i for i, mod in enumerate(modulation_types)}

    all_data = []
    all_labels = []
    file_info = []

    print(f"Loading test data from: {test_path}")

    for mod_type in modulation_types:
        file_path = os.path.join(test_path, f'{mod_type}_seg512.mat')

        if os.path.exists(file_path):
            try:
                mat_data = loadmat(file_path)

                # 尝试不同的变量名
                if 'frames' in mat_data:
                    frames = mat_data['frames']  # (512, 2, N)
                    complex_data = frames[:, 0, :].T  # 天线0的数据 (N, 512)
                else:
                    # 尝试其他可能的变量名
                    possible_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    if possible_keys:
                        data = mat_data[possible_keys[0]]
                        # 假设数据格式是 (512, 2, N) 或类似
                        if data.shape[0] == 512:
                            complex_data = data[:, 0, :].T
                        else:
                            print(f"  Warning: Unexpected data shape in {mod_type}: {data.shape}")
                            continue

                n_samples = complex_data.shape[0]
                labels = np.full(n_samples, label_map[mod_type])

                all_data.append(complex_data)
                all_labels.append(labels)
                file_info.extend([mod_type] * n_samples)

                print(f"  {mod_type}: {n_samples} samples loaded")

            except Exception as e:
                print(f"  Error loading {mod_type}: {str(e)}")
                continue
        else:
            print(f"  File not found: {file_path}")

    if len(all_data) == 0:
        raise ValueError("No data loaded!")

    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    print(f"\nTotal test samples: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # 打印每个类别的样本数
    print("\nSamples per class:")
    for i, mod_type in enumerate(modulation_types):
        count = np.sum(y == i)
        if count > 0:
            print(f"  {mod_type}: {count}")

    return X, y, modulation_types, file_info


def test_dann_model(model_path, test_path, batch_size=64):
    """测试DANN模型"""

    # 1. 加载模型
    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 创建模型
    model = DANN(input_channels=2, num_classes=12).to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")

    # 2. 加载测试数据
    X_test, y_test, modulation_names, file_info = load_test_data(test_path)

    # 3. 创建数据加载器
    test_dataset = TestDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. 进行预测
    print("\nRunning predictions...")
    all_predictions = []
    all_probabilities = []
    all_features = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if len(batch) == 2:
                data, labels = batch
                labels = labels.squeeze()
                all_targets.extend(labels.numpy())
            else:
                data = batch

            data = data.to(device)

            # 前向传播（不需要域判别）
            class_output, _, features = model(data, lambda_p=0)

            # 获取预测
            probabilities = torch.softmax(class_output, dim=1)
            predictions = class_output.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_features.extend(features.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_features = np.array(all_features)

    if all_targets:
        all_targets = np.array(all_targets)
    else:
        # 如果没有标签，使用文件信息推断
        all_targets = y_test

    # 5. 计算性能指标
    print("\n" + "=" * 60)
    print(f"TEST RESULTS ON {os.path.basename(test_path)}")
    print("=" * 60)

    # 整体准确率
    overall_accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

    # 每个类别的准确率
    print("\nPer-class Performance:")
    for i, mod_type in enumerate(modulation_names):
        mask = all_targets == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(all_targets[mask], all_predictions[mask])
            class_samples = np.sum(mask)
            print(f"  {mod_type:8s}: {class_acc:.4f} ({class_acc * 100:.2f}%) - {class_samples} samples")

    # 分类报告
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions,
                                target_names=modulation_names,
                                digits=4, zero_division=0))

    # 6. 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)

    # 绘制混淆矩阵
    plt.figure(figsize=(14, 12))

    # 计算百分比
    cm_percentage = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_percentage[i] = cm[i] / row_sum * 100

    # 创建标注文本
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i].sum() > 0:  # 避免除零
                annot_text[i, j] = f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)'
            else:
                annot_text[i, j] = '0\n(0.0%)'

    sns.heatmap(cm_percentage, annot=annot_text, fmt='', cmap='Blues',
                xticklabels=modulation_names, yticklabels=modulation_names,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Confusion Matrix - {os.path.basename(test_path)} (Accuracy: {overall_accuracy:.4f})', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 7. 错误分析
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # 找出最容易混淆的类别对
    print("\nMost Confused Pairs:")
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)

    confusion_pairs = []
    for i in range(12):
        for j in range(12):
            if i != j and cm_copy[i, j] > 0:
                confusion_pairs.append((i, j, cm_copy[i, j]))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    for idx, (true_idx, pred_idx, count) in enumerate(confusion_pairs[:10]):
        if true_idx < len(modulation_names) and pred_idx < len(modulation_names):
            true_class = modulation_names[true_idx]
            pred_class = modulation_names[pred_idx]
            total_true = np.sum(all_targets == true_idx)
            if total_true > 0:
                percentage = count / total_true * 100
                print(f"  {idx + 1}. {true_class} → {pred_class}: {count} samples ({percentage:.1f}%)")

    # 8. 置信度分析
    print("\nConfidence Analysis:")

    # 计算每个预测的置信度
    max_probs = np.max(all_probabilities, axis=1)

    # 正确预测和错误预测的置信度
    correct_mask = all_predictions == all_targets
    correct_confidences = max_probs[correct_mask]
    incorrect_confidences = max_probs[~correct_mask]

    print(f"  Average confidence (correct predictions): {np.mean(correct_confidences):.4f}")
    if len(incorrect_confidences) > 0:
        print(f"  Average confidence (incorrect predictions): {np.mean(incorrect_confidences):.4f}")

    # 绘制置信度分布
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 51)
    plt.hist(correct_confidences, bins=bins, alpha=0.5, label='Correct', density=True)
    if len(incorrect_confidences) > 0:
        plt.hist(incorrect_confidences, bins=bins, alpha=0.5, label='Incorrect', density=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title(f'Confidence Distribution - {os.path.basename(test_path)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 9. 特征可视化（t-SNE）
    print("\nVisualizing learned features...")
    from sklearn.manifold import TSNE

    # 使用部分数据进行t-SNE（避免计算时间过长）
    n_samples_viz = min(2000, len(all_features))
    indices = np.random.choice(len(all_features), n_samples_viz, replace=False)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(all_features[indices])

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=all_targets[indices], cmap='tab10',
                          alpha=0.6, s=10)

    # 添加图例
    handles = []
    for i, mod_type in enumerate(modulation_names):
        if i in all_targets[indices]:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=plt.cm.tab12(i / 12),
                                      markersize=10, label=mod_type))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(f't-SNE Visualization of Learned Features - {os.path.basename(test_path)}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()

    # 10. 保存结果
    results = {
        'test_path': test_path,
        'overall_accuracy': overall_accuracy,
        'predictions': all_predictions,
        'true_labels': all_targets,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'modulation_names': modulation_names,
        'features': all_features
    }

    # 保存结果
    save_path = f'test_results_{os.path.basename(test_path)}.npz'
    np.savez(save_path, **results)
    print(f"\nTest results saved to: {save_path}")

    return results


def main():
    # 设置路径
    model_path = 'best_dann_cnnlstm_model.pth'  # 训练好的DANN模型
    test_path = r'D:\RadioData\testcase0714'  # 要测试的数据集

    # 运行测试
    results = test_dann_model(model_path, test_path, batch_size=64)

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print(f"Dataset: {os.path.basename(test_path)}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy'] * 100:.2f}%)")

    # 如果想测试其他数据集，可以继续添加
    # test_path2 = r'D:\RadioData\testcase0715'
    # results2 = test_dann_model(model_path, test_path2)


if __name__ == "__main__":
    main()