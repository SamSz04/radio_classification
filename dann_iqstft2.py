import sys
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from scipy.io import loadmat
from scipy.signal import stft, get_window
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ===== 改进的梯度反转层 =====
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


# ===== 改进的特征提取器 =====
class ImprovedIQFeatureExtractor(nn.Module):
    """改进的I/Q特征提取器，增加了针对QAM识别的特殊设计"""

    def __init__(self):
        super(ImprovedIQFeatureExtractor, self).__init__()

        # 初始I/Q特征提取 - 使用多尺度卷积
        self.iq_conv1 = nn.Conv2d(1, 32, kernel_size=(1, 2), stride=1, padding=0)
        self.iq_conv2 = nn.Conv2d(1, 32, kernel_size=(3, 2), stride=1, padding=(1, 0))
        self.iq_conv3 = nn.Conv2d(1, 32, kernel_size=(5, 2), stride=1, padding=(2, 0))

        # 特征融合
        self.fusion_conv = nn.Conv1d(96, 128, kernel_size=1)

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, downsample=True),
            ResidualBlock(512, 512),
        ])

        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 特征维度
        self.feature_dim = 1024  # 512 * 2 (avg + max)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch, 1, 512, 2)

        # 多尺度I/Q特征提取
        feat1 = self.iq_conv1(x)  # (batch, 32, 512, 1)
        feat2 = self.iq_conv2(x)  # (batch, 32, 512, 1)
        feat3 = self.iq_conv3(x)  # (batch, 32, 512, 1)

        # 拼接多尺度特征
        x = torch.cat([feat1, feat2, feat3], dim=1)  # (batch, 96, 512, 1)
        x = x.squeeze(-1)  # (batch, 96, 512)

        # 特征融合
        x = self.fusion_conv(x)  # (batch, 128, 512)

        # 残差块处理
        for block in self.residual_blocks:
            x = block(x)

        # 全局池化
        avg_feat = self.global_pool(x).squeeze(-1)  # (batch, 512)
        max_feat = self.max_pool(x).squeeze(-1)  # (batch, 512)

        # 组合特征
        x = torch.cat([avg_feat, max_feat], dim=1)  # (batch, 1024)

        return x


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ImprovedSTFTFeatureExtractor(nn.Module):
    """改进的STFT特征提取器"""

    def __init__(self):
        super(ImprovedSTFTFeatureExtractor, self).__init__()

        # 使用更深的网络结构
        self.features = nn.Sequential(
            # 初始卷积块
            ConvBlock(1, 64, pool=False),
            ConvBlock(64, 64),

            # 深层特征提取
            ConvBlock(64, 128),
            ConvBlock(128, 128, pool=False),

            ConvBlock(128, 256),
            ConvBlock(256, 256, pool=False),

            ConvBlock(256, 512),
            ConvBlock(512, 512, pool=False),

            # 全局池化
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.feature_dim = 512

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class ConvBlock(nn.Module):
    """卷积块"""

    def __init__(self, in_channels, out_channels, pool=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else None
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        x = self.dropout(x)
        return x


class ImprovedFusionFeatureExtractor(nn.Module):
    """改进的融合特征提取器"""

    def __init__(self, fusion_method='attention', use_stft=True):
        super(ImprovedFusionFeatureExtractor, self).__init__()

        self.use_stft = use_stft
        self.fusion_method = fusion_method

        # 特征提取器
        self.iq_extractor = ImprovedIQFeatureExtractor()
        if self.use_stft:
            self.stft_extractor = ImprovedSTFTFeatureExtractor()

        # 特征投影层（确保维度一致）
        self.iq_proj = nn.Linear(self.iq_extractor.feature_dim, 512)
        if self.use_stft:
            self.stft_proj = nn.Linear(self.stft_extractor.feature_dim, 512)

        # 特征融合
        if fusion_method == 'attention':
            self.attention = MultiHeadAttentionFusion(512, num_heads=8)
            self.feature_dim = 512
        elif fusion_method == 'concat':
            self.feature_dim = 1024  # 512 * 2
        else:
            self.feature_dim = 512

    def forward(self, iq_data, stft_data=None):
        # 提取I/Q特征
        iq_features = self.iq_extractor(iq_data)
        iq_features = self.iq_proj(iq_features)

        if not self.use_stft or stft_data is None:
            return iq_features

        # 提取STFT特征
        stft_features = self.stft_extractor(stft_data)
        stft_features = self.stft_proj(stft_features)

        # 特征融合
        if self.fusion_method == 'concat':
            fused_features = torch.cat([iq_features, stft_features], dim=1)
        elif self.fusion_method == 'add':
            fused_features = iq_features + stft_features
        elif self.fusion_method == 'attention':
            fused_features = self.attention(iq_features, stft_features)

        return fused_features


class MultiHeadAttentionFusion(nn.Module):
    """多头注意力特征融合"""

    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, feat1, feat2):
        batch_size = feat1.size(0)

        # 组合特征
        combined = torch.stack([feat1, feat2], dim=1)  # (batch, 2, d_model)

        # 多头注意力
        Q = self.W_q(combined).view(batch_size, 2, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(combined).view(batch_size, 2, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(combined).view(batch_size, 2, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, 2, self.d_model)
        output = self.W_o(context)

        # 残差连接和层归一化
        output = self.norm1(output + combined)

        # FFN
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)

        # 取平均作为最终输出
        return output.mean(dim=1)


# ===== 改进的分类器 =====
class HierarchicalClassifier(nn.Module):
    """层次化分类器 - 先分大类，再分小类"""

    def __init__(self, feature_dim=512, num_classes=12):
        super(HierarchicalClassifier, self).__init__()

        # 主分类器
        self.main_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # QAM专用分类器（用于区分不同阶数的QAM）
        self.qam_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)  # 16QAM, 32QAM, 64QAM, 128QAM
        )

        # 调制类型分组
        self.qam_indices = [8, 9, 10, 11]  # 16QAM, 32QAM, 64QAM, 128QAM的索引

    def forward(self, x):
        main_output = self.main_classifier(x)
        qam_output = self.qam_classifier(x)
        return main_output, qam_output


# ===== 改进的DANN模型 =====
class ImprovedFusionDANN(nn.Module):
    """改进的Fusion-DANN模型"""

    def __init__(self, num_classes=12, fusion_method='attention', use_stft=True):
        super(ImprovedFusionDANN, self).__init__()

        # 特征提取器
        self.feature_extractor = ImprovedFusionFeatureExtractor(fusion_method, use_stft)

        # 层次化分类器
        self.classifier = HierarchicalClassifier(
            self.feature_extractor.feature_dim,
            num_classes
        )

        # 域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        # 梯度反转层
        self.grl = GradientReversalLayer()

        self.use_stft = use_stft

    def forward(self, iq_data, stft_data=None, lambda_p=1.0):
        # 提取特征
        features = self.feature_extractor(iq_data, stft_data)

        # 分类
        class_output, qam_output = self.classifier(features)

        # 域判别
        self.grl.lambda_p = lambda_p
        reversed_features = self.grl(features)
        domain_output = self.domain_discriminator(reversed_features)

        return class_output, qam_output, domain_output, features


# ===== 改进的损失函数 =====
class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""

    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class CenterLoss(nn.Module):
    """Center Loss - 增强类内聚合"""

    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss


# ===== 改进的训练器 =====
class ImprovedFusionDANNTrainer:
    """改进的Fusion-DANN训练器"""

    def __init__(self, model, device, use_stft=True, num_classes=12):
        self.model = model.to(device)
        self.device = device
        self.use_stft = use_stft
        self.num_classes = num_classes

        # 优化器 - 使用不同的学习率
        self.optimizer = optim.Adam([
            {'params': model.feature_extractor.parameters(), 'lr': 0.0001},
            {'params': model.classifier.parameters(), 'lr': 0.001},
            {'params': model.domain_discriminator.parameters(), 'lr': 0.001}
        ])

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

        # 损失函数
        self.focal_loss = FocalLoss(gamma=2.0)
        self.center_loss = CenterLoss(num_classes, model.feature_extractor.feature_dim)
        self.center_loss = self.center_loss.to(device)
        self.domain_criterion = nn.BCEWithLogitsLoss()

        # Center loss优化器
        self.center_optimizer = optim.SGD(self.center_loss.parameters(), lr=0.5)

        # QAM indices
        self.qam_indices = torch.tensor([8, 9, 10, 11]).to(device)  # 16,32,64,128 QAM

        # 历史记录
        self.history = {
            'train_class_loss': [], 'train_domain_loss': [],
            'train_class_acc': [], 'train_domain_acc': [],
            'val_class_loss': [], 'val_class_acc': [],
            'target_domain_acc': []
        }

    def train_epoch(self, source_loader, target_loader, epoch, total_epochs):
        self.model.train()

        # 计算lambda_p
        p = float(epoch) / total_epochs
        lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

        # 统计量
        running_class_loss = 0.0
        running_domain_loss = 0.0
        running_center_loss = 0.0
        correct_class = 0
        correct_domain = 0
        total_class = 0
        total_domain = 0

        # 创建迭代器
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        n_batches = max(len(source_loader), len(target_loader))
        progress_bar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}")

        for batch_idx in progress_bar:
            # 获取数据
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_batch = next(source_iter)

            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            # 解包数据
            if self.use_stft:
                source_iq, source_stft, source_labels, source_domains = source_batch
                target_iq, target_stft, target_domains = target_batch

                source_iq = source_iq.to(self.device)
                source_stft = source_stft.to(self.device)
                target_iq = target_iq.to(self.device)
                target_stft = target_stft.to(self.device)
            else:
                source_iq, source_labels, source_domains = source_batch
                target_iq, target_domains = target_batch

                source_iq = source_iq.to(self.device)
                target_iq = target_iq.to(self.device)
                source_stft = None
                target_stft = None

            source_labels = source_labels.squeeze().to(self.device)
            source_domains = source_domains.to(self.device)
            target_domains = target_domains.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            self.center_optimizer.zero_grad()

            # 源域分类
            source_class_output, source_qam_output, source_domain_output, source_features = self.model(
                source_iq, source_stft, lambda_p
            )

            # 主分类损失（使用Focal Loss）
            class_loss = self.focal_loss(source_class_output, source_labels)

            # QAM辅助损失（只对QAM样本计算）
            qam_mask = torch.zeros_like(source_labels, dtype=torch.bool)
            for idx in self.qam_indices:
                qam_mask |= (source_labels == idx)

            if qam_mask.any():
                qam_labels = source_labels[qam_mask] - 8  # 转换为0-3
                qam_pred = source_qam_output[qam_mask]
                qam_loss = F.cross_entropy(qam_pred, qam_labels)
                class_loss = class_loss + 0.5 * qam_loss

            # Center Loss
            center_loss = self.center_loss(source_features, source_labels)

            # 域判别损失
            target_class_output, _, target_domain_output, _ = self.model(
                target_iq, target_stft, lambda_p
            )

            combined_domain_output = torch.cat([source_domain_output, target_domain_output])
            combined_domains = torch.cat([source_domains, target_domains])

            domain_loss = self.domain_criterion(
                combined_domain_output.squeeze(),
                combined_domains.squeeze()
            )

            # 总损失
            total_loss = class_loss + domain_loss + 0.01 * center_loss

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.center_optimizer.step()

            # 统计
            running_class_loss += class_loss.item()
            running_domain_loss += domain_loss.item()
            running_center_loss += center_loss.item()

            # 分类准确率
            _, predicted = source_class_output.max(1)
            correct_class += predicted.eq(source_labels).sum().item()
            total_class += source_labels.size(0)

            # 域判别准确率
            domain_pred = (torch.sigmoid(combined_domain_output) > 0.5).float()
            correct_domain += domain_pred.squeeze().eq(combined_domains.squeeze()).sum().item()
            total_domain += combined_domains.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'λ': f'{lambda_p:.3f}',
                'C_loss': f'{running_class_loss / (batch_idx + 1):.4f}',
                'D_loss': f'{running_domain_loss / (batch_idx + 1):.4f}',
                'Ctr_loss': f'{running_center_loss / (batch_idx + 1):.4f}',
                'C_acc': f'{100. * correct_class / total_class:.1f}%'
            })

        # 更新学习率
        self.scheduler.step()

        epoch_class_loss = running_class_loss / n_batches
        epoch_domain_loss = running_domain_loss / n_batches
        epoch_class_acc = 100. * correct_class / total_class
        epoch_domain_acc = 100. * correct_domain / total_domain

        return epoch_class_loss, epoch_domain_loss, epoch_class_acc, epoch_domain_acc, lambda_p

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                if self.use_stft:
                    iq_data, stft_data, labels, _ = batch
                    iq_data = iq_data.to(self.device)
                    stft_data = stft_data.to(self.device)
                else:
                    iq_data, labels, _ = batch
                    iq_data = iq_data.to(self.device)
                    stft_data = None

                labels = labels.squeeze().to(self.device)

                class_output, _, _, _ = self.model(iq_data, stft_data, lambda_p=0)
                loss = self.focal_loss(class_output, labels)

                running_loss += loss.item()
                _, predicted = class_output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def evaluate_target_domain(self, target_loader_with_labels):
        """评估目标域性能"""
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in target_loader_with_labels:
                if self.use_stft:
                    iq_data, stft_data, labels, _ = batch
                    iq_data = iq_data.to(self.device)
                    stft_data = stft_data.to(self.device)
                else:
                    iq_data, labels, _ = batch
                    iq_data = iq_data.to(self.device)
                    stft_data = None

                labels = labels.squeeze().to(self.device)

                class_output, qam_output, _, _ = self.model(iq_data, stft_data, lambda_p=0)
                _, predicted = class_output.max(1)

                # 对QAM类别使用QAM分类器的结果进行修正
                qam_mask = torch.zeros_like(predicted, dtype=torch.bool)
                for idx in self.qam_indices:
                    qam_mask |= (predicted == idx)

                if qam_mask.any():
                    _, qam_pred = qam_output[qam_mask].max(1)
                    predicted[qam_mask] = qam_pred + 8

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        target_acc = 100. * correct / total
        return target_acc, all_predictions, all_targets

    def train(self, source_loader, target_loader, val_loader, target_test_loader=None, epochs=50):
        """训练"""
        best_target_acc = 0

        for epoch in range(epochs):
            # 训练
            class_loss, domain_loss, class_acc, domain_acc, lambda_p = self.train_epoch(
                source_loader, target_loader, epoch, epochs
            )

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 记录历史
            self.history['train_class_loss'].append(class_loss)
            self.history['train_domain_loss'].append(domain_loss)
            self.history['train_class_acc'].append(class_acc)
            self.history['train_domain_acc'].append(domain_acc)
            self.history['val_class_loss'].append(val_loss)
            self.history['val_class_acc'].append(val_acc)

            # 评估目标域
            if target_test_loader is not None:
                target_acc, _, _ = self.evaluate_target_domain(target_test_loader)
                self.history['target_domain_acc'].append(target_acc)
                target_info = f", Target Acc: {target_acc:.2f}%"

                # 保存最佳模型
                if target_acc > best_target_acc:
                    best_target_acc = target_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'center_loss_state_dict': self.center_loss.state_dict(),
                        'best_target_acc': best_target_acc,
                    }, 'best_improved_fusion_dann_model.pth')
                    print(f"  --> New best model saved! (Target Acc: {target_acc:.2f}%)")
            else:
                target_info = ""

            # 打印进度
            print(f"\nEpoch {epoch + 1}/{epochs} (λ={lambda_p:.3f}):")
            print(f"  Train - Class Loss: {class_loss:.4f}, Class Acc: {class_acc:.2f}%, "
                  f"Domain Loss: {domain_loss:.4f}, Domain Acc: {domain_acc:.2f}%")
            print(f"  Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%{target_info}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def plot_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 分类损失
        ax = axes[0, 0]
        ax.plot(self.history['train_class_loss'], label='Train Class Loss')
        ax.plot(self.history['val_class_loss'], label='Val Class Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Classification Loss')
        ax.legend()
        ax.grid(True)

        # 域判别损失
        ax = axes[0, 1]
        ax.plot(self.history['train_domain_loss'], label='Domain Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Domain Discrimination Loss')
        ax.legend()
        ax.grid(True)

        # 分类准确率
        ax = axes[1, 0]
        ax.plot(self.history['train_class_acc'], label='Train Acc')
        ax.plot(self.history['val_class_acc'], label='Val Acc')
        if self.history['target_domain_acc']:
            ax.plot(self.history['target_domain_acc'], label='Target Domain Acc', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Classification Accuracy')
        ax.legend()
        ax.grid(True)

        # 域判别准确率
        ax = axes[1, 1]
        ax.plot(self.history['train_domain_acc'], label='Domain Acc')
        ax.axhline(y=50, color='r', linestyle='--', label='Random (50%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Domain Discrimination Accuracy')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f'./logs/{current_time}_training_history.png')
        plt.show()


# ===== 数据增强 =====
class SignalAugmentation:
    """信号增强"""

    @staticmethod
    def add_noise(signal, snr_db):
        """添加高斯白噪声"""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) +
                                            1j * np.random.randn(*signal.shape))
        return signal + noise

    @staticmethod
    def phase_shift(signal, phase):
        """相位偏移"""
        return signal * np.exp(1j * phase)

    @staticmethod
    def frequency_offset(signal, offset, fs=4e6):
        """频率偏移"""
        t = np.arange(len(signal)) / fs
        return signal * np.exp(2j * np.pi * offset * t)


# ===== 改进的数据集类 =====
class ImprovedFusionDANNDataset(Dataset):
    """改进的数据集类，增加数据增强"""

    def __init__(self, X, y=None, domain_label=0, compute_stft=True,
                 stft_params=None, augment=False, is_training=True):
        self.X = X
        self.y = y
        self.domain_label = domain_label
        self.compute_stft = compute_stft
        self.augment = augment and is_training

        # STFT参数
        if stft_params is None:
            self.stft_params = {
                'fs': 4e6,
                'window': 'hamming',
                'nperseg': 16,  # 增加频率分辨率
                'noverlap': 12,
                'nfft': 128
            }
        else:
            self.stft_params = stft_params

        # 预计算STFT
        if self.compute_stft:
            print(f"Computing STFT features for {'source' if domain_label == 0 else 'target'} domain...")
            self.stft_features = self._compute_all_stft()

    def _compute_stft(self, complex_signal):
        """计算STFT"""
        window = get_window('hamming', self.stft_params['nperseg'])

        f, t, Zxx = stft(
            complex_signal,
            fs=self.stft_params['fs'],
            window=window,
            nperseg=self.stft_params['nperseg'],
            noverlap=self.stft_params['noverlap'],
            nfft=self.stft_params['nfft'],
            return_onesided=False,
            boundary='zeros'
        )

        # 使用对数幅度谱
        stft_mag = np.abs(Zxx)
        stft_log = np.log1p(stft_mag)

        return stft_log

    def _compute_all_stft(self):
        """预计算所有STFT"""
        stft_list = []
        for i in tqdm(range(len(self.X)), desc="Computing STFT", leave=False):
            stft_feat = self._compute_stft(self.X[i])
            stft_list.append(stft_feat)
        return np.array(stft_list)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 获取复数信号
        complex_signal = self.X[idx].copy()

        # 数据增强
        if self.augment:
            # 随机添加噪声
            if np.random.rand() > 0.5:
                snr = np.random.uniform(10, 30)
                complex_signal = SignalAugmentation.add_noise(complex_signal, snr)

            # 随机相位偏移
            if np.random.rand() > 0.5:
                phase = np.random.uniform(-np.pi, np.pi)
                complex_signal = SignalAugmentation.phase_shift(complex_signal, phase)

            # 随机频率偏移
            if np.random.rand() > 0.5:
                offset = np.random.uniform(-1000, 1000)
                complex_signal = SignalAugmentation.frequency_offset(complex_signal, offset)

        # 提取I/Q特征
        I = np.real(complex_signal)
        Q = np.imag(complex_signal)

        # # 归一化
        # max_val = max(np.max(np.abs(I)), np.max(np.abs(Q)))
        # if max_val > 0:
        #     I = I / max_val
        #     Q = Q / max_val

        # 组合特征
        iq_features = np.stack([I, Q], axis=1)
        iq_features = torch.FloatTensor(iq_features)

        # 域标签
        domain = torch.FloatTensor([self.domain_label])

        # STFT特征
        if self.compute_stft:
            if self.augment:
                # 实时计算增强后的STFT
                stft_features = self._compute_stft(complex_signal)
                stft_features = torch.FloatTensor(stft_features).unsqueeze(0)
            else:
                stft_features = torch.FloatTensor(self.stft_features[idx]).unsqueeze(0)
        else:
            stft_features = None

        if self.y is not None:
            label = torch.LongTensor([self.y[idx]])
            if stft_features is not None:
                return iq_features, stft_features, label, domain
            else:
                return iq_features, label, domain
        else:
            if stft_features is not None:
                return iq_features, stft_features, domain
            else:
                return iq_features, domain


# ===== 主函数 =====
def load_source_target_data(source_path, target_path, test_case_path=None):
    """加载源域和目标域数据"""
    modulation_types = [
        '4ask', '8ask', '8psk', '16psk', '16qam',
        '32psk', '32qam', 'bpsk', 'ook', 'qpsk'
    ]

    label_map = {mod: i for i, mod in enumerate(modulation_types)}

    def load_dataset(path, dataset_name):
        all_data = []
        all_labels = []

        print(f"\nLoading {dataset_name} from {path}")

        for mod_type in modulation_types:
            file_path = os.path.join(path, f'{mod_type}_seg512.mat')

            if os.path.exists(file_path):
                mat_data = loadmat(file_path)
                frames = mat_data['frames']  # (512, 2, N)
                complex_data = frames[:, 0, :].T  # (N, 512)

                n_samples = complex_data.shape[0]
                labels = np.full(n_samples, label_map[mod_type])

                all_data.append(complex_data)
                all_labels.append(labels)

                print(f"  {mod_type}: {n_samples} samples")

        X = np.vstack(all_data)
        y = np.hstack(all_labels)

        print(f"Total {dataset_name}: {X.shape}")

        return X, y

    # 加载数据
    X_source, y_source = load_dataset(source_path, "source domain")
    X_target_unlabeled, _ = load_dataset(target_path, "target domain (unlabeled)")

    # 如果有测试集路径，加载带标签的目标域测试集
    if test_case_path:
        X_target_test, y_target_test = load_dataset(test_case_path, "target domain test")
        return X_source, y_source, X_target_unlabeled, X_target_test, y_target_test

    return X_source, y_source, X_target_unlabeled, None, None


def main():
    # 路径设置
    source_path = r'D:\RadioData\dataset1'  # 源域（训练数据）
    target_path = r'D:\RadioData\testcase0714'  # 目标域（无标签）
    test_case_path = r'D:\RadioData\testcase0714'  # 目标域测试集（有标签）

    # 加载数据
    X_source, y_source, X_target_unlabeled, X_target_test, y_target_test = load_source_target_data(
        source_path, target_path, test_case_path
    )

    # 划分源域数据
    X_train, X_val, y_train, y_val = train_test_split(
        X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
    )

    print(f"\nSource domain split:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"Target domain:")
    print(f"  Unlabeled: {X_target_unlabeled.shape}")
    if X_target_test is not None:
        print(f"  Test: {X_target_test.shape}")

    # 创建数据集
    train_dataset = ImprovedFusionDANNDataset(
        X_train, y_train, domain_label=0, augment=True, is_training=True
    )
    val_dataset = ImprovedFusionDANNDataset(
        X_val, y_val, domain_label=0, augment=False, is_training=False
    )
    target_dataset = ImprovedFusionDANNDataset(
        X_target_unlabeled, domain_label=1, augment=False, is_training=False
    )

    # 数据加载器
    batch_size = 32  # 减小batch size以获得更好的梯度估计
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    target_loader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # 如果有目标域测试集
    target_test_loader = None
    if X_target_test is not None:
        target_test_dataset = ImprovedFusionDANNDataset(
            X_target_test, y_target_test, domain_label=1, augment=False, is_training=False
        )
        target_test_loader = DataLoader(
            target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

    # 创建模型和训练器
    model = ImprovedFusionDANN(num_classes=12, fusion_method='attention', use_stft=True)
    trainer = ImprovedFusionDANNTrainer(model, device, use_stft=True, num_classes=12)

    # 打印模型结构
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 训练
    print("\nTraining Improved Fusion-DANN...")
    trainer.train(
        train_loader,
        target_loader,
        val_loader,
        target_test_loader,
        epochs=50  # 增加训练轮数
    )

    # 绘制训练历史
    trainer.plot_history()

    # 评估最终性能
    if target_test_loader is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TARGET DOMAIN TEST SET")
        print("=" * 60)

        # 加载最佳模型
        checkpoint = torch.load('best_improved_fusion_dann_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.center_loss.load_state_dict(checkpoint['center_loss_state_dict'])
        model.eval()

        # 评估
        final_acc, all_predictions, all_targets = trainer.evaluate_target_domain(target_test_loader)

        print(f"\nTarget Domain Test Accuracy: {final_acc:.2f}%")

        # 混淆矩阵
        modulation_names = ['ook', '4ask', '8ask', 'bpsk', 'qpsk', '8psk',
                            '16psk', '32psk', '16qam', '32qam', '64qam', '128qam']

        cm = confusion_matrix(all_targets, all_predictions)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=modulation_names, yticklabels=modulation_names)
        plt.title(f'Confusion Matrix - Improved Model (Accuracy: {final_acc:.2f}%)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'./logs/{current_time}_improved_confusion_matrix.png')
        plt.show()

        # 分类报告
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions,
                                    target_names=modulation_names, digits=4))

        # 单独分析QAM性能
        qam_indices = [8, 9, 10, 11]  # 16QAM, 32QAM, 64QAM, 128QAM
        qam_mask = np.isin(all_targets, qam_indices)
        if qam_mask.any():
            qam_targets = all_targets[qam_mask]
            qam_predictions = all_predictions[qam_mask]
            qam_accuracy = np.mean(qam_targets == qam_predictions)
            print(f"\nQAM-only Accuracy: {qam_accuracy:.4f} ({qam_accuracy * 100:.2f}%)")

            # QAM混淆矩阵
            qam_names = ['16QAM', '32QAM', '64QAM', '128QAM']
            qam_cm = confusion_matrix(qam_targets - 8, qam_predictions - 8)

            plt.figure(figsize=(8, 6))
            sns.heatmap(qam_cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=qam_names, yticklabels=qam_names)
            plt.title(f'QAM Confusion Matrix (Accuracy: {qam_accuracy:.4f})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(f'./logs/{current_time}_qam_confusion_matrix.png')
            plt.show()


if __name__ == "__main__":
    main()