import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedLSTMGRU_V1(nn.Module):
    """改进版本1: 添加注意力机制和残差连接"""

    def __init__(self, input_channels=2, input_length=512, num_classes=12, dropout=0.2):
        super(ImprovedLSTMGRU_V1, self).__init__()

        # 双向LSTM
        self.lstm = nn.LSTM(input_channels, 128, 2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)

        # 双向GRU
        self.gru = nn.GRU(256, 128, 2,  # 256 because bidirectional LSTM outputs 128*2
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout)

        # 自注意力机制
        self.attention = nn.MultiheadAttention(256, num_heads=8, dropout=dropout)

        # Layer Normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 256)

        # GRU处理
        gru_out, _ = self.gru(lstm_out)  # (batch, seq_len, 256)

        # 残差连接
        gru_out = self.ln1(gru_out + lstm_out)

        # 自注意力机制 (需要转换维度: batch_first -> seq_first)
        gru_out_t = gru_out.transpose(0, 1)  # (seq_len, batch, 256)
        attn_out, attn_weights = self.attention(gru_out_t, gru_out_t, gru_out_t)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, 256)

        # 残差连接
        out = self.ln2(attn_out + gru_out)

        # 全局平均池化和最大池化
        avg_pool = torch.mean(out, dim=1)  # (batch, 256)
        max_pool, _ = torch.max(out, dim=1)  # (batch, 256)

        # 特征融合
        out = avg_pool + max_pool

        # 分类
        out = self.fc(out)

        return out


class ImprovedLSTMGRU_V2(nn.Module):
    """改进版本2: CNN+RNN混合架构"""

    def __init__(self, input_channels=2, input_length=512, num_classes=12, dropout=0.2):
        super(ImprovedLSTMGRU_V2, self).__init__()

        # 1D卷积层用于局部特征提取
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # 计算卷积后的序列长度
        conv_output_length = input_length // 4  # 两次MaxPool1d

        # RNN层
        self.lstm = nn.LSTM(128, 128, 2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)

        self.gru = nn.GRU(256, 128, 1,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout)

        # Squeeze-and-Excitation模块
        self.se = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 转换为CNN格式: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # CNN特征提取
        conv_out = self.conv_blocks(x)  # (batch, 128, seq_len//4)

        # 转换回RNN格式
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len//4, 128)

        # RNN处理
        lstm_out, _ = self.lstm(conv_out)
        gru_out, _ = self.gru(lstm_out)

        # 时序池化
        avg_pool = torch.mean(gru_out, dim=1)
        max_pool, _ = torch.max(gru_out, dim=1)

        # 特征融合
        pooled = avg_pool + max_pool  # (batch, 256)

        # SE模块：通道注意力
        se_weights = self.se(pooled)
        out = pooled * se_weights

        # 分类
        out = self.fc(out)

        return out


class ImprovedLSTMGRU_V3(nn.Module):
    """改进版本3: 多尺度特征提取 + Transformer"""

    def __init__(self, input_channels=2, input_length=512, num_classes=12, dropout=0.2):
        super(ImprovedLSTMGRU_V3, self).__init__()

        # 多尺度卷积
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm1d(192)  # 64*3

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, input_length, 192))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # BiLSTM
        self.lstm = nn.LSTM(192, 128, 2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)

        # 时序卷积网络 (TCN)
        self.tcn = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # 全局上下文注意力
        self.global_attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 多尺度卷积特征提取
        x_t = x.transpose(1, 2)  # (batch, channels, seq_len)

        conv1_out = self.conv1(x_t)
        conv2_out = self.conv2(x_t)
        conv3_out = self.conv3(x_t)

        # 特征拼接
        multi_scale = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)  # (batch, 192, seq_len)
        multi_scale = self.bn(multi_scale)
        multi_scale = multi_scale.transpose(1, 2)  # (batch, seq_len, 192)

        # 添加位置编码
        multi_scale = multi_scale + self.pos_encoding

        # Transformer编码
        trans_out = self.transformer(multi_scale)  # (batch, seq_len, 192)

        # BiLSTM处理
        lstm_out, _ = self.lstm(trans_out)  # (batch, seq_len, 256)

        # TCN处理（残差）
        lstm_out_t = lstm_out.transpose(1, 2)  # (batch, 256, seq_len)
        tcn_out = self.tcn(lstm_out_t)
        tcn_out = tcn_out.transpose(1, 2)  # (batch, seq_len, 256)

        # 残差连接
        combined = lstm_out + tcn_out

        # 全局上下文注意力
        attn_weights = self.global_attn(combined)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权聚合
        context = torch.sum(combined * attn_weights, dim=1)  # (batch, 256)

        # 分类
        output = self.classifier(context)

        return output


class ImprovedLSTMGRU_Lightweight(nn.Module):
    """轻量级改进版本：适合实时应用"""

    def __init__(self, input_channels=2, input_length=512, num_classes=12, dropout=0.1):
        super(ImprovedLSTMGRU_Lightweight, self).__init__()

        # 深度可分离卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=7,
                      padding=3, groups=input_channels),
            nn.Conv1d(input_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 单层双向GRU（比LSTM更快）
        self.gru = nn.GRU(64, 64, 1,
                          batch_first=True,
                          bidirectional=True)

        # 简化的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 卷积特征提取
        x = x.transpose(1, 2)
        conv_out = self.depthwise_conv(x)  # (batch, 64, seq_len//2)
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len//2, 64)

        # GRU处理
        gru_out, _ = self.gru(conv_out)  # (batch, seq_len//2, 128)

        # 注意力池化
        attn_weights = self.attention(gru_out)  # (batch, seq_len//2, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权平均
        context = torch.sum(gru_out * attn_weights, dim=1)  # (batch, 128)

        # 分类
        output = self.classifier(context)

        return output


# 使用示例
def test_models():
    """测试所有改进模型"""
    batch_size = 32
    seq_len = 512
    input_channels = 2
    num_classes = 12

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, input_channels)

    # 测试各个模型
    models = [
        ("V1: 注意力+残差", ImprovedLSTMGRU_V1()),
        ("V2: CNN+RNN混合", ImprovedLSTMGRU_V2()),
        ("V3: 多尺度+Transformer", ImprovedLSTMGRU_V3()),
        ("轻量级版本", ImprovedLSTMGRU_Lightweight())
    ]

    for name, model in models:
        model.eval()
        with torch.no_grad():
            output = model(x)
            print(f"{name}: 输出形状 {output.shape}")
            params = sum(p.numel() for p in model.parameters())
            print(f"  参数量: {params:,}\n")


if __name__ == "__main__":
    test_models()