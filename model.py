# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def _infer_flatten_dim(module: nn.Module, in_shape: torch.Size) -> int:
    """
    Infer flattened dimension after feature extractor by doing a dry forward pass
    with a zeros tensor of shape (1, *in_shape). This avoids hard-coding fc in_features.
    """
    module.eval()
    with torch.no_grad():
        x = torch.zeros((1, *in_shape))  # (B=1, C, T)
        y = module(x)
        y = y.view(y.size(0), -1)
    return y.size(1)


# -----------------------------
# Positional Encoding (sine-cosine)
# -----------------------------
class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding (Add & Dropout).
    x expected shape: (B, T, D)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)           # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, T, D)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# -----------------------------
# Transformer Encoder Classifier
# -----------------------------
class TransformerModel(nn.Module):
    """
    Input:  (B, C, T)
    Output: (B, num_classes)
    """
    def __init__(self,
                 num_features: int = 51,
                 time_steps: int = 512,
                 num_classes: int = 10,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.time_steps = time_steps
        self.d_model = d_model

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=time_steps)

        # Use batch_first=False to keep the same behavior as classic Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)

        # Embedding + Positional encoding
        x = self.input_proj(x)         # (B, T, D)
        x = self.pos_encoder(x)        # (B, T, D)

        # Transformer expects (T, B, D)
        x = x.transpose(0, 1)          # (T, B, D)
        x = self.transformer_encoder(x)

        # Temporal aggregation (mean pooling)
        x = x.mean(dim=0)              # (B, D)
        x = self.fc(x)                 # (B, num_classes)
        return x


# # -----------------------------
# # Simple MLP
# # -----------------------------
# class FCModel(nn.Module):
#     """
#     Input:  (B, C, T) with T==1 typically -> squeeze to (B, C)
#     Output: (B, num_classes)
#     """
#     def __init__(self, num_features: int = 51, num_classes: int = 10):
#         super().__init__()
#         self.fc1 = nn.Linear(num_features, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, C, T) -> (B, C) if T==1
#         x = x.squeeze(-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x


# # -----------------------------
# # Plain CNN 1D
# # -----------------------------
# class _CNNFeatureExtractor(nn.Module):
#     """
#     Shared CNN feature extractor without final FC.
#     Keeps your original conv/pool stack. Used for automatic flatten-dim inference.
#     """
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3)
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
#         self.relu = nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, C, T)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         return x  # (B, 128, T')


# class CNNModel(nn.Module):
#     """
#     Input:  (B, C, T)
#     Output: (B, num_classes)
#     """
#     def __init__(self, num_features: int = 51, time_steps: int = 512, num_classes: int = 10):
#         super().__init__()
#         self.feat = _CNNFeatureExtractor(in_channels=num_features)
#         # Infer flatten size based on time_steps to avoid hard-coding
#         flat_dim = _infer_flatten_dim(self.feat, torch.Size([num_features, time_steps]))
#         self.fc1 = nn.Linear(flat_dim, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.feat(x)
#         x = x.view(x.size(0), -1)  # flatten
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# -----------------------------
# Simple MLP
# -----------------------------

class FCModel(nn.Module):
    """
    Input:  (B, C, T) -> flattens to (B, C * T)
    Output: (B, num_classes)
    """
    def __init__(self, num_features: int = 52, time_steps: int = 200, num_classes: int = 45):
        super().__init__()
        # 把每一帧的特征数(num_features)和时间步长(time_steps)乘起来，作为全连接层的输入维度
        flattened_dim = num_features * time_steps
        
        self.fc1 = nn.Linear(flattened_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的初始形状是 (Batch, Channels, TimeSteps) 即 (B, C, T)
        # 【关键修改】：将 view 改为 reshape，以支持非连续内存的张量展平
        x = x.reshape(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# -----------------------------
# Plain CNN 1D (自适应长度版)
# -----------------------------
class _CNNFeatureExtractor(nn.Module):
    """
    带有自适应池化层的 CNN 特征提取器。
    不管输入的序列长度 (time_steps) 是多少，输出始终是固定的 (B, 128, 1)。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu = nn.ReLU()
        
        # 【核心修改】：添加自适应平均池化层，将任意时间维度压缩为 1
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 将 (B, 128, T') 强制池化为 (B, 128, 1)
        x = self.adaptive_pool(x)
        return x


class CNNModel(nn.Module):
    """
    Input:  (B, C, T) 
    Output: (B, num_classes)
    """
    def __init__(self, num_features: int = 52, time_steps: int = 200, num_classes: int = 45):
        super().__init__()
        self.feat = _CNNFeatureExtractor(in_channels=num_features)
        
        # 【核心修改】：因为用了 AdaptiveAvgPool1d(1)，展平后的维度永远是 128
        # 再也不需要 _infer_flatten_dim 去推断长度了，彻底告别长度不匹配的报错！
        flat_dim = 128 
        
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feat(x)
        x = x.view(x.size(0), -1)  # flatten: (B, 128, 1) -> (B, 128)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------------
# LSTM Classifier
# -----------------------------
class LSTMModel(nn.Module):
    """
    Input:  (B, C, T) -> permute to (B, T, C)
    Output: (B, num_classes)
    """
    def __init__(self, input_size: int = 512, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 30):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))    # (B, T, H)
        out = out[:, -1, :]                # last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

