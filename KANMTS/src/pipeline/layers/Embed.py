import math

import torch
import torch.nn as nn
from layers.KANLinear import KANLinear



class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, grid_size,dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        # self.value_embedding = KANLinear(c_in,  d_model, grid_size=grid_size)
        self.dropout = nn.Dropout(p=dropout)

# 32,96,137
    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (32,7,96)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 结果的形状为 (32, 11, 96)，其中 11 是 7 + 4。
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))  # (32,11,512)
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # 创建位置编码矩阵.这里创建了一个大小为 (max_len, d_model) 的全零矩阵
        # pe，并将其数据类型设置为浮点型。require_grad = False 表示这个矩阵不需要梯度，即在反向传播时不计算其梯度。
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 生成位置索引
        # 这行代码生成了一个从 0 到 max_len-1 的序列，并将其转换为浮点型，然后通过 unsqueeze(1) 增加一个维度，使其形状变为 (max_len, 1)。
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # 计算除法项
        # 生成了一个从 0 到 d_model-1 的偶数序列，并将其转换为浮点型。然后计算 -(math.log(10000.0) / d_model) 并乘以这个序列，最后取指数。
        # 这个 div_term 用于生成位置编码中的正弦和余弦项。
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        # 填充位置编码矩阵的偶数和奇数列
        # 0::2 表示从第 0 列开始，每隔 2 列填充一次。
        # 1::2 表示从第 1 列开始，每隔 2 列填充一次。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个维度并注册缓冲区
        # 这里增加了一个维度，使得 pe 的形状变为 (1, max_len, d_model)，并注册为缓冲区。
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # 后使用 register_buffer 将 pe 注册为一个缓冲区，这样它会被自动保存并在模型加载时恢复，但不会被优化器更新。

    def forward(self, x):
        return self.pe[:, :x.size(1)]
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # 32,7,96
        # do patching
        n_vars = x.shape[1]  # 7
        x = self.padding_patch_layer(x)  # 32,7,104  使用 padding_patch_layer 对输入 x 进行填充。
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 224,12,16
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)  # 224,12,512
        return self.dropout(x), n_vars