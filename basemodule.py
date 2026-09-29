import torch
import torch.nn as nn
from einops import rearrange, einsum

# 实现Linear层
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        # 创建w权重参数
        self.w = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))
        mean, std = 0, 2 / (in_features + out_features) ** 0.5
        # 初始化w权重参数
        nn.init.normal_(self.w, mean = mean, std = std, a = -3*std, b = 3 * std)

        def forward(self, x: torch.Tensor) -> torch.Tensor: 
            y = einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")
            return y


# 实现Embedding层
class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device = None, dtype = None):
        super().__init__()
        self.embed = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype))
        mean, std = 0, 1
        nn.init.normal_(self.embed, mean = mean, std = std, a = -3*std, b = 3 * std)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 查找token_ids对应的embedding向量
        return self.embed[token_ids]

