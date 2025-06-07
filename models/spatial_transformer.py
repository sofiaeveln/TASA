import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LoRAAdapter(nn.Module):
    """使用LoRA (Low-Rank Adaptation) 降低私有参数的复杂度"""
    def __init__(self, dim, rank=16, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 低秩矩阵分解
        self.lora_A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, dim))
        self.scaling = self.alpha / self.rank
        
    def forward(self, x):
        lora_output = (x @ self.lora_A) @ self.lora_B
        return x + lora_output * self.scaling

class EfficientDualAttBlock(nn.Module):
    """改回原始DualAttBlock实现，但集成LoRA优化"""
    
    def __init__(self, config):
        super().__init__()
        
        # 从config中提取参数
        hidden_size = config.n_embd
        num_heads = config.n_head
        self.num = config.blocknum // config.factors  # P
        self.size = config.blocksize * config.factors  # N
        mlp_ratio = 4.0  # 恢复原始的4倍MLP
        dropout = config.dropout
        bias = config.bias
        
        self.hidden_size = hidden_size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.dropout = dropout

        # Intra-Block Attention (原始实现)
        self.snorm1 = LayerNorm(hidden_size, bias=bias)
        self.sattn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, bias=True)
        self.snorm2 = LayerNorm(hidden_size, bias=bias)
        
        # Intra-Block MLP with LoRA
        self.smlp_base = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.smlp_lora = LoRAAdapter(hidden_size, rank=min(16, hidden_size // 4))

        # Inter-Block Attention (原始实现)
        self.nnorm1 = LayerNorm(hidden_size, bias=bias)
        self.nattn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, bias=True)
        self.nnorm2 = LayerNorm(hidden_size, bias=bias)
        
        # Inter-Block MLP with LoRA
        self.nmlp_base = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.nmlp_lora = LoRAAdapter(hidden_size, rank=min(16, hidden_size // 4))

    def forward(self, x_private, x_shared):
        # x_private, x_shared: [B,T,N,D]
        B, T, N, D = x_private.shape
        P, S = self.num, self.size
        assert P * S >= N, "P*S must be >= N"

        if P * S > N:
            pad_size = P * S - N
            pad_tensor = torch.zeros(B, T, pad_size, D).to(x_private.device)
            x_private = torch.cat([x_private, pad_tensor], dim=2)  # (B, T, P*S, D)
            x_shared = torch.cat([x_shared, pad_tensor], dim=2)    # (B, T, P*S, D)
            N_padded = P * S  # 更新N
        else:
            N_padded = N

        # Intra-Attention
        # reshape为(B*T*P, S, D)
        x_private_reshaped = x_private.reshape(B * T * P, S, D)
        x_shared_reshaped = x_shared.reshape(B * T * P, S, D)

        # Q,K来自shared，V来自private
        # 先LayerNorm
        qs = self.snorm1(x_shared_reshaped)  # Q,K
        vs = self.snorm1(x_private_reshaped)  # V 使用与Q,K相同的归一化处理，但实际V来自private表示
        qs = qs.transpose(0, 1)  # (S, B*T*P, D)
        ks = qs
        vs = vs.transpose(0, 1)  # (S, B*T*P, D)

        attn_out, _ = self.sattn(qs, ks, vs)
        attn_out = attn_out.transpose(0, 1)  # (B*T*P, S, D)
        x_private_reshaped = x_private_reshaped + attn_out
        
        # MLP with LoRA
        mlp_input = self.snorm2(x_private_reshaped)
        mlp_base_out = self.smlp_base(mlp_input)
        mlp_lora_out = self.smlp_lora(mlp_base_out)
        x_private_reshaped = x_private_reshaped + mlp_lora_out

        # Inter-Block Attention
        # 将 (B, T, P, S, D) 转换为 (B*T*S, P, D)
        x_private_local = x_private_reshaped.reshape(B, T, P, S, D).transpose(2, 3).reshape(B * T * S, P, D)
        x_shared_local = x_shared_reshaped.reshape(B, T, P, S, D).transpose(2, 3).reshape(B * T * S, P, D)

        qb = self.nnorm1(x_shared_local)  # Q,K from shared
        vb = self.nnorm1(x_private_local)  # V from private
        qb = qb.transpose(0, 1)  # (P, B*T*S, D)
        kb = qb
        vb = vb.transpose(0, 1)  # (P, B*T*S, D)
        attn_out_breadth, _ = self.nattn(qb, kb, vb)
        attn_out_breadth = attn_out_breadth.transpose(0, 1)  # (B*T*S, P, D)

        x_private_local = x_private_local + attn_out_breadth
        
        # MLP with LoRA
        mlp_input_breadth = self.nnorm2(x_private_local)
        mlp_base_out_breadth = self.nmlp_base(mlp_input_breadth)
        mlp_lora_out_breadth = self.nmlp_lora(mlp_base_out_breadth)
        x_private_local = x_private_local + mlp_lora_out_breadth

        # 重塑回原形状 (B*T*S, P, D) -> (B, T, S, P, D) -> (B, T, P * S, D) -> (B, T, N, D)
        x_private_final = x_private_local.reshape(B, T, S, P, D).transpose(2, 3).reshape(B, T, P * S, D)

        if P * S > N:
            # 去掉填充部分
            x_private_final = x_private_final[:, :, :N, :]
            x_shared_final = x_shared[:, :, :N, :]  # 同样去掉填充
        else:
            x_shared_final = x_shared

        return x_private_final, x_shared_final

class LinearAttention(nn.Module):
    """线性注意力机制"""
    def __init__(self, dim, heads=4, dim_head=None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head or (dim // heads)
        
        self.to_qkv = nn.Linear(dim, self.dim_head * heads * 3, bias=False)
        self.to_out = nn.Linear(self.dim_head * heads, dim)
        
        self.feature_map = nn.Sequential(
            nn.Linear(self.dim_head, self.dim_head),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads, self.dim_head).transpose(1, 2), qkv)
        
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        k_v = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, k_v)
        
        normalizer = torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=1, keepdim=False))
        out = out / (normalizer.unsqueeze(-1) + 1e-6)
        
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)