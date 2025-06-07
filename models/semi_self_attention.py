# models/semi_self_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

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

class OptimizedSemiSelfAttention(nn.Module):
    """优化的半自注意力机制"""
    def __init__(self, config, final_dim):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = final_dim // self.n_head
        self.final_dim = final_dim
        
        # Shared路径（Q,K）
        self.qk_proj = nn.Linear(final_dim, final_dim * 2, bias=False)
        
        # Private路径（V）- 使用LoRA
        self.v_proj = nn.Linear(final_dim, final_dim, bias=True)
        self.v_transform = LoRAAdapter(final_dim, rank=min(16, final_dim // 4))
        
        # 输出投影
        self.c_proj = nn.Linear(final_dim, final_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash Attention
        self.use_flash_attn = hasattr(F, 'scaled_dot_product_attention')
        
        # 局部注意力窗口
        self.local_window_size = 64
        
    def forward(self, h_private, h_shared, attn_dim=-2):
        """前向传播"""
        # 保存原始形状
        orig_shape = h_private.shape
        
        # 将张量reshape为3D用于注意力计算
        if attn_dim == 1:  # 时间注意力
            # (B, T, N, D) -> (B*N, T, D)
            B, T, N, D = h_private.shape
            h_private = h_private.permute(0, 2, 1, 3).reshape(B * N, T, D)
            h_shared = h_shared.permute(0, 2, 1, 3).reshape(B * N, T, D)
            L = T
        else:  # 空间注意力 (默认 attn_dim=-2 或 attn_dim=2)
            # (B, T, N, D) -> (B*T, N, D)
            B, T, N, D = h_private.shape
            h_private = h_private.reshape(B * T, N, D)
            h_shared = h_shared.reshape(B * T, N, D)
            L = N
        
        C = D  # 特征维度
        
        # 从shared生成Q,K
        qk = self.qk_proj(h_shared)
        q, k = qk.chunk(2, dim=-1)
        
        # 从private生成V
        v = self.v_proj(h_private)
        v = self.v_transform(v)
        
        # Reshape for multi-head attention
        # (B_new, L, D) -> (B_new, n_head, L, head_dim)
        q = q.view(-1, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(-1, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(-1, L, self.n_head, self.head_dim).transpose(1, 2)
        
        # 选择注意力策略
        if L > self.local_window_size * 2:
            y = self._local_attention(q, k, v)
        elif self.use_flash_attn and torch.cuda.is_available():
            y = self._flash_attention(q, k, v)
        else:
            y = self._standard_attention(q, k, v)
        
        # 合并多头
        # (B_new, n_head, L, head_dim) -> (B_new, L, D)
        y = y.transpose(1, 2).contiguous().view(-1, L, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        
        # 恢复原始形状
        if attn_dim == 1:  # 时间注意力
            # (B*N, T, D) -> (B, T, N, D)
            y = y.view(B, N, T, D).permute(0, 2, 1, 3)
        else:  # 空间注意力
            # (B*T, N, D) -> (B, T, N, D)
            y = y.view(B, T, N, D)
        
        return y
    
    def _standard_attention(self, q, k, v):
        """标准注意力 - 输入形状: (B, H, L, D)"""
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, v)
    
    def _flash_attention(self, q, k, v):
        """Flash Attention - 输入形状: (B, H, L, D)"""
        # Flash attention需要 (B, L, H, D) 格式
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        # 转回 (B, H, L, D)
        return y.transpose(1, 2)
    
    def _local_attention(self, q, k, v):
        """局部滑动窗口注意力 - 输入形状: (B, H, L, D)"""
        B, H, L, D = q.shape
        window_size = self.local_window_size
        output = torch.zeros_like(v)
        
        for i in range(0, L, window_size // 2):
            start = max(0, i - window_size // 2)
            end = min(L, i + window_size)
            
            local_q = q[:, :, start:end, :]
            
            ctx_start = max(0, start - window_size // 4)
            ctx_end = min(L, end + window_size // 4)
            local_k = k[:, :, ctx_start:ctx_end, :]
            local_v = v[:, :, ctx_start:ctx_end, :]
            
            local_attn = self._standard_attention(local_q, local_k, local_v)
            output[:, :, start:end, :] = local_attn
        
        return output

class OptimizedSemiSelfAttentionBlock(nn.Module):
    """优化的半自注意力块"""
    def __init__(self, config):
        super().__init__()
        final_dim = config.n_embd
        
        # Layer Normalization
        self.ln_1_priv = nn.LayerNorm(final_dim, eps=1e-5, elementwise_affine=config.bias)
        self.ln_1_shared = nn.LayerNorm(final_dim, eps=1e-5, elementwise_affine=config.bias)
        
        # 半自注意力层
        self.attn = OptimizedSemiSelfAttention(config, final_dim)
        
        # Feed Forward
        self.ln_2 = nn.LayerNorm(final_dim, eps=1e-5, elementwise_affine=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, 4 * final_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * final_dim, final_dim),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, h_private, h_shared, attn_dim=-2):
        # 注意力计算
        h_private_ln = self.ln_1_priv(h_private)
        h_shared_ln = self.ln_1_shared(h_shared)
        
        attn_output = self.attn(h_private_ln, h_shared_ln, attn_dim)
        h_private = h_private + attn_output
        
        # FFN
        h_private = h_private + self.mlp(self.ln_2(h_private))
        
        return h_private, h_shared