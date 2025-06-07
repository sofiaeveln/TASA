import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from .semi_self_attention import OptimizedSemiSelfAttentionBlock, LoRAAdapter
from .spatial_transformer import EfficientDualAttBlock

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTASAConfig:
    """优化后的TASA配置"""
    # 基本参数
    seed: int = 0
    data: str = ''
    datapath: str = ''
    seq_len: int = 12
    horizons: List[int] = field(default_factory=lambda: [3, 6, 12])
    num_nodes: int = 325
    node_dim: int = 2
    device: torch.device = field(default_factory=lambda: torch.device('cuda:0'))
    
    # 模型架构参数
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = True
    n_linear: int = 1
    
    # 嵌入参数
    input_dim: int = 1  # 🎯 修改为1，只使用交通数据
    tod_embedding_dim: int = 12
    dow_embedding_dim: int = 6
    spatial_embedding_dim: int = 6
    adaptive_embedding_dim: int = 36
    steps_per_day: int = 288
    output_dim: int = 1
    
    # 空间划分参数
    blocksize: int = 8
    blocknum: int = 4
    factors: int = 1
    
    # 训练参数
    meta_lr: float = 5e-5
    update_lr: float = 1e-3
    meta_epochs: int = 5
    city_epochs: int = 1
    test_epochs: int = 50
    
    # 优化参数
    use_lora: bool = True
    lora_rank: int = 16
    use_flash_attn: bool = True
    gradient_checkpointing: bool = True
    
    # 领域特定参数
    domain_specific_params: List[str] = field(default_factory=lambda: ['value_mlp', 'value_transform', 'predictors'])
    
    # 层数配置
    temporal_layers: int = 1
    spatial_layers: int = 1
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

class OptimizedSTAE(nn.Module):
    """优化的空间-时间自适应嵌入层"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 🔧 关键修正：恢复原始input_dim=2的处理
        self.input_proj = nn.Linear(config.input_dim, config.n_embd)

        # 🔧 修正：恢复传统嵌入方式，避免信息丢失
        if config.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(config.steps_per_day, config.tod_embedding_dim)
        else:
            self.tod_embedding = None
            
        if config.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, config.dow_embedding_dim)
        else:
            self.dow_embedding = None
        
        # 🔧 修正：恢复原始空间嵌入，避免聚类丢失信息
        if config.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(config.num_nodes, config.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb = None
        
        # 🔧 修正：恢复原始自适应嵌入
        if config.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(config.seq_len, config.num_nodes, config.adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.adaptive_embedding = None
        
        # 🔧 修正：恢复原始维度计算
        self.output_dim = (config.n_embd 
                         + config.tod_embedding_dim 
                         + config.dow_embedding_dim 
                         + config.spatial_embedding_dim 
                         + config.adaptive_embedding_dim)
    
    def _compute_node_clusters(self, num_nodes):
        """计算节点聚类"""
        nodes_per_cluster = max(1, num_nodes // self.num_clusters)
        node_to_cluster = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(num_nodes):
            node_to_cluster[i] = min(i // nodes_per_cluster, self.num_clusters - 1)
        return node_to_cluster
    
    def forward(self, x):
        """🔧 修正：恢复原始嵌入逻辑"""
        B, T, N, C = x.size()
        features = []

        # 基础input投射 - 恢复原始逻辑
        main_input = x[..., :self.config.input_dim]  # 现在是前2个特征
        main_input = self.input_proj(main_input)
        features.append(main_input)

        # tod embedding - 恢复原始逻辑
        if self.tod_embedding is not None and C > self.config.input_dim:
            tod_idx = (x[..., self.config.input_dim] * self.config.steps_per_day).long().clamp(0, self.config.steps_per_day-1)
            tod_emb = self.tod_embedding(tod_idx)
            features.append(tod_emb)

        # dow embedding - 恢复原始逻辑
        if self.dow_embedding is not None and C > self.config.input_dim+1:
            dow_idx = x[..., self.config.input_dim+1].long().clamp(0, 6)
            dow_emb = self.dow_embedding(dow_idx)
            features.append(dow_emb)

        # spatial embedding - 恢复原始逻辑
        if self.node_emb is not None:
            spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, N, self.config.spatial_embedding_dim)
            features.append(spatial_emb)

        # adaptive embedding - 恢复原始逻辑
        if self.adaptive_embedding is not None:
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(B, T, N, self.config.adaptive_embedding_dim)
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)
        return x

class CityAdaptationMechanism(nn.Module):
    """城市适应机制"""
    def __init__(self, input_dim: int, hidden_dim: int, lora_rank: int):
        super().__init__()
        # 共享知识提取器
        self.shared_knowledge_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 城市特定适配器
        self.city_adapter = LoRAAdapter(hidden_dim, rank=lora_rank)
        
        # 融合权重
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_components=False):
        """前向传播"""
        shared_knowledge = self.shared_knowledge_extractor(x)
        city_specific = self.city_adapter(shared_knowledge)
        
        concat_features = torch.cat([shared_knowledge, city_specific], dim=-1)
        gate = self.fusion_gate(concat_features)
        output = gate * shared_knowledge + (1 - gate) * city_specific
        
        if return_components:
            return output, {
                'shared': shared_knowledge,
                'specific': city_specific,
                'gate': gate
            }
        return output

class OptimizedTASA(nn.Module):
    def __init__(self, config: OptimizedTASAConfig):
        super().__init__()
        self.config = config

        # 1. 嵌入层
        self.embedding = OptimizedSTAE(config)
        final_dim = self.embedding.output_dim

        # 2. 城市适应机制
        self.city_adaptation = CityAdaptationMechanism(
            input_dim=final_dim,
            hidden_dim=config.n_embd,
            lora_rank=config.lora_rank
        )

        # 3. 参数分离投影
        self.shared_proj = nn.Linear(config.n_embd, config.n_embd)
        self.private_proj = nn.Linear(config.n_embd, config.n_embd)

        # 4. 时间 Transformer
        self.temporal_blocks = nn.ModuleList([
            OptimizedSemiSelfAttentionBlock(config)
            for _ in range(config.temporal_layers)
        ])

        # 5. 空间 Transformer - 使用改回原始实现的版本
        self.spatial_transformer = EfficientDualAttBlock(config)

        # 6. 输出层
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.predictors = nn.ModuleDict({
            str(h): nn.Linear(config.n_embd, h * config.output_dim)
            for h in config.horizons
        })

        # 7. 初始化
        self._init_weights()

        # 8. 配置参数组
        self._configure_parameter_groups()

        logger.info(f"Model initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight is not None:
                    torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                if module.weight is not None:
                    torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def _configure_parameter_groups(self):
        """配置参数组"""
        self.shared_params = []
        self.private_params = []
        self.domain_specific_params = []
        
        for name, param in self.named_parameters():
            if any(keyword in name for keyword in self.config.domain_specific_params):
                self.domain_specific_params.append(param)
            elif 'shared' in name or 'embedding' in name:
                self.shared_params.append(param)
            else:
                self.private_params.append(param)
        
        logger.info(f"Parameter groups - Shared: {len(self.shared_params)}, "
                   f"Private: {len(self.private_params)}, "
                   f"Domain-specific: {len(self.domain_specific_params)}")
    
    def get_num_params(self):
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def copy_invariant_params(self, target_model):
        """从元模型复制不变参数到目标模型（简化版本）"""
        source_dict = self.state_dict()
        target_dict = target_model.state_dict()
        
        # 只复制完全匹配的参数
        for name, param in source_dict.items():
            # 跳过领域特定参数
            if any(keyword in name for keyword in self.config.domain_specific_params):
                continue
                
            # 只复制存在且形状完全匹配的参数
            if name in target_dict and param.shape == target_dict[name].shape:
                target_dict[name] = param.clone()
        
        # 使用strict=False忽略不匹配的参数
        target_model.load_state_dict(target_dict, strict=False)
    
    def forward(self, x, y=None, train=True, stage='target',
                ori_parts_idx=None, reo_parts_idx=None, reo_all_idx=None):
        B, T, N, C = x.shape
        
        # 1. 嵌入
        x_embedded = self.embedding(x)
        
        # 2. 城市适应
        x_adapted = self.city_adaptation(x_embedded)
        
        # 3. 参数分离
        h_shared = self.shared_proj(x_adapted)
        h_private = self.private_proj(x_adapted)
        
        # 4. 时间建模
        for block in self.temporal_blocks:
            if self.config.gradient_checkpointing and self.training:
                h_private, h_shared = torch.utils.checkpoint.checkpoint(
                    block, h_private, h_shared, 1
                )
            else:
                h_private, h_shared = block(h_private, h_shared, attn_dim=1)
        
        # 5. 空间建模
        if self.config.gradient_checkpointing and self.training:
            h_private, h_shared = torch.utils.checkpoint.checkpoint(
                self.spatial_transformer, h_private, h_shared
            )
        else:
            h_private, h_shared = self.spatial_transformer(h_private, h_shared)
        
        # 6. 融合
        h_final = h_private + h_shared
        h_final = self.ln_f(h_final)
        h_last = h_final[:, -1, :, :]
        
        # 7. 预测
        preds = {}
        for h in self.config.horizons:
            out = self.predictors[str(h)](h_last)
            out = out.view(B, N, h, self.config.output_dim)
            out = out.permute(0, 2, 1, 3).squeeze(-1)
            preds[h] = out
        
        if train and y is not None:
            loss = self.compute_loss(preds, y, stage)
            return preds, loss
        
        return preds
    
    def compute_loss(self, preds, targets, stage='target'):
        """计算损失"""
        total_loss = 0
        loss_weights = {'meta': 1.0, 'source': 1.0, 'target': 1.5}
        weight = loss_weights.get(stage, 1.0)
        
        for h in self.config.horizons:
            if h in preds and h in targets:
                l1 = F.l1_loss(preds[h], targets[h])
                l2 = F.mse_loss(preds[h], targets[h])
                total_loss += weight * (l1 + 0.5 * l2)
        
        return total_loss / len(self.config.horizons)
    
    def configure_optimizers(self, weight_decay=0.01, learning_rate=1e-3, 
                             betas=(0.9, 0.999), device_type='cuda'):
        """配置优化器"""
        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = torch.optim.AdamW(
            [{'params': decay_params, 'weight_decay': weight_decay},
             {'params': no_decay_params, 'weight_decay': 0.0}],
            lr=learning_rate, betas=betas
        )
        return optimizer
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        """评估模型 - 添加Z-score反归一化处理"""
        self.eval()
        preds = {h: [] for h in self.config.horizons}
        targets = {h: [] for h in self.config.horizons}
        
        # 🎯 关键修改：获取scaler用于反归一化
        scaler = getattr(data_loader, 'scaler', None)
        
        for x, y in data_loader:
            x = x.to(self.config.device)
            y = {h: y[h].to(self.config.device) for h in self.config.horizons}
            p = self(x, train=False)
            
            for h in self.config.horizons:
                pred = p[h].cpu().numpy()
                target = y[h].cpu().numpy()
                
                # 🎯 关键修改：进行反归一化处理
                if scaler is not None:
                    # 预测值反归一化
                    pred_shape = pred.shape
                    pred_flat = pred.reshape(-1, 1)
                    pred_denorm = scaler.inverse_transform(pred_flat)
                    pred = pred_denorm.reshape(pred_shape)
                    
                    # 目标值反归一化
                    target_shape = target.shape
                    target_flat = target.reshape(-1, 1)
                    target_denorm = scaler.inverse_transform(target_flat)
                    target = target_denorm.reshape(target_shape)
                
                preds[h].append(pred)
                targets[h].append(target)
        
        results = {}
        for h in self.config.horizons:
            pr = np.concatenate(preds[h], axis=0)
            tr = np.concatenate(targets[h], axis=0)
            
            # 计算评估指标
            mae = np.mean(np.abs(pr - tr))
            mse = np.mean((pr - tr) ** 2)
            rmse = np.sqrt(mse)
            # MAPE计算时避免除零
            mask = np.abs(tr) > 1e-5
            mape = np.mean(np.abs((tr[mask] - pr[mask]) / tr[mask])) * 100 if mask.any() else 0.0
            
            results[h] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
        
        return results
    
    def visualize_attention(self, x, layer_idx=0):
        """可视化注意力"""
        self.eval()
        with torch.no_grad():
            x_emb = self.embedding(x)
            x_ad, comps = self.city_adaptation(x_emb, return_components=True)
            if layer_idx < len(self.temporal_blocks):
                return comps
            return None