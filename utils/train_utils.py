import torch
import numpy as np
import random
import logging
from datetime import datetime
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from scipy import stats

def set_random_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logger(log_dir='./logs/', log_prefix=''):
    """设置日志"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s',
            "%Y-%m-%d %H:%M:%S")
        
        # 控制台日志
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        # 文件日志
        ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        fh = RotatingFileHandler(f'{log_dir}/{log_prefix}-{ts}.log', maxBytes=10*1024*1024, backupCount=5)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

# 导入数据处理函数
from utils.data_utils import (
    read_meta_datasets, generate_data, get_dataloader,
    construct_adj, reorderData, kdTree, compute_node_clusters
)

def compute_grad_norm(model):
    """计算模型的梯度范数"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)

def compute_mape(pred, target):
    """计算MAPE指标"""
    epsilon = 1e-8
    return torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100

def evaluate_with_significance(model, test_loader, baseline_results=None, num_runs=5):
    """包含统计显著性测试的评估函数"""
    our_results = {h: {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': []} 
                   for h in model.config.horizons}
    
    # 多次运行收集结果
    for seed in range(num_runs):
        set_random_seed(seed)
        run_results = model.evaluate(test_loader)
        for h in model.config.horizons:
            for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                our_results[h][metric].append(run_results[h][metric])
    
    # 计算统计量
    final_results = {}
    for h in model.config.horizons:
        final_results[h] = {}
        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            values = our_results[h][metric]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # 统计检验
            if baseline_results and h in baseline_results:
                baseline_vals = baseline_results[h][metric]
                # 配对t检验
                t_stat, p_value = stats.ttest_rel(values, baseline_vals)
                # Wilcoxon秩和检验
                w_stat, w_p_value = stats.wilcoxon(values, baseline_vals)
            else:
                t_stat, p_value, w_stat, w_p_value = None, None, None, None
            
            final_results[h][metric] = {
                'mean': mean_val,
                'std': std_val,
                't_test': {'statistic': t_stat, 'p_value': p_value},
                'wilcoxon': {'statistic': w_stat, 'p_value': w_p_value}
            }
    
    return final_results
def evaluate_with_statistical_tests(model, test_loader, num_runs=5):
    """🔧 新增：包含统计显著性测试的评估"""
    import scipy.stats as stats
    
    all_results = []
    for seed in range(num_runs):
        # 设置不同随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 评估
        results = model.evaluate(test_loader)
        all_results.append(results)
    
    # 计算统计量
    final_results = {}
    for h in model.config.horizons:
        final_results[h] = {}
        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            values = [r[h][metric] for r in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # 计算95%置信区间
            ci_95 = stats.t.interval(0.95, len(values)-1, 
                                   loc=mean_val, 
                                   scale=stats.sem(values))
            
            final_results[h][metric] = {
                'mean': mean_val,
                'std': std_val,
                'ci_95': ci_95,
                'values': values
            }
            
            print(f"Horizon {h} {metric}: {mean_val:.3f} ± {std_val:.3f} "
                 f"(95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}])")
    
    return final_results