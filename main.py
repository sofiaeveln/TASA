import argparse
import os
import socket
import math
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from models import OptimizedTASA, OptimizedTASAConfig
from utils import (
    set_random_seed, set_logger,
    read_meta_datasets, generate_data, get_dataloader,
    construct_adj, reorderData, kdTree,
    evaluate_with_significance
)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cuda', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--horizons', nargs='+', type=int, default=None)
    parser.add_argument('--load_type', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    return args

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.cuda is not None:
        config['training']['cuda'] = args.cuda
    if args.seq_len is not None:
        config['data']['seq_len'] = args.seq_len
    if args.horizons is not None:
        config['data']['horizons'] = args.horizons
    if args.load_type is not None:
        config['training']['load_type'] = args.load_type
    if args.eval_only:
        config['training']['eval_only'] = True
    
    # 设置随机种子和设备
    set_random_seed(config['training']['seed'])
    hostname = socket.gethostname()
    device = torch.device(f"cuda:{config['training']['cuda']}" if torch.cuda.is_available() else "cpu")
    
    # 混合精度设置
    if config['optimization']['mixed_precision'] and 'cuda' in str(device):
        ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }.get(config['optimization']['dtype'], torch.float32)
        
        try:
            ctx = torch.autocast(device_type='cuda', dtype=ptdtype)
        except (AttributeError, TypeError):
            if ptdtype == torch.float16:
                ctx = autocast()
            else:
                ctx = None
                config['optimization']['mixed_precision'] = False
        
        if config['optimization']['mixed_precision']:
            scaler = GradScaler(enabled=(config['optimization']['dtype'] != 'float32'))
        else:
            scaler = None
    else:
        ctx = None
        scaler = None
    
    # 设置输出目录
    out_dir = config['output']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    # 设置日志
    log_prefix = f"TASA-{config['data']['target_city']}-train-{hostname}-gpu{config['training']['cuda']}"
    logger = set_logger(log_dir=config['output']['log_dir'], log_prefix=log_prefix)
    logger.info("Configuration loaded:")
    logger.info(config)
    
    # 🎯 关键信息日志
    logger.info(f"使用input_dim={config['model']['input_dim']}，只使用交通数据特征")
    logger.info(f"自动生成时间特征: TOD={config['model']['tod_embedding_dim']}, DOW={config['model']['dow_embedding_dim']}")
    
    # 读入数据集
    train_cities = config['data']['train_cities']
    target_city = config['data']['target_city']
    
    data_neural, meta_num_nodes = read_meta_datasets(
        train_cities=train_cities,
        test_city=target_city,
        path=config['data']['datapath'],
        tod_embedding_dim=config['model']['tod_embedding_dim'],
        dow_embedding_dim=config['model']['dow_embedding_dim'],
        steps_per_day=config['data']['steps_per_day']
    )
    
    # 生成数据集
    splits_dict = {}
    horizons = config['data']['horizons']
    seq_len = config['data']['seq_len']
    adaptation_days = config['data']['adaptation_days']
    steps_per_day = config['data']['steps_per_day']
    
    # 🔧 修正：移除scaler参数传递，让generate_data内部处理标准化
    for c in train_cities + [target_city]:
        if c == target_city:
            logger.info(f"Using limited data for target city {c}: {adaptation_days} days for adaptation")
            adaptation_dataset = data_neural[c]['dataset'][:adaptation_days * steps_per_day]
            remaining_dataset = data_neural[c]['dataset'][adaptation_days * steps_per_day:]
            
            # 🔧 修正：不传递scaler参数，让函数内部处理
            adaptation_splits = generate_data(adaptation_dataset, seq_len, horizons, 
                                            split_ratios=(1.0, 0.0, 0.0))
            splits_dict[c] = adaptation_splits
            
            eval_splits = generate_data(remaining_dataset, seq_len, horizons, 
                                      split_ratios=(0.7, 0.1, 0.2))
            splits_dict[f"{c}_eval"] = eval_splits
        else:
            dataset = data_neural[c]['dataset']
            # 🔧 修正：不传递scaler参数，让函数内部处理
            splits = generate_data(dataset, seq_len, horizons, 
                                 split_ratios=(0.7, 0.1, 0.2))
            splits_dict[c] = splits
    
    # 初始化数据加载器
    batch_size = config['optimization']['batch_size']
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    
    for c in train_cities:
        train_loaders[c] = get_dataloader(splits_dict[c], batch_size, split='train', shuffle=True, horizons=horizons)
        val_loaders[c] = get_dataloader(splits_dict[c], batch_size, split='val', shuffle=False, horizons=horizons)
        test_loaders[c] = get_dataloader(splits_dict[c], batch_size, split='test', shuffle=False, horizons=horizons)
    
    train_loaders[target_city] = get_dataloader(splits_dict[target_city], batch_size, split='train', shuffle=True, horizons=horizons)
    val_loaders[f"{target_city}_eval"] = get_dataloader(splits_dict[f"{target_city}_eval"], batch_size, split='val', shuffle=False, horizons=horizons)
    test_loaders[f"{target_city}_eval"] = get_dataloader(splits_dict[f"{target_city}_eval"], batch_size, split='test', shuffle=False, horizons=horizons)
    
    # 处理空间划分信息
    all_parts_idx_info = {}
    blocknum_per_city = {}
    
    for c in train_cities + [target_city]:
        coords_path = os.path.join(config['data']['datapath'], c, 'coords.npy')
        if not os.path.exists(coords_path):
            logger.warning(f"Coords file not found for city {c}, skip KDTree for it.")
            all_parts_idx_info[c] = (None, None, None)
            blocknum_per_city[c] = config['model']['blocknum']
            continue
            
        coords = np.load(coords_path)
        locations = coords.T
        
        recur_times = 2
        parts_idx, mxlen = kdTree(locations, recur_times, 0)
        
        from utils.data_utils import construct_adj_safe
        dataset = data_neural[c]['dataset']
        
        # 只用训练集构建邻接矩阵，避免数据泄露
        adj = construct_adj_safe(dataset, dataset.shape[1], train_ratio=0.7)
        
        num_nodes = dataset.shape[1]
        blocksize = config['model']['blocksize']
        factors = config['model']['factors']
        blocknum = math.ceil(num_nodes / blocksize)
        blocknum_per_city[c] = blocknum
        
        sps = blocksize * factors
        ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, mxlen, adj, sps)
        all_parts_idx_info[c] = (ori_parts_idx, reo_parts_idx, reo_all_idx)
    
    blocknum_max = max(blocknum_per_city.values())
    logger.info(f"Maximum blocknum across cities: {blocknum_max}")
    
    # 配置元模型
    meta_config = OptimizedTASAConfig(
        seed=config['training']['seed'],
        data="meta",
        datapath=config['data']['datapath'],
        domain_specific_params=config['training']['domain_specific_params'],
        n_linear=config['model']['n_linear'],
        seq_len=seq_len,
        horizons=horizons,
        num_nodes=meta_num_nodes,
        node_dim=2,
        device=device,
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embd=config['model']['n_embd'],
        dropout=config['model']['dropout'],
        bias=config['model']['bias'],
        meta_lr=config['optimization']['meta_lr'],
        update_lr=config['optimization']['update_lr'],
        meta_epochs=config['optimization']['meta_epochs'],
        city_epochs=config['optimization']['city_epochs'],
        test_epochs=config['optimization']['test_epochs'],
        input_dim=config['model']['input_dim'],  
        tod_embedding_dim=config['model']['tod_embedding_dim'],
        dow_embedding_dim=config['model']['dow_embedding_dim'],
        spatial_embedding_dim=config['model']['spatial_embedding_dim'],
        adaptive_embedding_dim=config['model']['adaptive_embedding_dim'],
        steps_per_day=config['data']['steps_per_day'],
        output_dim=config['model']['output_dim'],
        temporal_layers=config['model']['temporal_layers'],
        spatial_layers=config['model']['spatial_layers'],
        blocksize=config['model']['blocksize'],
        blocknum=blocknum_max,
        factors=config['model']['factors'],
        use_lora=config['optimization']['use_lora'],
        lora_rank=config['optimization']['lora_rank'],
        use_flash_attn=config['optimization']['use_flash_attn'],
        gradient_checkpointing=config['optimization']['gradient_checkpointing']
    )
    
    meta_model = OptimizedTASA(meta_config).to(device)
    
    # 初始化城市模型
    model_dict = {}
    optim_dict = {}
    
    for c in train_cities + [target_city]:
        city_config = OptimizedTASAConfig(
            seed=config['training']['seed'],
            data=c,
            datapath=config['data']['datapath'],
            domain_specific_params=config['training']['domain_specific_params'],
            n_linear=config['model']['n_linear'],
            seq_len=seq_len,
            horizons=horizons,
            num_nodes=data_neural[c]['dataset'].shape[1],
            node_dim=2,
            device=device,
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            n_embd=config['model']['n_embd'],
            dropout=config['model']['dropout'],
            bias=config['model']['bias'],
            meta_lr=config['optimization']['meta_lr'],
            update_lr=config['optimization']['update_lr'],
            meta_epochs=config['optimization']['meta_epochs'],
            city_epochs=config['optimization']['city_epochs'],
            test_epochs=config['optimization']['test_epochs'],
            input_dim=config['model']['input_dim'],  
            tod_embedding_dim=config['model']['tod_embedding_dim'],
            dow_embedding_dim=config['model']['dow_embedding_dim'],
            spatial_embedding_dim=config['model']['spatial_embedding_dim'],
            adaptive_embedding_dim=config['model']['adaptive_embedding_dim'],
            steps_per_day=config['data']['steps_per_day'],
            output_dim=config['model']['output_dim'],
            temporal_layers=config['model']['temporal_layers'],
            spatial_layers=config['model']['spatial_layers'],
            blocksize=config['model']['blocksize'],
            blocknum=blocknum_per_city[c],
            factors=config['model']['factors'],
            use_lora=config['optimization']['use_lora'],
            lora_rank=config['optimization']['lora_rank'],
            use_flash_attn=config['optimization']['use_flash_attn'],
            gradient_checkpointing=config['optimization']['gradient_checkpointing']
        )
        
        model = OptimizedTASA(city_config).to(device)
        model_dict[c] = model
        optim_dict[c] = model.configure_optimizers(
            weight_decay=config['optimization']['weight_decay'],
            learning_rate=config['optimization']['update_lr'],
            betas=(0.9, 0.999),
            device_type='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    # 保存路径
    save_prefix = f"{target_city}-s{config['training']['seed']}"
    path_meta_model = os.path.join(out_dir, f'{save_prefix}_meta.pth')
    path_test_model = os.path.join(out_dir, f'{save_prefix}_best.pth')
    path_test_model_last = os.path.join(out_dir, f'{save_prefix}_last.pth')
    
    # 如果只是评估
    if config['training']['eval_only']:
        logger.info("Evaluation only mode")
        if os.path.exists(path_test_model):
            ckpt = torch.load(path_test_model, map_location=device)
            model_dict[target_city].load_state_dict(ckpt['model'])
            logger.info("Loaded best model for evaluation")
        else:
            logger.error(f"Model file not found: {path_test_model}")
            sys.exit(1)
        
        # 评估
        model_dict[target_city].eval()
        test_loader = test_loaders[f"{target_city}_eval"]
        
        with torch.no_grad():
            test_results = model_dict[target_city].evaluate(test_loader)
            logger.info("========== Test Results ==========")
            for h in horizons:
                logger.info(f"Horizon {h}: MAE={test_results[h]['MAE']:.3f}, "
                           f"MAPE={test_results[h]['MAPE']:.3f}%, "
                           f"MSE={test_results[h]['MSE']:.3f}, "
                           f"RMSE={test_results[h]['RMSE']:.3f}")
        
        return
    
    # 训练循环
    best_val_res = {h: {'MAE': float('inf'), 'MSE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf')} 
                    for h in horizons}
    
    meta_epochs = config['optimization']['meta_epochs']
    city_epochs = config['optimization']['city_epochs']
    test_epochs = config['optimization']['test_epochs']
    eval_interval = config['training']['eval_interval']
    grad_clip = config['optimization']['grad_clip']
    
    for m_epoch in range(meta_epochs):
        logger.info(f"========== Meta Epoch {m_epoch+1}/{meta_epochs} ==========")
        
        # 源城市训练
        for c in train_cities:
            meta_model.copy_invariant_params(model_dict[c])
            model_dict[c].train()
            
            tra_loader = train_loaders[c]
            ori_parts_idx, reo_parts_idx, reo_all_idx = all_parts_idx_info[c]
            
            for c_epoch in range(city_epochs):
                logger.info(f"MetaEpoch: {m_epoch+1}, City: {c}, CityEpoch: {c_epoch+1}/{city_epochs}")
                
                for batch_idx, (x_seq, y_seq) in enumerate(tra_loader):
                    x_seq = x_seq.to(device)
                    y_seq = {h: y_seq[h].to(device) for h in horizons}
                    
                    optim_dict[c].zero_grad(set_to_none=True)
                    
                    if ctx:
                        with ctx:
                            preds, loss = model_dict[c](x_seq, y_seq, train=True, stage='source',
                                                       ori_parts_idx=ori_parts_idx, 
                                                       reo_parts_idx=reo_parts_idx, 
                                                       reo_all_idx=reo_all_idx)
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optim_dict[c])
                            nn.utils.clip_grad_norm_(model_dict[c].parameters(), grad_clip)
                            scaler.step(optim_dict[c])
                            scaler.update()
                        else:
                            loss.backward()
                            nn.utils.clip_grad_norm_(model_dict[c].parameters(), grad_clip)
                            optim_dict[c].step()
                    else:
                        preds, loss = model_dict[c](x_seq, y_seq, train=True, stage='source',
                                                   ori_parts_idx=ori_parts_idx, 
                                                   reo_parts_idx=reo_parts_idx, 
                                                   reo_all_idx=reo_all_idx)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model_dict[c].parameters(), grad_clip)
                        optim_dict[c].step()
                    
                    if torch.isnan(loss):
                        logger.error("Loss is NaN. Stopping training.")
                        sys.exit(1)
            
            # 验证源城市
            if (m_epoch + 1) % eval_interval == 0:
                val_loader = val_loaders[c]
                model_dict[c].eval()
                with torch.no_grad():
                    v_preds = model_dict[c].evaluate(val_loader)
                    for h in horizons:
                        logger.info(f"MetaEpoch: {m_epoch+1}, City: {c}, Horizon: {h}, "
                                   f"Val MAE: {v_preds[h]['MAE']:.5f}, "
                                   f"Val MAPE: {v_preds[h]['MAPE']:.2f}%")
            
            # 更新元模型
            meta_model.eval()
            meta_params_dict = dict(meta_model.named_parameters())

            for name, param in model_dict[c].named_parameters():
                # 跳过领域特定参数
                if any(sub_str in name for sub_str in meta_model.config.domain_specific_params):
                    continue
                
                # 跳过与节点数相关的参数
                node_dependent_params = [
                    'embedding.adaptive_emb_B',
                    'embedding.node_to_cluster', 
                    'embedding.spatial_emb.weight',
                    'spatial_transformer.upsample_weight'
                ]
                if any(node_param in name for node_param in node_dependent_params):
                    continue
                
                # 检查参数是否存在于元模型中且有梯度
                if param.grad is not None and name in meta_params_dict:
                    meta_param = meta_params_dict[name]
                    
                    # 检查形状是否匹配
                    if meta_param.shape == param.shape:
                        meta_param.data -= config['optimization']['meta_lr'] * param.grad
                    else:
                        # 形状不匹配，跳过此参数
                        logger.debug(f"Skipping parameter {name} due to shape mismatch in meta update")

        # 目标城市适应
        meta_model.copy_invariant_params(model_dict[target_city])
        model_dict[target_city].train()
        tar_loader = train_loaders[target_city]
        ori_parts_idx, reo_parts_idx, reo_all_idx = all_parts_idx_info[target_city]
        
        for t_epoch in range(test_epochs):
            logger.info(f"MetaEpoch: {m_epoch+1}, Target City: {target_city}, TestEpoch: {t_epoch+1}/{test_epochs}")
            
            # 动态权重调整
            current_epoch = m_epoch * test_epochs + t_epoch
            total_epochs = meta_epochs * test_epochs
            target_weight = min(1.5, 1.0 + 0.5 * current_epoch / total_epochs)
            
            for batch_idx, (x_seq, y_seq) in enumerate(tar_loader):
                x_seq = x_seq.to(device)
                y_seq = {h: y_seq[h].to(device) for h in horizons}
                
                optim_dict[target_city].zero_grad(set_to_none=True)
                
                if ctx:
                    with ctx:
                        preds, loss = model_dict[target_city](x_seq, y_seq, train=True, stage='target',
                                                             ori_parts_idx=ori_parts_idx, 
                                                             reo_parts_idx=reo_parts_idx, 
                                                             reo_all_idx=reo_all_idx)
                        weighted_loss = target_weight * loss
                    
                    if scaler:
                        scaler.scale(weighted_loss).backward()
                        scaler.unscale_(optim_dict[target_city])
                        nn.utils.clip_grad_norm_(model_dict[target_city].parameters(), grad_clip)
                        scaler.step(optim_dict[target_city])
                        scaler.update()
                    else:
                        weighted_loss.backward()
                        nn.utils.clip_grad_norm_(model_dict[target_city].parameters(), grad_clip)
                        optim_dict[target_city].step()
                else:
                    preds, loss = model_dict[target_city](x_seq, y_seq, train=True, stage='target',
                                                          ori_parts_idx=ori_parts_idx, 
                                                          reo_parts_idx=reo_parts_idx, 
                                                          reo_all_idx=reo_all_idx)
                    weighted_loss = target_weight * loss
                    weighted_loss.backward()
                    nn.utils.clip_grad_norm_(model_dict[target_city].parameters(), grad_clip)
                    optim_dict[target_city].step()
            
            # 定期验证
            if (t_epoch + 1) % eval_interval == 0:
                val_loader = val_loaders[f"{target_city}_eval"]
                model_dict[target_city].eval()
                with torch.no_grad():
                    val_results = model_dict[target_city].evaluate(val_loader)
                    
                    for h in horizons:
                        logger.info(f"Test_epoch {t_epoch+1}, Horizon {h}: "
                                   f"Val MAE {val_results[h]['MAE']:.5f}, "
                                   f"Val MAPE {val_results[h]['MAPE']:.2f}%")
                        
                        # 保存最佳模型
                        if (val_results[h]['MAE'] < best_val_res[h]['MAE'] and
                            val_results[h]['RMSE'] < best_val_res[h]['RMSE']):
                            best_val_res[h] = val_results[h].copy()
                            
                            if m_epoch > 0 or t_epoch > 0:
                                ckpt = {
                                    'model': model_dict[target_city].state_dict(),
                                    'optimizer': optim_dict[target_city].state_dict(),
                                    'config': city_config.to_dict(),
                                    'm_epoch': m_epoch,
                                    't_epoch': t_epoch,
                                    'best_val_res': best_val_res
                                }
                                logger.info(f"Saving best checkpoint to {path_test_model}")
                                torch.save(ckpt, path_test_model)
    
    # 保存最终模型
    torch.save({'state_dict': meta_model.state_dict(), 'config': meta_config.to_dict()}, path_meta_model)
    
    ckpt_last = {
        'model': model_dict[target_city].state_dict(),
        'optimizer': optim_dict[target_city].state_dict(),
        'config': city_config.to_dict(),
        'm_epoch': meta_epochs - 1,
        't_epoch': test_epochs - 1,
        'best_val_res': best_val_res
    }
    torch.save(ckpt_last, path_test_model_last)
    
    # 最终评估
    logger.info("========== Final Evaluation ==========")
    
    if os.path.exists(path_test_model):
        ckpt = torch.load(path_test_model, map_location=device)
        model_dict[target_city].load_state_dict(ckpt['model'])
        logger.info("Loaded best model for final evaluation")
    
    model_dict[target_city].eval()
    
    # 验证集评估
    val_loader = val_loaders[f"{target_city}_eval"]
    with torch.no_grad():
        val_results = model_dict[target_city].evaluate(val_loader)
        logger.info("========== Validation Results (基于原始尺度) ==========")
        for h in horizons:
            logger.info(f"Horizon {h}: MAE={val_results[h]['MAE']:.3f}, "
                       f"MAPE={val_results[h]['MAPE']:.3f}%, "
                       f"MSE={val_results[h]['MSE']:.3f}, "
                       f"RMSE={val_results[h]['RMSE']:.3f}")
    
    # 测试集评估
    test_loader = test_loaders[f"{target_city}_eval"]
    
    if config['training']['enable_significance_test']:
        logger.info("========== Test Results with Statistical Significance (基于原始尺度) ==========")
        from utils.train_utils import evaluate_with_statistical_tests
        statistical_results = evaluate_with_statistical_tests(
            model_dict[target_city], 
            test_loader, 
            num_runs=config['training']['num_test_runs']
        )
        
        for h in horizons:
            logger.info(f"Horizon {h}:")
            for metric in ['MAE', 'MAPE', 'MSE', 'RMSE']:
                mean_val = statistical_results[h][metric]['mean']
                std_val = statistical_results[h][metric]['std']
                logger.info(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")
    else:
        with torch.no_grad():
            test_results = model_dict[target_city].evaluate(test_loader)
            logger.info("========== Test Results (基于原始尺度) ==========")
            for h in horizons:
                logger.info(f"Horizon {h}: MAE={test_results[h]['MAE']:.3f}, "
                           f"MAPE={test_results[h]['MAPE']:.3f}%, "
                           f"MSE={test_results[h]['MSE']:.3f}, "
                           f"RMSE={test_results[h]['RMSE']:.3f}")
    
    logger.info("Training completed successfully!")
    logger.info("🎯 所有评估指标都基于反归一化后的原始数据尺度")

if __name__ == "__main__":
    main()