import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os

def read_meta_datasets(train_cities=[], test_city='', path='',
                       tod_embedding_dim=24, dow_embedding_dim=8, steps_per_day=288):
    """
    🔧 修正：保持原有input_dim=2的成功配置，避免数据泄露
    """
    data_neural = {c: {} for c in train_cities + [test_city]}
    num_nodes = 0
    
    for c in train_cities + [test_city]:
        dataset = np.load(f'{path}/{c}/dataset.npy')  # (T, N, C)
        adj_mat = np.load(f'{path}/{c}/matrix.npy') if os.path.exists(f'{path}/{c}/matrix.npy') else None

        T, N, C = dataset.shape
        time_idx = np.arange(T)

        # 🔧 关键修正：恢复使用前2个特征，保持原有精度
        if C >= 2:
            traffic_data = dataset[:, :, :2]  # 保持前2个特征
        else:
            traffic_data = dataset[:, :, :1]  # 如果只有1个特征
        
        dataset_processed = traffic_data

        new_features = []
        # TOD特征
        if tod_embedding_dim > 0:
            tod_feature = ((time_idx % steps_per_day) / steps_per_day).reshape(T, 1)
            tod_feature = np.tile(tod_feature, (1, N))[:, :, None]
            new_features.append(tod_feature)

        # DOW特征
        if dow_embedding_dim > 0:
            dow_feature = ((time_idx // steps_per_day) % 7).reshape(T, 1)
            dow_feature = np.tile(dow_feature, (1, N))[:, :, None]
            new_features.append(dow_feature)

        if len(new_features) > 0:
            new_features_all = np.concatenate(new_features, axis=-1)
            dataset_processed = np.concatenate([dataset_processed, new_features_all], axis=-1)

        data_neural[c]['dataset'] = dataset_processed
        data_neural[c]['adj_mat'] = adj_mat
        num_nodes = max(num_nodes, dataset_processed.shape[1])
    
    return data_neural, num_nodes

def construct_adj_safe(data, num_node, train_ratio=0.7):
    """
    🔧 新增：安全的邻接矩阵构建，避免数据泄露
    """
    # 只使用训练集比例的数据构建邻接矩阵
    total_samples = data.shape[0]
    train_end = int(total_samples * train_ratio)
    train_data = data[:train_end]
    
    # 只使用第一个特征（交通速度）
    if len(train_data.shape) == 3:
        train_traffic = train_data[:, :, 0]  # (T, N)
    else:
        train_traffic = train_data  # (T, N)
    
    # 计算相似度矩阵
    train_traffic_normalized = (train_traffic - train_traffic.mean(axis=0, keepdims=True)) / (train_traffic.std(axis=0, keepdims=True) + 1e-8)
    correlation_matrix = np.corrcoef(train_traffic_normalized.T)
    
    # 处理NaN值
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    # 转换为邻接矩阵
    threshold = 0.5
    adj_matrix = (correlation_matrix > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)  # 对角线设为0
    
    return adj_matrix

def strict_temporal_split(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    🔧 新增：严格按时间顺序分割数据，避免数据泄露
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    total_samples = dataset.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    return {
        'train': dataset[:train_end],
        'val': dataset[train_end:val_end],
        'test': dataset[val_end:]
    }

def generate_data(dataset, seq_len, horizons, split_ratios=(0.7, 0.1, 0.2), scaler=None):
    """
    生成多时间步的预测数据，修正标准化流程避免数据泄露
    🔧 修正：严格按时间顺序分割
    """
    # 🔧 修正：先按时间顺序分割原始数据
    temporal_splits = strict_temporal_split(dataset, *split_ratios)
    
    final_splits = {}
    
    for split_name, split_data in temporal_splits.items():
        if len(split_data) == 0:
            final_splits[split_name] = {
                'x': np.array([]), 
                'y': {h: np.array([]) for h in horizons}, 
                'scaler': None
            }
            continue
            
        x, ys = [], {h: [] for h in horizons}
        max_horizon = max(horizons)
        T = len(split_data)
        
        # 生成序列数据
        for i in range(T - seq_len - max_horizon):
            a = split_data[i: i + seq_len, :, :]
            skip = False
            for h in horizons:
                # 只使用第一个特征作为预测目标
                b = split_data[i + seq_len: i + seq_len + h, :, 0]
                if np.isnan(a).any() or np.isnan(b).any() or np.isinf(a).any() or np.isinf(b).any():
                    skip = True
                    break
                ys[h].append(b)
            if not skip:
                x.append(a)

        if not x:
            final_splits[split_name] = {
                'x': np.array([]), 
                'y': {h: np.array([]) for h in horizons}, 
                'scaler': None
            }
            continue

        x = np.stack(x, axis=0)
        for h in horizons:
            ys[h] = np.stack(ys[h], axis=0)
        
        final_splits[split_name] = {'x': x, 'y': ys, 'scaler': None}
    
    # 🔧 修正：只用训练集数据训练scaler
    if 'train' in final_splits and len(final_splits['train']['x']) > 0:
        new_scaler = StandardScaler()
        train_traffic_data = final_splits['train']['x'][:, :, :, 0:1]  # 只取第一个特征
        train_traffic_2d = train_traffic_data.reshape(-1, 1)
        new_scaler.fit(train_traffic_2d)
        
        # 用训练集统计信息标准化所有分割的数据
        for split_name in final_splits:
            if len(final_splits[split_name]['x']) > 0:
                # 标准化输入数据的第一个特征
                x_data = final_splits[split_name]['x']
                traffic_features = x_data[:, :, :, 0:1]
                traffic_2d = traffic_features.reshape(-1, 1)
                traffic_normalized = new_scaler.transform(traffic_2d)
                traffic_normalized = np.clip(traffic_normalized, -3, 3)
                x_data[:, :, :, 0:1] = traffic_normalized.reshape(traffic_features.shape)
                final_splits[split_name]['x'] = x_data
                
                # 标准化目标数据
                for h in horizons:
                    if len(final_splits[split_name]['y'][h]) > 0:
                        y_data = final_splits[split_name]['y'][h]
                        y_2d = y_data.reshape(-1, 1)
                        y_norm = new_scaler.transform(y_2d)
                        y_norm = np.clip(y_norm, -3, 3)
                        final_splits[split_name]['y'][h] = y_norm.reshape(y_data.shape)
                
                final_splits[split_name]['scaler'] = new_scaler
    
    return final_splits

def construct_adj_from_coords(coords_path, num_nodes, threshold=0.1):
    """
    🔧 新增：基于地理坐标构建邻接矩阵，避免数据泄露
    """
    if os.path.exists(coords_path):
        coords = np.load(coords_path)  # (N, 2) 或 (2, N)
        if coords.shape[0] == 2:
            coords = coords.T  # 转换为 (N, 2)
        
        # 计算节点间的欧式距离
        distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        
        # 基于距离阈值构建邻接矩阵
        adj_matrix = (distances < threshold).astype(float)
        
        # 设置对角线为0（节点不与自己连接）
        np.fill_diagonal(adj_matrix, 0)
        
        return adj_matrix
    else:
        # 如果没有坐标文件，返回预定义的邻接矩阵（例如格状图）
        adj_matrix = np.eye(num_nodes)
        # 添加相邻节点连接（简单的链式连接）
        for i in range(num_nodes - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1
        return adj_matrix

def construct_adj(data, num_node):
    """
    🔧 修正：提供备用方法，但推荐使用基于坐标的方法
    注意：此方法仍可能造成数据泄露，仅作为fallback
    """
    print("警告：使用基于时间序列的邻接矩阵构建可能造成数据泄露，建议使用construct_adj_from_coords")
    
    steps_per_day = 288
    days_count = data.shape[0] // steps_per_day
    if days_count == 0:
        data_mean = data.mean(axis=0)
    else:
        data_subset = data[:days_count * steps_per_day]
        data_mean = data_subset.mean(axis=0)
    
    # 将1D数组转换为2D数组以适配cosine_similarity
    data_for_similarity = data_subset.T if days_count > 0 else data.T
    
    tem_matrix = cosine_similarity(data_for_similarity, data_for_similarity)
    tem_matrix = np.exp((tem_matrix - tem_matrix.mean()) / tem_matrix.std())
    return tem_matrix

# 其他函数保持不变
def augmentAlign(dist_matrix, auglen):
    """根据余弦相似度对分块进行填充"""
    sorted_idx = np.argsort(dist_matrix.reshape(-1) * -1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)

def reorderData(parts_idx, mxlen, adj, sps):
    """对分块后的数据进行重排序和填充"""
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if sps - part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0]) + sps * i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    return ori_parts_idx, reo_parts_idx, reo_all_idx

def kdTree(locations, times, axis):
    """KDTree分块算法"""
    sorted_idx = np.argsort(locations[axis])
    half = locations.shape[1] // 2
    part1, part2 = np.sort(sorted_idx[:half]), np.sort(sorted_idx[half:])
    parts = []
    
    if times == 1:
        return [part1, part2], max(part1.shape[0], part2.shape[0])
    else:
        left_parts, lmxlen = kdTree(locations[:, part1], times - 1, axis ^ 1)
        right_parts, rmxlen = kdTree(locations[:, part2], times - 1, axis ^ 1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    
    return parts, max(lmxlen, rmxlen)

def compute_node_clusters(num_nodes, num_clusters=None):
    """计算节点聚类"""
    if num_clusters is None:
        num_clusters = min(num_nodes // 10, 100)
    
    nodes_per_cluster = num_nodes // num_clusters
    node_to_cluster = torch.zeros(num_nodes, dtype=torch.long)
    
    for i in range(num_nodes):
        node_to_cluster[i] = min(i // nodes_per_cluster, num_clusters - 1)
    
    return node_to_cluster

class DataLoaderWithScaler:
    """带有标准化器的数据加载器"""
    def __init__(self, data_batches, scaler=None):
        self.data_batches = data_batches
        self.scaler = scaler
    
    def __iter__(self):
        return iter(self.data_batches)
    
    def __len__(self):
        return len(self.data_batches)

def get_dataloader(splits, batch_size, split='train', shuffle=True, drop_last=False, horizons=[12]):
    """根据数据集划分生成数据加载器，包含scaler信息"""
    split_data = splits[split]
    x, y_tests, scaler = split_data['x'], split_data['y'], split_data.get('scaler', None)
    data_len = x.shape[0]
    num_batches = data_len // batch_size
    if not drop_last:
        num_batches += int((data_len % batch_size) != 0)
    
    data_batches = []
    indices = list(range(data_len))
    if shuffle and split == 'train':
        np.random.shuffle(indices)
    
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)
        batch_index = indices[start_index:end_index]
        batch_x = torch.from_numpy(x[batch_index]).float()
        batch_y = {h: torch.from_numpy(y_tests[h][batch_index]).float() for h in horizons}
        data_batches.append((batch_x, batch_y))
    
    data_loader = DataLoaderWithScaler(data_batches, scaler)
    
    return data_loader