# Transfer Needs More: Cross-city Traffic Prediction with Semantic-Topological Decoupling and Spatial Attention Enhancement

**ICDM 2025 Submission - Official Implementation**

## Installation

### Requirements
```bash
Python >= 3.9
PyTorch >= 2.1.0
CUDA >= 12.1 (recommended for GPU acceleration)
```

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install pyyaml scipy matplotlib seaborn
```

## Project Structure

```
├── models/
│   ├── __init__.py                    # Model module exports
│   ├── optimized_tasa.py             # Main TASA model implementation
│   │   ├── OptimizedTASAConfig       # Model configuration class
│   │   ├── OptimizedSTAE             # Spatio-temporal adaptive embedding
│   │   ├── CityAdaptationMechanism   # LoRA-based city adaptation
│   │   └── OptimizedTASA             # Main model class
│   ├── semi_self_attention.py        # Semi-self attention mechanisms
│   │   ├── LoRAAdapter               # Low-rank adaptation layer
│   │   ├── OptimizedSemiSelfAttention # Enhanced attention mechanism
│   │   └── OptimizedSemiSelfAttentionBlock # Complete attention block
│   └── spatial_transformer.py        # Spatial attention components
│       ├── EfficientDualAttBlock     # Dual attention for spatial modeling
│       └── LinearAttention           # Linear attention implementation
├── utils/
│   ├── __init__.py                   # Utility module exports
│   ├── data_utils.py                 # Data processing utilities
│   │   ├── read_meta_datasets()      # Multi-city data loading
│   │   ├── generate_data()           # Sequence generation with normalization
│   │   ├── construct_adj_safe()      # Leak-free adjacency matrix construction
│   │   ├── strict_temporal_split()   # Temporal data splitting
│   │   └── DataLoaderWithScaler      # Data loader with normalization info
│   ├── train_utils.py                # Training utilities
│   │   ├── set_random_seed()         # Reproducibility control
│   │   ├── set_logger()              # Logging configuration
│   │   └── evaluate_with_statistical_tests() # Statistical significance testing
│   └── logger_utils.py               # Advanced logging utilities
├── config/                           # Configuration files
│   ├── chengdu_config.yaml           # Chengdu city configuration
│   ├── metr-la_config.yaml           # METR-LA dataset configuration
│   ├── pems-bay_config.yaml          # PEMS-BAY dataset configuration
│   └── shenzhen_config.yaml          # Shenzhen city configuration
└── main.py                          # Main training and evaluation script
```

## Configuration Files

### Data Configuration
```yaml
data:
  datapath: "/root/ggg/DATA/data"          # Root path to datasets
  train_cities: ["metr-la", "pems-bay", "shenzhen"]  # Source cities for meta-learning
  target_city: "chengdu"                   # Target city for adaptation
  seq_len: 12                             # Input sequence length
  horizons: [3, 6, 12]                    # Prediction horizons (time steps)
  steps_per_day: 288                      # Time steps per day (5-min intervals)
  adaptation_days: 3                      # Days of target city data for adaptation
```

### Model Architecture Configuration
```yaml
model:
  # Transformer Architecture
  n_layer: 6                              # Number of transformer layers
  n_head: 4                               # Number of attention heads per layer
  n_embd: 64                              # Hidden embedding dimension
  dropout: 0.1                            # Dropout rate
  bias: false                             # Whether to use bias in linear layers
  n_linear: 1                             # Number of linear layers in MLP
  
  # Input/Output Dimensions
  input_dim: 2                            # Input feature dimensions (traffic + metadata)
  output_dim: 1                           # Output dimension (traffic speed prediction)
  
  # Temporal Embeddings
  tod_embedding_dim: 8                   # Time-of-day embedding dimension
  dow_embedding_dim: 4                    # Day-of-week embedding dimension
  
  # Spatial Embeddings
  spatial_embedding_dim: 8                # Node spatial embedding dimension
  adaptive_embedding_dim: 8               # Adaptive embedding dimension
  
  # Layer Configuration
  temporal_layers: 1                      # Number of temporal attention layers
  spatial_layers: 1                       # Number of spatial attention layers
  
  # Spatial Partitioning
  blocksize: 8                            # Nodes per spatial block
  blocknum: 4                             # Number of spatial blocks
  factors: 1                              # Spatial partitioning factor
```

### Optimization Configuration
```yaml
optimization:
  # Learning Rates
  meta_lr: 0.00005                        # Meta-learning rate for shared parameters
  update_lr: 0.001                        # Adaptation learning rate for city-specific parameters
  
  # Training Epochs
  meta_epochs: 5                          # Number of meta-training epochs
  city_epochs: 1                          # Epochs per source city in meta-training
  test_epochs: 50                         # Target city adaptation epochs
  
  # Optimization Settings
  batch_size: 32                          # Training batch size
  grad_clip: 1.0                          # Gradient clipping threshold
  weight_decay: 0.001                     # L2 regularization coefficient
  
  # Efficiency Optimizations
  use_lora: true                          # Enable LoRA (Low-Rank Adaptation)
  lora_rank: 32                           # LoRA decomposition rank
  use_flash_attn: false                   # Enable Flash Attention (experimental)
  gradient_checkpointing: false           # Enable gradient checkpointing for memory
  mixed_precision: true                   # Enable mixed precision training
  dtype: "float32"                        # Training data type
```

### Training Configuration
```yaml
training:
  seed: 42                                # Random seed for reproducibility
  cuda: "2"                               # GPU device ID
  eval_interval: 2                        # Evaluation frequency (epochs)
  eval_only: false                        # Whether to only run evaluation
  enable_significance_test: true          # Enable statistical significance testing
  num_test_runs: 5                        # Number of runs for significance testing
  domain_specific_params: ["value_mlp", "value_transform"]  # City-specific parameters
```

### Output Configuration
```yaml
output:
  out_dir: "out/improved_main"            # Output directory for checkpoints
  log_dir: "logs"                         # Directory for log files
  save_best: true                         # Save best validation model
  save_last: true                         # Save final model state
```

## Dataset Structure

Expected dataset organization:
```
DATA/data/
├── metr-la/
│   ├── dataset.npy          # Shape: (T, N, C) - Time, Nodes, Features
│   ├── matrix.npy           # Shape: (N, N) - Adjacency matrix
│   └── coords.npy           # Shape: (N, 2) - Node coordinates
├── pems-bay/
│   ├── dataset.npy          # Shape: (T, N, C)
│   ├── matrix.npy           # Shape: (N, N)
│   └── coords.npy           # Shape: (N, 2)
├── chengdu/
│   ├── dataset.npy          # Shape: (T, N, C)
│   ├── matrix.npy           # Shape: (N, N)
│   └── coords.npy           # Shape: (N, 2)
└── shenzhen/
    ├── dataset.npy          # Shape: (T, N, C)
    ├── matrix.npy           # Shape: (N, N)
    └── coords.npy           # Shape: (N, 2)
```

### Data Format Details
- **dataset.npy**: Traffic data with shape `(T, N, C)` where:
  - `T`: Number of time steps
  - `N`: Number of traffic sensors/nodes
  - `C`: Number of features (typically 2: speed + occupancy)
- **matrix.npy**: Adjacency matrix with shape `(N, N)` representing spatial relationships
- **coords.npy**: Node coordinates with shape `(N, 2)` for geographic positioning

## Usage

### Basic Training
```bash
# Train on Chengdu as target city
python main.py --config config/chengdu_config.yaml

# Train on METR-LA as target city
python main.py --config config/metr-la_config.yaml

# Train on PEMS-BAY as target city
python main.py --config config/pems-bay_config.yaml

# Train on Shenzhen as target city
python main.py --config config/shenzhen_config.yaml
```



### Custom Configuration
```bash
# Override configuration parameters
python main.py --config config/chengdu_config.yaml \
    --seed 42 \
    --cuda 0 \
    --seq_len 24 \
    --horizons 3，6，12
```

## Key Model Components

### 1. OptimizedTASA (Main Model)
- **Semantic-Topological Decoupling**: Separates shared traffic semantics from city-specific topology
- **Meta-Learning Framework**: Learns transferable patterns across multiple source cities
- **Parameter Separation**: Distinguishes between shared, private, and domain-specific parameters

### 2. Spatio-Temporal Adaptive Embedding (STAE)
- **Multi-Modal Embedding**: Processes traffic data, temporal patterns, and spatial relationships
- **Temporal Features**: Automatic generation of time-of-day and day-of-week embeddings
- **Spatial Features**: Node-specific and adaptive spatial embeddings

### 3. Semi-Self Attention Mechanism
- **Parameter Separation**: Uses shared parameters for Q,K and private parameters for V
- **LoRA Adaptation**: Efficient low-rank adaptation for city-specific fine-tuning
- **Attention Variants**: Standard, Flash, and Local attention implementations

### 4. Spatial Transformer
- **Dual Attention**: Intra-block and inter-block attention for hierarchical spatial modeling
- **Block Partitioning**: KD-tree based spatial partitioning for scalability
- **Efficient Implementation**: LoRA-enhanced MLPs for reduced parameter overhead

## Data Processing Pipeline

### 1. Temporal Splitting Strategy
```python
# Strict chronological splitting to prevent data leakage
def strict_temporal_split(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # Ensures no future information leaks into training
```

### 2. Normalization Strategy
```python
# Z-score normalization using only training data statistics
def generate_data(dataset, seq_len, horizons, split_ratios):
    # Fit scaler only on training data
    # Apply same normalization to validation and test sets
```

### 3. Adjacency Matrix Construction
```python
# Safe adjacency matrix construction to avoid data leakage
def construct_adj_safe(data, num_node, train_ratio=0.7):
    # Use only training portion of data for correlation computation
```

## Training Strategy

### Meta-Learning Process
1. **Source City Training**: Learn shared traffic patterns from multiple source cities
2. **Meta-Parameter Update**: Update shared parameters using gradients from all source cities
3. **Target City Adaptation**: Fine-tune on limited target city data
4. **Dynamic Weighting**: Gradually increase target city loss weight during adaptation

### Optimization Features
- **Gradient Clipping**: Prevents gradient explosion during meta-learning
- **Mixed Precision**: Accelerates training while maintaining numerical stability
- **Statistical Testing**: Multiple runs with significance testing for robust evaluation

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 24GB memory, 24GB system RAM
- **Recommended**: NVIDIA RTX 4090/V100 or better, 32GB system RAM
- **Storage**: 50GB for datasets and model checkpoints
