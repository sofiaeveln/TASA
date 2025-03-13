# TASA
The framework of TASA
# Exrimental Sets


For PEMS-BAY:
seed=0, cuda='0', dtype='float32', method='Cross-city-Traffic-Transformer', train_cities=['shenzhen', 'metr-la', 'chengdu'], data='pems-bay', spec_type='sharemlp', domain_specific_params=['value_mlp', 'value_transform'], datapath='/root/mycode/data', out_dir='out/main/ln1', seq_len=12, horizons=[3], load_type='best', meta_lr=5e-05, update_lr=0.001, meta_epochs=5, city_epochs=1, test_epochs=50, batch_size=32, n_head=4, n_layer=6, n_linear=1, n_embd=64, dropout=0.1, bias=False, grad_clip=1.0, eval_only=False, eval_interval=2, input_dim=2, tod_embedding_dim=8, dow_embedding_dim=4, spatial_embedding_dim=8, adaptive_embedding_dim=8, steps_per_day=288, output_dim=1, temporal_layers=1, spatial_layers=1

For Chengdu:
seed=0, cuda='0', dtype='float32', method='Cross-city-Traffic-Transformer', train_cities=['metr-la', 'pems-bay', 'shenzhen'], data='chengdu', spec_type='sharemlp', domain_specific_params=['value_mlp', 'value_transform'], datapath='/root/data', out_dir='out/main/ln1', seq_len=12, horizons=[1], load_type='best', meta_lr=5e-05, update_lr=0.001, meta_epochs=5, city_epochs=1, test_epochs=50, batch_size=32, n_head=4, n_layer=6, n_linear=1, n_embd=64, dropout=0.1, bias=False, grad_clip=1.0, eval_only=False, eval_interval=2, input_dim=2, tod_embedding_dim=8, dow_embedding_dim=4, spatial_embedding_dim=24, adaptive_embedding_dim=8, steps_per_day=288, output_dim=1, temporal_layers=2, spatial_layers=2

For METR-LA:
seed=0, cuda='0', dtype='float32', method='Cross-city-Traffic-Transformer', train_cities=['chengdu', 'pems-bay', 'shenzhen'], data='metr-la', spec_type='sharemlp', domain_specific_params=['value_mlp', 'value_transform'], datapath='/root/mycode/data', out_dir='out/main/ln1', seq_len=12, horizons=[3], load_type='best', meta_lr=5e-05, update_lr=0.001, meta_epochs=5, city_epochs=1, test_epochs=50, batch_size=32, n_head=4, n_layer=6, n_linear=1, n_embd=64, dropout=0.1, bias=False, grad_clip=1.0, eval_only=False, eval_interval=2, input_dim=2, tod_embedding_dim=8, dow_embedding_dim=4, spatial_embedding_dim=8, adaptive_embedding_dim=8, steps_per_day=288, output_dim=1, temporal_layers=2, spatial_layers=2

For shenzhen:
seed=0, cuda='0', dtype='float32', method='Cross-city-Traffic-Transformer', train_cities=['metr-la', 'pems-bay', 'shenzhen'], data='chengdu', spec_type='sharemlp', domain_specific_params=['value_mlp', 'value_transform'], datapath='/root/data', out_dir='out/main/ln1', seq_len=12, horizons=[1], load_type='best', meta_lr=5e-05, update_lr=0.001, meta_epochs=5, city_epochs=1, test_epochs=50, batch_size=32, n_head=4, n_layer=6, n_linear=1, n_embd=64, dropout=0.1, bias=False, grad_clip=1.0, eval_only=False, eval_interval=2, input_dim=2, tod_embedding_dim=8, dow_embedding_dim=4, spatial_embedding_dim=64, adaptive_embedding_dim=128, steps_per_day=288, output_dim=1, temporal_layers=2, spatial_layers=2
