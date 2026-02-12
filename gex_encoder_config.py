
seq_len = 5001
gex_encoder_config = {'mask_gene_name': False, 'gene_num': seq_len, 'seq_len': seq_len,

          'encoder': {'hidden_dim': 64, 'depth': 6, 'heads': 8, 'dim_head': 8,
                      'seq_len': seq_len, 'module_type': 'transformer', 'norm_first': False},

          'decoder': {'hidden_dim': 64, 'depth': 3, 'heads': 8, 'dim_head': 8,
                      'module_type': 'performer', 'seq_len': seq_len, 'norm_first': False},

          'n_class': 104, 'pad_token_id': 103, 'mask_token_id': 102, 'cell_token_id': 101,
          'bin_num': 100, 'bin_alpha': 1.0, 'rawcount': True, 'model': 'mae_autobin',
          'num_tokens': 13, 'train_data_path': None, 'isPanA': False, 'isPlanA1': False, 'max_files_to_load': 5,
          'bin_type': 'auto_bin', 'value_mask_prob': 0.3, 'zero_mask_prob': 0.03, 'replace_prob': 0.8,
          'random_token_prob': 0.1, 'mask_ignore_token_ids': [0], 'decoder_add_zero': True,
          'mae_encoder_max_seq_len': seq_len,
          'isPlanA': False, 'mask_prob': 0.3, 'model_type': 'mae_autobin', 'pos_embed': False, 'device': 'cuda'}
