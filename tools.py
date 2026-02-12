import torch
from modules.transformer import pytorchTransformerModule
from modules.performer import PerformerModule
from modules.gene_encoder import MaeAutobin


def select_module(config, sub_config, module_name):
    if module_name == 'performer':
        return PerformerModule(
            max_seq_len=config['seq_len'],  # 19266
            dim=sub_config['hidden_dim'],  # 512
            depth=sub_config['depth'],  # 6
            heads=sub_config['heads'],  # 8
            dim_head=sub_config['dim_head'],  # 64
            ff_dropout=sub_config.get('ff_dropout',0.0),  # 0.0
            attn_dropout=sub_config.get('attn_dropout',0.0)  # 0.0
        )
    elif module_name == 'transformer':
        return pytorchTransformerModule(
            max_seq_len=config['seq_len'],  # 19266
            dim=sub_config['hidden_dim'],  # 768
            depth=sub_config['depth'],  # 12
            heads=sub_config['heads']  # 12
        )
    else:
        print('module type error')
        exit(0)


def select_model(config):
    if config["model"] == "mae_autobin":
        encoder_config =config['encoder']
        decoder_config = config['decoder']
        encoder = select_module(config, encoder_config, config['encoder']['module_type'])
        decoder = select_module(config, decoder_config, config['decoder']['module_type'])
        model = MaeAutobin(
            num_tokens=config['n_class'],  # 104
            max_seq_len=config['seq_len'],  # 19266
            embed_dim=config['encoder']['hidden_dim'],  # 768
            decoder_embed_dim=config['decoder']['hidden_dim'],  # 512
            bin_alpha = config['bin_alpha'],  # 1.0
            bin_num = config['bin_num'],  # 100
            pad_token_id = config['pad_token_id'],  # 103
            mask_token_id = config['mask_token_id'],  # 102
        )
        model.encoder = encoder
        model.decoder = decoder
    else:
        raise NotImplementedError("Unknown model type!")
    return model


def convertconfig(ckpt):
    newconfig = {}
    newconfig['config'] = {}
    model_type = ckpt['config']['model']

    for key, val in ckpt['config']['model_config'][model_type].items():
        newconfig['config'][key] = val

    for key, val in ckpt['config']['dataset_config']['rnaseq'].items():
        newconfig['config'][key] = val

    if model_type == 'performergau_resolution':
        model_type = 'performer_gau'

    import collections
    d = collections.OrderedDict()
    for key, val in ckpt['state_dict'].items():
        d[str(key).split('model.')[1]] = val

    newconfig['config']['model_type'] = model_type
    newconfig['model_state_dict'] = d
    newconfig['config']['pos_embed'] = False
    newconfig['config']['device'] = 'cuda'
    return newconfig


def load_model_frommmf(best_ckpt_path, key='gene'):
    model_data = torch.load(best_ckpt_path,map_location='cpu')
    model_data = model_data[key]
    model_data = convertconfig(model_data)
    if not model_data.__contains__('config'):
        print('***** No config *****')
        config={}
        config['model_type']='flash_all'
    else:
        config=model_data['config']
        print(config)
    if not config.__contains__('qv_dim'):
        if config['model'] != 'mae_autobin':
            if config.__contains__('dim_head'):
                config['qv_dim']=config['dim_head']
            else:
                print('***** No qv_dim ***** set 64')
                config['qv_dim']= 64
    if not config.__contains__('ppi_edge'):
        config['ppi_edge']=None
    model = select_model(config)
    model_state_dict = model_data['model_state_dict']
    model.load_state_dict(model_state_dict)
    return model, config


def gatherData(data, labels, pad_token_id):
    value_nums = labels.sum(1)
    max_num = max(value_nums)

    fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                           device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1,
                            device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data

    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels


def mask_gene_expression(gene_expression_tensor, encoder_data_padding, mask_value=102, mask_prob=0.15):
    random_mask = torch.rand(gene_expression_tensor.size()) < mask_prob
    random_mask[encoder_data_padding] = False
    mask_tensor = random_mask.int()
    masked_tensor = gene_expression_tensor * (1 - mask_tensor) + mask_value * mask_tensor
    label_tensor = gene_expression_tensor.clone()
    label_tensor[~random_mask] = -100
    return masked_tensor, label_tensor


'''
encoder_data: no zero gene expression values, pad is 103
encoder_position_gene_ids: no zero gene expression values corresponding gene index, pad is seq_len
encoder_data_padding: corresponding value in encoder data is pad or not
encoder_data_labels: original gene expression no zero label (no zero is labeled as True)
decoder_data: original gene expression
decoder_data_padding: the size is the same with original gene expression, all is False
new_data_raw: original gene expression
data_mask_labels: None
decoder_position_gene_ids: each row is 0 to seq_len
'''
def getEncoerDecoderData(data, data_raw, pad_token_id=103, mask_token_id=102, cell_token_id=101, seq_len=5001, need_mask=True):
    cell_token = torch.FloatTensor([cell_token_id]).repeat(data.shape[0], 1)
    data = torch.cat((cell_token, data), dim=-1)
    data_raw = torch.cat((cell_token, data_raw), dim=-1)
    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(data.device)

    encoder_data_labels = data_raw > 0
    encoder_data, encoder_data_padding = gatherData(decoder_data, encoder_data_labels, pad_token_id)
    encoder_data_ori = encoder_data
    if need_mask:
        encoder_data, encoder_mask_label = mask_gene_expression(encoder_data, encoder_data_padding, mask_token_id)
    # encoder_data, encoder_mask_label = mask_gene_expression(encoder_data, encoder_data_padding, mask_token_id)
    new_data_raw = data_raw
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = gatherData(data_gene_ids, encoder_data_labels, pad_token_id)
    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = seq_len
    decoder_position_gene_ids[decoder_data_padding] = seq_len
    if need_mask:
        decoder_data, decoder_mask_label = mask_gene_expression(decoder_data, decoder_data_padding, mask_token_id,
                                                                mask_prob=0.05)
        batch_idx, gen_idx = (encoder_data_labels == True).nonzero(as_tuple=True)
        decoder_mask_label[batch_idx, gen_idx] = encoder_mask_label[~encoder_data_padding].to(decoder_data.dtype)
        decoder_data[:, 0] = cell_token_id
    else:
        decoder_mask_label = decoder_data

    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, \
           decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids, decoder_mask_label, encoder_data_ori, data_raw
