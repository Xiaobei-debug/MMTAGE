import torch
import random


def replace_masked_tokens(token_ids, candidate_pred_positions, num_mlm_preds, all_mlm_id, masked_token_rate=1,
                          context_length=25):
    pred_positions = []

    mlm_input_tokens_id = [token_id for token_id in token_ids]
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions) >= num_mlm_preds:
            break
        masked_token_id = None
        if random.random() < masked_token_rate:  # 0.8
            masked_token_id = 22
        mlm_input_tokens_id[mlm_pred_position] = masked_token_id
        pred_positions.append(mlm_pred_position)

    words_tokens = []
    for word in mlm_input_tokens_id:
        word_token = word
        words_tokens.append(word_token)

    all_tokens = words_tokens
    mlm_label = [-100 if idx not in pred_positions else all_mlm_id[idx] for idx in range(len(token_ids))]
    flattened_mlm_label = []
    for item in mlm_label:
        if isinstance(item, list):
            flattened_mlm_label.extend(item)
        else:
            flattened_mlm_label.append(item)
    mlm_label = flattened_mlm_label
    results = torch.zeros(context_length, dtype=torch.long)
    results_labels = -100 * torch.ones(context_length, dtype=torch.long)
    # pdb.set_trace()

    if len(all_tokens) > context_length:
        raise RuntimeError(f"Input is too long for context length {context_length}")
    results[:len(all_tokens)] = torch.tensor(all_tokens)
    results_labels[:len(all_tokens)] = torch.tensor(mlm_label)
    # pdb.set_trace()
    # all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in [' '.join(mlm_input_tokens_id)]]

    return results.tolist(), results_labels.tolist()


def get_masked_sample1(text, mlm_label_dict, masked_rate=0.15):
    candidate_pred_positions = []
    # pdb.set_trace()

    all_mlm_id = [mlm_label_dict.get(word.lower(), 0) for word in text.split()]
    # pdb.set_trace()
    text_list = list(text.lower().split())
    for i, ids in enumerate(text_list):
        candidate_pred_positions.append(i)
    random.shuffle(candidate_pred_positions)
    num_mlm_preds = max(1, round(len(candidate_pred_positions) * masked_rate))
    # pdb.set_trace()
    mlm_input_tokens_id, mlm_label = replace_masked_tokens(
        text_list, candidate_pred_positions, num_mlm_preds, all_mlm_id)
    return mlm_input_tokens_id, mlm_label


def get_masked_sample(tcr, masked_rate=0.15):
    candidate_pred_positions = []
    # pdb.set_trace()
    for i, ids in enumerate(tcr):
        candidate_pred_positions.append(i)
    random.shuffle(candidate_pred_positions)
    num_mlm_preds = max(1, round(len(candidate_pred_positions) * masked_rate))
    mlm_input_tokens_id, mlm_label = replace_masked_tokens(tcr, candidate_pred_positions, num_mlm_preds, tcr)
    return mlm_input_tokens_id, mlm_label


# mlm_input_tokens_id, mlm_label = get_masked_sample([0, 11, 12, 5, 6, 7, 10, 16, 2, 3, 4, 11, 5])
# print(mlm_input_tokens_id)
# print(mlm_label)
