import torch
from torch.utils.data import Dataset
import scanpy as sc
from utils import get_masked_sample
import numpy as np


alphabets = {
    "PAD": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "*": 21,
    "MASK": 22,
    "CLS": 23,
    "BOS": 24,
}
all_batch_num = {"GSE114724": 0, "GSE139555": 1, "GSE181061": 2, "GSE145370": 3, "GSE148190": 4,
                 "GSE211504": 5, "GSE242477": 6, "GSE162500": 7, "GSE168844": 8, "GSE215219": 9,
                 "GSE180268": 10, "GSE200996": 11, "GSE195486": 12}


class pretrain_dataset(Dataset):
    def __init__(self, gex_file):
        data = sc.read_h5ad(gex_file)
        self.gex_profile = data.X
        all_barcodes = data.to_df().index.tolist()
        self.batch_list = []
        for barcode in all_barcodes:
            for k in all_batch_num:
                if k in barcode:
                    self.batch_list.append(all_batch_num[k])

        self.beta_chains = list(data.obs.cdr3)
        n = 0
        self.TCR_ids = {}
        for idx, beta in enumerate(self.beta_chains):
            if beta not in self.TCR_ids:
                self.TCR_ids[beta] = n
                n += 1

    def aa_index(self, tcr, max_len=25):
        tcr_index = [23]
        tcr_index_no_pad = [23]
        tcr_att = [True]
        for i in range(max_len):
            if i < len(tcr):
                tcr_index.append(alphabets[tcr[i]])
                tcr_index_no_pad.append(alphabets[tcr[i]])
                tcr_att.append(True)
            else:
                tcr_index.append(0)
                tcr_att.append(False)
        return tcr_index, tcr_index_no_pad, tcr_att

    def trucate_gex(self, gex, max_gex_len=800):
        non_zero_indices = np.where(gex != 0)[0]
        non_zero_count = len(non_zero_indices)

        if non_zero_count > max_gex_len:
            selected_indices = np.random.choice(non_zero_indices, size=max_gex_len, replace=False)

            for index in non_zero_indices:
                if index not in selected_indices:
                    gex[index] = 0
        return gex

    def __getitem__(self, item):
        # batch_profile = self.gex_profile[item].toarray().squeeze()
        batch_profile = self.gex_profile[item].squeeze()
        # batch_profile = self.trucate_gex(batch_profile)
        batch_profile = torch.tensor(batch_profile, dtype=torch.float32)
        tcr_index, tcr_index_no_pad, tcr_att = self.aa_index(self.beta_chains[item])
        mlm_input_tokens_id, mlm_label = get_masked_sample(tcr_index_no_pad[1:])
        mlm_input_tokens_id, mlm_label = [23] + mlm_input_tokens_id, [-100] + mlm_label
        mlm_input_tokens_id, mlm_label = torch.LongTensor(mlm_input_tokens_id), torch.LongTensor(mlm_label)
        tcr_index, tcr_att = torch.LongTensor(tcr_index), torch.BoolTensor(tcr_att)
        return batch_profile, self.TCR_ids[self.beta_chains[item]], item, mlm_input_tokens_id, mlm_label, tcr_index, tcr_att

    def __len__(self):
        return self.gex_profile.shape[0]
