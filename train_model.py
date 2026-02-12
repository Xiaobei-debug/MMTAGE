import argparse
import torch
import numpy as np
import random
import os
from pretrain_model import pretrain_model
from tools import getEncoerDecoderData
from data import pretrain_dataset
import scanpy as sc
import anndata as ad


def parser():
    ap = argparse.ArgumentParser(description='TCR-peptide model')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', default=20, type=int)
    ap.add_argument('--batch_size', default=2, type=int)
    ap.add_argument('--lr', default=0.0001, type=float)
    ap.add_argument('--extract_feat', default=False, type=bool)
    ap.add_argument('--save_dir', default="./checkpoints/", type=str)
    ap.add_argument('--dataset', default="./data/10X_all_data.h5ad", type=str)
    ap.add_argument('--load_model', default="./checkpoints/checkpoint19.pt", type=str)
    args = ap.parse_args()
    return args


def pretrain(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pre_model = pretrain_model(learning_rate=args.lr).to(device)

    pre_model.optimizer = torch.optim.AdamW(pre_model.parameters(), lr=args.lr)
    dataset = pretrain_dataset(args.dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs):
        for idx, b in enumerate(dataloader):
            encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, \
            new_data_raw, data_mask_labels, decoder_position_gene_ids, mask_label, encoder_data_ori, decoder_data_ori = getEncoerDecoderData(
                b[0], b[0], 103, 102, 101, b[0].shape[1] + 1)
            tcr_self_loss, gex_self_loss, contrastive_loss, mm_tcr_loss, emb_to_gex_loss = pre_model(b[1].to(device),
                              (encoder_data.to(device), encoder_data_padding.to(device), encoder_position_gene_ids.to(device),
                               encoder_labels.to(device), decoder_data.to(device), decoder_position_gene_ids.to(device),
                               decoder_data_padding.to(device), mask_label.to(device), encoder_data_ori.to(device), decoder_data_ori.to(device)),
                               b[3].to(device), b[4].to(device), b[5].to(device), b[6].to(device))

            if epoch < 3:
                loss_b = 0.5 * tcr_self_loss + 0.5 * gex_self_loss + 0.5 * contrastive_loss
            elif epoch < 6:
                loss_b = 0.5 * (contrastive_loss + mm_tcr_loss) + 0.5 * tcr_self_loss + 0.5 * gex_self_loss
            else:
                loss_b = 0.6 * (contrastive_loss + mm_tcr_loss + emb_to_gex_loss) + 0.3 * tcr_self_loss + 0.3 * gex_self_loss
                if idx % 500 == 1:
                    torch.save(pre_model.state_dict(), args.save_dir + f"checkpoint{epoch}.pt")
            pre_model.optimizer.zero_grad()
            loss_b.backward()
            pre_model.optimizer.step()

        torch.save(pre_model.state_dict(), args.save_dir + f"checkpoint{epoch}.pt")


def extract_feat(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pre_model = pretrain_model(learning_rate=args.lr).to(device)
    checkpoint = torch.load(args.load_model, map_location=torch.device(device))
    pre_model.load_state_dict(checkpoint)
    pre_model.eval()

    dataset = pretrain_dataset(args.dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    Multimodal_embedding_cls = []
    Multimodal_embedding_mean = []
    with torch.no_grad():
        for idx, b in enumerate(dataloader):
            encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, \
            new_data_raw, data_mask_labels, decoder_position_gene_ids, mask_label, encoder_data_ori, decoder_data_ori = getEncoerDecoderData(
                b[0], b[0], 103, 102, 101, b[0].shape[1] + 1, False)
            multimodal_feat_cls, multimodal_feat_mean = pre_model.extract_multimodal_feat(
                              (encoder_data.to(device), encoder_data_padding.to(device), encoder_position_gene_ids.to(device),encoder_labels.to(device), decoder_data.to(device), decoder_position_gene_ids.to(device),
                               decoder_data_padding.to(device), mask_label.to(device), encoder_data_ori.to(device)),
                               b[5].to(device), b[6].to(device))
            Multimodal_embedding_cls.append(multimodal_feat_cls.to(torch.device("cpu")))
            Multimodal_embedding_mean.append(multimodal_feat_mean.to(torch.device("cpu")))
            if idx % 50 == 1:
                print(idx)

    original_data = sc.read_h5ad(args.dataset)

    Multimodal_embedding = torch.cat(Multimodal_embedding_cls, dim=0)
    adata_multimodal = ad.AnnData(Multimodal_embedding.numpy())
    adata_multimodal.obs = original_data.obs
    adata_multimodal.write("multimodal_emb_cls.h5ad")

    Multimodal_embedding = torch.cat(Multimodal_embedding_mean, dim=0)
    adata_multimodal = ad.AnnData(Multimodal_embedding.numpy())
    adata_multimodal.obs = original_data.obs
    adata_multimodal.write("multimodal_emb_mean.h5ad")


if __name__ == '__main__':
    args = parser()
    if args.extract_feat:
        extract_feat(args)
    else:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        pretrain(args)
