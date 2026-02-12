import torch
from torch import nn
from torch.nn import functional as F
from tools import select_model
from gex_encoder_config import gex_encoder_config
from modules.TCR_encoder import TCR_encoder
from modules.multimodal_encoder import FLAVATransformerWithoutEmbeddings
from modules.mask_prediction_head import MaskedPredictionHead
from modules.cross_pred import EmbtoGeneDecoder


class pretrain_model(nn.Module):
    def __init__(self,
                 encoder_dim=64,
                 temperature=0.2,
                 learning_rate=0.0001
                 ):
        super().__init__()
        # gene expression encoder
        self.gene_exp_encoder = select_model(gex_encoder_config)
        self.gex_proj_layer = nn.Linear(encoder_dim, encoder_dim)

        # tcr encoder
        self.tcr_encoder = TCR_encoder()
        self.tcr_self_mlm_pred = nn.Linear(encoder_dim, 22, bias=False)
        self.tcr_proj_layer = nn.Linear(encoder_dim, encoder_dim)

        # multimodal
        self.multimodal_encoder = FLAVATransformerWithoutEmbeddings()
        self.tcr_to_mm = nn.Linear(encoder_dim, 64)
        self.gex_to_mm = nn.Linear(encoder_dim, 64)
        self.mm_tcr_pred = MaskedPredictionHead(hidden_size=64)
        self.mm_gex_norm = nn.LayerNorm(64)
        self.mm_gex_pred = nn.Linear(64, 1)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.temperature = temperature

        self.emb_to_gex = EmbtoGeneDecoder()

    def forward(self, idx_TCR, gene_exp_input, TCR_mask, mask_TCR_label, TCR_index, TCR_att):
        ############# gene expression ##############
        encoder_data, encoder_data_padding, encoder_position_gene_ids, encoder_labels, \
        decoder_data, decoder_position_gene_ids, decoder_data_padding, decoder_mask_label, encoder_data_ori, decoder_data_ori = gene_exp_input
        gene_out, gene_encoder_output, gene_decoder_output = self.gene_exp_encoder.forward(
            x=encoder_data, padding_label=encoder_data_padding,
            encoder_position_gene_ids=encoder_position_gene_ids,
            encoder_labels=encoder_labels,
            decoder_data=decoder_data,
            mask_gene_name=False,
            mask_labels=None,
            decoder_position_gene_ids=decoder_position_gene_ids,
            decoder_data_padding_labels=decoder_data_padding,
            pred_ori_val=True,
            )

        # create mask
        mask_all = decoder_mask_label != -100
        # apply mask
        filtered_predictions = gene_out[mask_all]
        filtered_targets = decoder_mask_label[mask_all]
        # MSE loss
        gex_self_loss = F.mse_loss(filtered_predictions, filtered_targets)

        # cell embedding
        cell_output = gene_decoder_output[:, 0]
        cell_output = self.gex_proj_layer(cell_output)
        cell_output = F.normalize(cell_output, dim=-1)

        ############# tcr ##############
        tcr_enc_output = self.tcr_encoder(TCR_mask, TCR_att)
        mlm_tcr_encoder = tcr_enc_output

        tcr_enc_outputs_no_mask = self.tcr_encoder(TCR_index, TCR_att)
        tcr_enc_outputs_no_mask = tcr_enc_outputs_no_mask[:, 0]
        tcr_enc_outputs_no_mask = self.tcr_proj_layer(tcr_enc_outputs_no_mask)
        tcr_enc_outputs_no_mask = F.normalize(tcr_enc_outputs_no_mask, dim=-1)

        mlm_logits_self = self.tcr_self_mlm_pred(mlm_tcr_encoder)
        tcr_self_loss = F.cross_entropy(
            mlm_logits_self.view(-1, 22),
            mask_TCR_label.view(-1),
            ignore_index=-100,
        )

        ############# multimodal ##############
        gene_encoder_output = self.gex_to_mm(gene_encoder_output)
        tcr_enc_output = self.tcr_to_mm(tcr_enc_output)
        mm_feature = torch.cat((tcr_enc_output, gene_encoder_output), dim=1)

        mm_feature = self.multimodal_encoder(mm_feature).last_hidden_state

        mm_tcr_output = self.mm_tcr_pred(mm_feature[:, 1:27])
        mm_tcr_loss = F.cross_entropy(
            mm_tcr_output.view(-1, 22),
            mask_TCR_label.view(-1),
            ignore_index=-100,
        )

        mm_gex_output = self.mm_gex_pred(self.mm_gex_norm(mm_feature[:, 27:]))
        # apply mask
        mask_mm_gex = encoder_data == 102
        filtered_targets = encoder_data_ori[mask_mm_gex]
        mm_gex_output = mm_gex_output[mask_mm_gex].squeeze(-1)
        # MSE l0ss
        mm_gex_loss = F.mse_loss(mm_gex_output, filtered_targets)
        
        emb_to_gex_loss = self.emb_to_gex(mm_feature[:, 0], decoder_data_ori[:, 1:])

        ############# contrastive loss ##############
        sim_g2t = cell_output @ tcr_enc_outputs_no_mask.T / self.temperature
        sim_t2g = tcr_enc_outputs_no_mask @ cell_output.T / self.temperature

        with torch.no_grad():
            idx_TCR = idx_TCR.view(-1, 1)
            pos_idx_TCR = torch.eq(idx_TCR, idx_TCR.T)

            sim_targets = (pos_idx_TCR).float().to(tcr_enc_output.device)
            sim_targets = sim_targets / sim_targets.sum(1, keepdim=True)

        # calculate the ptc loss
        loss_p2t = -torch.sum(F.log_softmax(sim_g2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2p = -torch.sum(F.log_softmax(sim_t2g, dim=1) * sim_targets, dim=1).mean()
        contrastive_loss = (loss_p2t + loss_t2p) / 2

        return tcr_self_loss, gex_self_loss, contrastive_loss, mm_tcr_loss + mm_gex_loss, emb_to_gex_loss

    def extract_multimodal_feat(self, gene_exp_input, TCR_index, TCR_att):
        ############# gene expression ##############
        encoder_data, encoder_data_padding, encoder_position_gene_ids, encoder_labels, \
        decoder_data, decoder_position_gene_ids, decoder_data_padding, decoder_mask_label, encoder_data_ori = gene_exp_input
        gene_out, gene_encoder_output, gene_decoder_output = self.gene_exp_encoder.forward(x=encoder_data,
                                                                                           padding_label=encoder_data_padding,
                                                                                           encoder_position_gene_ids=encoder_position_gene_ids,
                                                                                           encoder_labels=encoder_labels,
                                                                                           decoder_data=decoder_data,
                                                                                           mask_gene_name=False,
                                                                                           mask_labels=None,
                                                                                           decoder_position_gene_ids=decoder_position_gene_ids,
                                                                                           decoder_data_padding_labels=decoder_data_padding,
                                                                                           )
        tcr_enc_output = self.tcr_encoder(TCR_index, TCR_att)

        gene_encoder_output = self.gex_to_mm(gene_encoder_output)
        tcr_enc_output = self.tcr_to_mm(tcr_enc_output)
        mm_feature = torch.cat((tcr_enc_output, gene_encoder_output), dim=1)
        mm_feature = self.multimodal_encoder(mm_feature).last_hidden_state
        cls_feat = mm_feature[:, 0]
        mm_feature = torch.cat((torch.mean(mm_feature[:, 1:27], dim=1).unsqueeze(1), torch.mean(mm_feature[:, 27:], dim=1).unsqueeze(1)),dim=1)
        # mm_feature = torch.cat((torch.mean(mm_feature[:, 1:27], dim=1), torch.mean(mm_feature[:, 27:], dim=1)), dim=1)
        # return torch.max(mm_feature, dim=1).values
        return cls_feat, torch.mean(mm_feature, dim=1)


