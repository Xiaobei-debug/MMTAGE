from modules.cross_attention import Cross_attention_encoder_layer
import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class tcr_to_gex(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.zeros(1, 1, 64)
        )
        self.cross_pred = Cross_attention_encoder_layer(64, 64, 4)
        self.pred_head = Decoder(64, 256, 5000)

    def forward(self, tcr_input, tcr_mask, gex_targets):
        query_tokens = self.query_tokens.expand(tcr_input.shape[0], -1, -1)
        cross_feat = self.cross_pred(query_tokens, tcr_input, tcr_mask)
        pred = self.pred_head(cross_feat).squeeze(1)

        cross_loss = F.mse_loss(pred, gex_targets)

        return cross_loss


class gex_to_tcr(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.zeros(1, 26, 256)
        )
        self.cross_pred = Cross_attention_encoder_layer(256, 256, 8)
        self.pred_head = Decoder(256, 256, 24)

    def forward(self, gex_input, gex_mask, TCR_index):
        query_tokens = self.query_tokens.expand(gex_input.shape[0], -1, -1)
        cross_feat = self.cross_pred(query_tokens, gex_input, gex_mask)
        pred = self.pred_head(cross_feat)

        cross_loss = F.cross_entropy(
            pred.view(-1, 24),
            TCR_index.view(-1),
        )
        return cross_loss


class EmbToTCRDecoder(nn.Module):
    def __init__(self, tcr_len=26, vocab_size=24):
        super(EmbToTCRDecoder, self).__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, tcr_len * vocab_size)
        self.relu = nn.ReLU()
        self.tcr_len = tcr_len
        self.vocab_size = vocab_size

    def forward(self, emb, TCR_index):
        x = self.relu(self.fc1(emb))
        tcr_representation = self.fc2(x)
        pred = tcr_representation.view(TCR_index.shape[0], self.tcr_len, self.vocab_size)

        cross_loss = F.cross_entropy(
            pred.view(-1, 24),
            TCR_index.view(-1),
        )
        return cross_loss


class EmbtoGeneDecoder(nn.Module):
    def __init__(self, gene_num=5000):
        super(EmbtoGeneDecoder, self).__init__()
        self.fc1 = nn.Linear(64, 1024)
        self.fc2 = nn.Linear(1024, gene_num)
        self.relu = nn.ReLU()

    def forward(self, emb, gex_targets):
        x = self.relu(self.fc1(emb))
        pred = self.fc2(x)

        cross_loss = F.mse_loss(pred, gex_targets)

        return cross_loss

