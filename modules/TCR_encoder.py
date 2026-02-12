import torch.nn as nn
from modules.models.blip2_models.blip2 import (
    Blip2Base,
    LayerNorm,
)
from torch.nn import functional as F


class TCR_encoder(Blip2Base):
    def __init__(
            self,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=64,
    ):
        super().__init__()
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, embed_dim, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(25)
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        text_output = self.Qformer.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = text_output.last_hidden_state
        text_features = self.text_proj(text_embeds)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
