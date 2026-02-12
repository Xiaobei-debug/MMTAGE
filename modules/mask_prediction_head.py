import torch
from torch import nn
from modules.normalizations import Fp32LayerNorm
from typing import Callable
from torch import Tensor


class MaskedPredictionHead(nn.Module):
    def __init__(
            self,
            hidden_size: int = 64,
            vocab_size: int = 22,
            transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
            layer_norm_eps: float = 1e-5,
            use_fp32_layer_norm: bool = True,
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn

        self.layer_norm: nn.LayerNorm
        if use_fp32_layer_norm:
            self.layer_norm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is
        # correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
