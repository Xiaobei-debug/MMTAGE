import torch
from torch import nn
import math
from torch.nn import functional as F


class Input_embedding(nn.Module):
    """
    this is the class used for embedding input

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        a new normalized data(bs, seq_len, out_dim) based on the linear transformation and layer normalization
    """

    def __init__(self, in_dim=5, out_dim=5):
        super().__init__()

        # define the dimension expansion layer, i.e. mapping original dimension into a new space
        self.expand = nn.Linear(in_dim, out_dim)

        # define the layer normalization
        self.layer_norm = nn.LayerNorm(out_dim)

        # define dropout layer
        self.dropout = nn.Dropout(0.15)

    def forward(self, input_data):
        # expand the input data
        expand_data = self.expand(input_data)

        # layer normalization the input data
        normalized_data = self.layer_norm(expand_data)

        # dropout
        normalized_data = self.dropout(normalized_data)

        return normalized_data


class Self_attention(nn.Module):
    """
    this is the basic self-attention class

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hidden states(bs, seq_len, out_dim) based on the self-attention mechanism
    """

    def __init__(self, in_dim=5, out_dim=5, head_num=1):
        super().__init__()
        if out_dim % head_num != 0:
            raise ValueError(
                "Hidden size is not a multiple of the number of attention head"
            )

        # define the head number
        self.head_num = head_num

        # define the head size
        self.head_size = int(out_dim / head_num)

        # define the all head dimension
        self.out_dim = out_dim

        # define the Q, K, V matrix
        self.Q = nn.Linear(in_dim, out_dim)
        self.K = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)

    def transpose_for_score(self, x):
        new_x_shape = x.size()[:-1] + (self.head_num, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_data, mask):
        # calculate the Q, K, V matrix
        Q = self.Q(input_data)
        K = self.K(input_data)
        V = self.V(input_data)

        # multi-head
        Q = self.transpose_for_score(Q)
        K = self.transpose_for_score(K)
        V = self.transpose_for_score(V)

        # calculate the logits based on the Query matrix and key matrix
        score = torch.matmul(Q, K.transpose(-1, -2))
        score /= math.sqrt(self.head_size)

        # mask the padding
        # score.transpose(-1, -2)[mask] = -1e15
        new_mask = mask[:, None, None, :] * (-1e15)
        score = score + new_mask

        # calculate the attention score
        att = F.softmax(score, dim=-1)

        # output the result based on the attention and value
        output = torch.matmul(att, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        new_output_shape = output.size()[:-2] + (self.out_dim,)
        output = output.view(*new_output_shape)

        return output


class Attention_output(nn.Module):
    """
    this is the linear transformation class

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hidden states(bs, seq_len, out_dim) based on the feedforward and residual connection
    """

    def __init__(self, in_dim=5, out_dim=5):
        super().__init__()

        # define the linear layer
        self.feedforward = nn.Linear(in_dim, out_dim)

        # define the layer normalization layer
        self.layernorm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, hidden_states, input_data):
        # feedforward the input data
        hidden_states = self.feedforward(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # resnet the input data and output with layer normalization
        hidden_states = self.layernorm(hidden_states + input_data)

        return hidden_states


class Attention_intermediate(nn.Module):
    """
    this is the intermediate class

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hidden states(bs, seq_len, out_dim) base on the feedforward and gelu activattion
    """

    def __init__(self, in_dim=5, out_dim=5):
        super().__init__()

        # define the linear layer
        self.feedforward = nn.Linear(in_dim, out_dim)

        # define the gelu as activate function
        self.gelu = nn.GELU()

    def forward(self, input_data):
        # feedforward the input data
        hidden_states = self.feedforward(input_data)

        # activate the hidden state
        hidden_states = self.gelu(hidden_states)

        return hidden_states


class Attention_pooler(nn.Module):
    """
    this is the mean pooling class

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hidden state(bs, out_dim) output by masking the padding and averaging the seq
    """

    def __init__(self, in_dim=5, out_dim=5):
        super().__init__()

        # define the linear layer
        self.feedforward = nn.Linear(in_dim, out_dim)

        # define the tanh as activation function
        self.activation = nn.Tanh()

    def forward(self, input_data, mask=None):

        if mask is not None:

            # setting the value of padding as 0
            input_data[mask] = 0

            # calculate the mean value
            pooled_output = input_data.sum(dim=1) * (1 / (~mask).sum(dim=1)).unsqueeze(0).T
        else:
            pooled_output = input_data.squeeze(1)

        # linear transformation of output
        pooled_output = self.feedforward(pooled_output)

        # activate the output
        pooled_output = self.activation(pooled_output)

        return pooled_output


class Self_attention_layer(nn.Module):
    """
    this is the self_attention_layer class, which contains self-attention blocks and output layer with residual connection

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hidden states(bs, seq_len, out_dim)
    """

    def __init__(self, in_dim=5, out_dim=5, head_num=1):
        super().__init__()

        # define the self-attention layer
        self.self_attention = Self_attention(in_dim, out_dim, head_num)

        # define the attention-output layer
        self.self_output = Attention_output(out_dim, in_dim)

    def forward(self, input_data, mask):
        # input data into the self-attention layer
        hidden_states = self.self_attention(input_data, mask)

        # feedforward the output
        hidden_states = self.self_output(hidden_states, input_data)

        return hidden_states


class Self_attention_encoder_layer(nn.Module):
    """
    this is the self-attention encoder blocks, which contains self-attention-layer, intermediate and output layer

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hiddens states(bs, seq_len, out_dim)
    """

    def __init__(self, in_dim=5, out_dim=5, head_num=1):
        super().__init__()

        # define the self-attention-layer
        self.attention_layer = Self_attention_layer(in_dim, out_dim, head_num)

        # define the attention intermediate layer
        self.intermediate = Attention_intermediate(out_dim, out_dim)

        # define the attention-output layer
        self.output = Attention_output(out_dim, in_dim)

    def forward(self, input_data, mask):
        # self-attention-layer output
        hidden_states = self.attention_layer(input_data, mask)

        # output of intermediate
        intermediate_states = self.intermediate(hidden_states)

        # output the feedforward layer
        output = self.output(intermediate_states, hidden_states)

        return output


class Cross_attention(nn.Module):
    """
    this is the cross-attention class

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hiddens states(bs, query_seq_len, out_dim)
    """

    def __init__(self, in_dim=5, out_dim=5, head_num=1):
        super().__init__()

        if out_dim % head_num != 0:
            raise ValueError(
                "Hidden size is not a multiple of the number of attention head"
            )

        # define the head number
        self.head_num = head_num

        # define the head size
        self.head_size = int(out_dim / head_num)

        # define the all head dimension
        self.out_dim = out_dim

        # define the Q, K, V matrix
        self.Q = nn.Linear(in_dim, out_dim)
        self.K = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)

    def transpose_for_score(self, x):
        new_x_shape = x.size()[:-1] + (self.head_num, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_input, value_input, v_mask):
        # size:(btz, n_samples, seq_len, features)

        # calculate the Q, K, V matrix
        Q = self.Q(query_input)
        K = self.K(value_input)
        V = self.V(value_input)

        # Multi-head
        Q = self.transpose_for_score(Q)
        K = self.transpose_for_score(K)
        V = self.transpose_for_score(V)

        # calculate the logits of cross-attention
        score = torch.matmul(Q, K.transpose(-1, -2))
        score /= math.sqrt(self.head_size)

        # mask the padding
        # score.transpose(1, 2)[v_mask] = -1e15
        new_mask = v_mask[:, None, None, :] * (-1e15)
        score = score + new_mask

        # calculate the attention score
        att = F.softmax(score, dim=-1)

        # output the result based on the attention and value
        output = torch.matmul(att, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        new_output_shape = output.size()[:-2] + (self.out_dim,)
        output = output.view(*new_output_shape)

        return output


class Cross_attention_layer(nn.Module):
    """
    this is the cross-attention-layer class, which contains cross-attention and output layer

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hidden states(bs, query_seq_len, out_dim)
    """

    def __init__(self, in_dim=5, out_dim=5, head_num=1):
        super().__init__()

        # define the cross-attention layer
        self.cross_attention = Cross_attention(in_dim, out_dim, head_num)

        # define the feedforward layer
        self.self_output = Attention_output(out_dim, in_dim)

    def forward(self, query_input, value_input, v_mask):
        # calculate the cross-attention
        hidden_states = self.cross_attention(query_input, value_input, v_mask)

        # calculate the feedforward output
        hidden_states = self.self_output(hidden_states, query_input)

        return hidden_states


class Cross_attention_encoder_layer(nn.Module):
    """
    this is the cross-attention-encoder-layer, which constains self-attention-layer, cross-attention-layer
    intermediate layer and output layer

    Parameters:
        param in_dim: the input dimension
        param out_dim: the output dimension

    Returns:
        hidden states(bs, query_seq_len, out_dim)
    """

    def __init__(self, in_dim=5, out_dim=5, head_num=1):
        super().__init__()

        # define the self-attention-layer
        # self.self_attention_layer = Self_attention_layer(in_dim, out_dim, head_num)

        # define the cross-attention-layer
        self.cross_attention_layer = Cross_attention_layer(in_dim, out_dim, head_num)

        # define the intermediate layer
        self.intermediate = Attention_intermediate(in_dim, out_dim)

        # define the output layer
        self.output = Attention_output(out_dim, in_dim)

    def forward(self, query_input, value_input, v_mask):
        # output the result of self-attention-layer based on the value input
        # hidden_value_states = self.self_attention_layer(value_input, v_mask)

        # output the result of cross-attention-layer
        hidden_states = self.cross_attention_layer(query_input, value_input, v_mask)

        # output the result of intermediate
        intermediate_states = self.intermediate(hidden_states)

        # output the feedforward
        output = self.output(intermediate_states, hidden_states)

        return output
