import math
import torch
import scipy.stats as stats
from pytorch_transformers import modeling_bert
from torch import nn
from torch.autograd import Variable
from utils import PAD_ID, DEPREL_TO_ID
import numpy as np


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32):  # 从截断的正态分布中输出随机值
    # 生成的值遵循具有指定均值和标准差的正态分布，期望大小与均值相差超过2个标准差的值会被丢弃并重新选取。
    lower = -2 * stddev + mean
    upper = 2 * stddev + mean
    X = stats.truncnorm((lower - mean) / stddev, (upper - mean) / stddev, loc=mean, scale=stddev)
    values = X.rvs(size=shape)
    return torch.from_numpy(values.astype(dtype))


class ScaledEmbedding(nn.Embedding):  # Embedding layer that initialises its values to using a truncated normal variable scaled by the inverse of the embedding dimension.
    def reset_parameters(self):  # Initialize parameters using Truncated Normal Initializer (default in Tensorflow)
        self.weight.data = truncated_normal(shape=(self.num_embeddings, self.embedding_dim), stddev=1.0 / math.sqrt(self.embedding_dim))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


def rnn_zero_state(batch_size, hidden_size, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_size)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class RelationalBertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, output_attentions=None):
        super(RelationalBertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                             % (hidden_size, num_attention_heads))
        self.output_attentions = output_attentions
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, dep_rel_matrix=None, use_dep_rel=True):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        rel_attention_scores = 0
        if use_dep_rel:
            rel_attention_scores = query_layer[:, :, :, None, :] * dep_rel_matrix[:, None, :, :, :]
            rel_attention_scores = torch.sum(rel_attention_scores, -1)
        attention_scores = (attention_scores + rel_attention_scores) / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        if use_dep_rel:
            val_edge = attention_probs[:, :, :, :, None] * dep_rel_matrix[:, None, :, :, :]
            context_layer = context_layer + torch.sum(val_edge, -2)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, hidden_act, intermediate_size, activation_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.W_1 = nn.Linear(hidden_size, intermediate_size)
        self.act = modeling_bert.ACT2FN[hidden_act]
        self.dropout = nn.Dropout(activation_dropout)
        self.W_2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, e):
        e = self.dropout(self.act(self.W_1(e)))
        e = self.W_2(e)
        return e


class GATEncoderLayer(nn.Module):
    def __init__(self, hidden_size, layer_prepostprocess_dropout, num_attention_heads, attention_probs_dropout_prob,
                 hidden_act, intermediate_size):
        super(GATEncoderLayer, self).__init__()
        self.syntax_attention = RelationalBertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.finishing_linear_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(layer_prepostprocess_dropout)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.feed_forward = FeedForwardLayer(hidden_size, hidden_act, intermediate_size, activation_dropout=0.1)
        self.dropout2 = nn.Dropout(layer_prepostprocess_dropout)
        self.ln_3 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, e, attention_mask, dep_rel_matrix=None):
        sub = self.finishing_linear_layer(self.syntax_attention(self.ln_2(e), attention_mask, dep_rel_matrix)[0])
        sub = self.dropout1(sub)
        sub = self.feed_forward(self.ln_3(e + sub))
        sub = self.dropout2(sub)
        return e + sub


class GATEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, layer_prepostprocess_dropout, num_attention_heads,
                 attention_probs_dropout_prob, hidden_act, intermediate_size):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GATEncoderLayer(hidden_size, layer_prepostprocess_dropout, num_attention_heads,
                                    attention_probs_dropout_prob, hidden_act, intermediate_size)
            self.layers.append(layer)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, e, attention_mask, dep_rel_matrix=None):
        for layer in self.layers:
            e = layer(e, attention_mask, dep_rel_matrix)
        e = self.ln(e)
        return e


class GNNRelationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_dep_rel = config.use_dep_rel
        self.embed_position = config.embed_position
        self.emb = ScaledEmbedding(config.vocab_size, config.emb_size, padding_idx=PAD_ID)
        self.input_dropout = nn.Dropout(config.input_dropout)
        if config.embed_position:
            self.embed_pos = ScaledEmbedding(config.max_position_embeddings, config.emb_size)
        if config.use_dep_rel:
            self.rel_emb = ScaledEmbedding(len(DEPREL_TO_ID), int(config.hidden_size / config.num_attention_heads),
                                           padding_idx=DEPREL_TO_ID['[PAD]'])
        # Graph Attention layer
        self.syntax_encoder = GATEncoder(config.num_layers, config.hidden_size, config.layer_prepostprocess_dropout,
                                         config.num_attention_heads, config.attention_probs_dropout_prob,
                                         config.hidden_act, config.intermediate_size)
        # output MLP layers
        layers = [nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh()]
        for _ in range(config.mlp_layers - 1):
            layers += [nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh()]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, input_ids_or_bert_hidden, adj=None, dep_rel_matrix=None):
        embeddings = self.emb(input_ids_or_bert_hidden)
        if self.embed_position:
            seq_length = input_ids_or_bert_hidden.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids_or_bert_hidden.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids_or_bert_hidden)
            position_embeddings = self.embed_pos(position_ids)
            embeddings += position_embeddings
        embeddings = self.input_dropout(embeddings)
        syntax_inputs = embeddings
        dep_rel_emb = None
        if self.use_dep_rel:
            dep_rel_emb = self.rel_emb(dep_rel_matrix)
        attention_mask = adj.clone().detach().unsqueeze(1)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0
        h = self.syntax_encoder(syntax_inputs, attention_mask, dep_rel_emb)
        return h