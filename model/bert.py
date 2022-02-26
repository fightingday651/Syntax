import math, torch, json
import torch.nn.functional as F
import numpy as np
from model.GNN import GNNRelationModel
from pytorch_transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from pytorch_transformers import modeling_bert
from model.tree import head_to_tree, tree_to_adj
from utils import INFINITY_NUMBER
from torch.autograd import Variable


class SyntaxBertConfig(PretrainedConfig):
    pretrained_config_archive_map = modeling_bert.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, vocab_size_or_config_json_file=32607, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2,
                 initializer_range=0.02, layer_norm_eps=1e-12, **kwargs):
        super(SyntaxBertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int) or the path to a pretrained model"
                             " config file (str)")


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


class Pooler(torch.nn.Module):
    def __init__(self, hidden_size=768, mlp_layers=1):
        super(Pooler, self).__init__()
        in_hidden_size = hidden_size
        self.pool_type = "max"
        # output MLP layers
        layers = [torch.nn.Linear(in_hidden_size, hidden_size), torch.nn.Tanh()]
        for _ in range(mlp_layers - 1):
            layers += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.Tanh()]
        self.out_mlp = torch.nn.Sequential(*layers)

    def forward(self, hidden_states, token_mask=None):
        pool_mask = token_mask.eq(0).unsqueeze(2)
        h_out = pool(hidden_states, pool_mask,  type=self.pool_type)
        pooled_output = self.out_mlp(h_out)
        return pooled_output


class BertSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, output_attentions=None):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                             % (hidden_size, num_attention_heads))
        self.output_attentions = output_attentions
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)

    def forward(self, h, z, attention_mask, attention_probs_dropout_prob=0.1):
        query = self.query(h)
        key = self.key(h)
        value = self.value(h)
        return self.multi_head_attention(key, value, query, attention_mask, dropout_ratio=attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def multi_head_attention(self, key, value, query, attention_mask, dropout_ratio=0.1):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = F.dropout(attention_probs, p=dropout_ratio, training=self.training)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class JointFusionAttention(BertSelfAttention):
    def __init__(self, hidden_size, num_attention_heads, output_attentions=None):
        super(JointFusionAttention, self).__init__(hidden_size, num_attention_heads, output_attentions)
        self.ukey = torch.nn.Linear(hidden_size, self.all_head_size)
        self.uvalue = torch.nn.Linear(hidden_size, self.all_head_size)

    def forward(self, h, z, attention_mask, attention_probs_dropout_prob=0.1):
        query = self.query(h)
        key = self.key(h) + self.ukey(z)
        value = self.value(h) + self.uvalue(z)
        return self.multi_head_attention(key, value, query, attention_mask, dropout_ratio=attention_probs_dropout_prob)


class BertAttention(torch.nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.output = modeling_bert.BertSelfOutput(config)
        self.self = JointFusionAttention(config.hidden_size, config.num_attention_heads)
        self.pruned_heads = set()

    def forward(self, input_tensor, z, attention_mask):
        self_outputs = self.self(input_tensor, z, attention_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayer(torch.nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = modeling_bert.BertIntermediate(config)
        self.output = modeling_bert.BertOutput(config)

    def forward(self, hidden_states, z, attention_mask):
        attention_outputs = self.attention(hidden_states, z, attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(torch.nn.Module):
    def __init__(self, config, output_attentions=None, output_hidden_states=None):
        super(BertEncoder, self).__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.layer = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, z, attention_mask):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, z, attention_mask)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPreTrainedModel(PreTrainedModel):
    config_class = SyntaxBertConfig  # 处理权重初始化的抽象类和用于下载和加载预训练模型的简单接口
    pretrained_model_archive_map = modeling_bert.BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = modeling_bert.load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):  # Initialize the weights.
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, modeling_bert.BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config
        self.syntax_encoder = GNNRelationModel(config)
        self.embeddings = modeling_bert.BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = Pooler(hidden_size=768, mlp_layers=1)
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def inputs_to_tree_reps(self, dep_head, seq_len, dep_rel, device):
        maxlen = max(seq_len)
        trees = [head_to_tree(dep_head[i], seq_len[i], dep_rel[i]) for i in range(len(seq_len))]
        adj_matrix_list, dep_rel_matrix_list = [], []
        for tree in trees:  # 将self_loop=True设为邻边将在图注意力期间用作掩蔽矩阵
            adj_matrix, dep_rel_matrix = tree_to_adj(maxlen, tree, directed=False, self_loop=True)
            adj_matrix = adj_matrix.reshape(1, maxlen, maxlen)
            adj_matrix_list.append(adj_matrix)
            dep_rel_matrix = dep_rel_matrix.reshape(1, maxlen, maxlen)
            dep_rel_matrix_list.append(dep_rel_matrix)
        batch_adj_matrix = torch.from_numpy(np.concatenate(adj_matrix_list, axis=0))
        batch_dep_rel_matrix = torch.from_numpy(np.concatenate(dep_rel_matrix_list, axis=0))
        return Variable(batch_adj_matrix.to(device)), Variable(batch_dep_rel_matrix.to(device))

    def postprocess_attention_mask(self, mask):
        # 由于attention_mask对于参加训练的位置是1.0，对于被屏蔽的位置是0.0，
        # 这个操作将创建一个张量，对于参加训练的位置是0.0，对于被屏蔽的位置是 -10000.0
        # 由于我们在softmax之前将其添加到原始分数中，因此这实际上与完全删除这些分数相同
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        mask = (1.0 - mask) * -10000.0
        return mask

    @classmethod
    def to_linguistic_space(cls, wp_tensor, wp_rows, align_sizes, wp_seq_lengths):
        device = wp_tensor.device
        new_tensor = []
        for i, (seq_len, size) in enumerate(zip(wp_seq_lengths, align_sizes)):
            wp_weighted = wp_tensor[i, :seq_len] / torch.FloatTensor(size).to(device).unsqueeze(1)
            new_row = []
            for j, word_piece_slice in enumerate(wp_rows[i]):
                tensor = torch.sum(wp_weighted[word_piece_slice],
                                   dim=0, keepdim=True)
                new_row.append(tensor)
            new_row = torch.cat(new_row)
            new_tensor.append(new_row)
        new_tensor = torch.nn.utils.rnn.pad_sequence(new_tensor, batch_first=True)
        return new_tensor

    def forward(self, input_ids, wp_token_mask=None, token_type_ids=None, dep_head=None, dep_rel=None,
                seq_len=None, wp_rows=None, align_sizes=None):
        if wp_token_mask is None:
            wp_token_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # if self.config.model_type in {"late_fusion", "joint_fusion"}:
        adj_matrix, dep_rel_matrix = self.inputs_to_tree_reps(dep_head, seq_len, dep_rel, input_ids.device)
        self_attention_mask = wp_token_mask[:, None, None, :]
        self_attention_mask = self.postprocess_attention_mask(self_attention_mask)
        syntax_enc_outputs = self.syntax_encoder(input_ids, adj_matrix, dep_rel_matrix)
        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, syntax_enc_outputs, self_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output, wp_token_mask)
        # to linguistic space
        # token_output = self.to_linguistic_space(sequence_output, wp_rows, align_sizes, seq_len)
        # outputs = (sequence_output, token_output, pooled_output)
        outputs = (sequence_output, pooled_output)
        return outputs