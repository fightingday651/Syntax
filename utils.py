import json, os, random, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
from pytorch_lightning import seed_everything
from torch.backends import cudnn
PAD_TOKEN = '[PAD]'
PAD_ID = 0
UNK_TOKEN = '[UNK]'
UNK_ID = 1
DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, "ATT": 2, "ADV": 3, "VOB": 4, "SBV": 5, "HED": 6, "COO": 7, "POB": 8,
                "IC": 9, "CMP": 10, "DBL": 11, "VV": 12, "F": 13, "DOB": 14, "MT": 15, 'subtokens': 16, 'special_rel': 17}
Dtags_TO_ID = {"CORRECT": 0, "INCORRECT": 1}
INFINITY_NUMBER = 1e12


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else UNK_ID for t in tokens]
    return ids


class InputExample(object):  # A single training/test example
    def __init__(self, text_a, labels=None, dep_head=None, dep_rel=None):
        self.text_a = text_a  # string. The untokenized text of the first sequence.
        self.labels = labels  # The label of the example. This should be specified for train and dev examples,
        # but not for test examples.
        if labels:
            self.d_tags = [Dtags_TO_ID["CORRECT"] if label == "$KEEP" else Dtags_TO_ID["INCORRECT"] for label in labels]
        self.dep_head = dep_head  # The dependency head for each word
        self.dep_rel = dep_rel  # The dependency relation between head and tail words


class DataProcessor(object):
    def __init__(self, data_path):
        self.data_dir = data_path
        self.label_map = self.get_label2id()
        self.num_labels_classes = len(self.label_map)
        self.num_detect_classes = len(Dtags_TO_ID)

    def read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        return data

    def get_examples(self, set_type):
        return self._create_examples(self.read_json(os.path.join(self.data_dir, set_type + ".json")), set_type)

    def get_label2id(self):
        return self.read_json(os.path.join(self.data_dir, "label_map.json"))["label2id"]

    def get_id2label(self):
        return self.read_json(os.path.join(self.data_dir, "label_map.json"))["id2label"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line['original_text']
            # Adding dependency tree information
            deprel = [t for t in line['deprel']]
            deprel = map_to_ids(deprel, DEPREL_TO_ID)
            assert all([x != 1 for x in deprel]), print(
                line['deprel'])  # To make sure no relation gets mapped to UNK token
            head = [int(x) for x in line['head']]
            assert any([x == 0 for x in head]), print(line['head'])
            if set_type == 'test':  # 分词后的字符串
                examples.append(InputExample(text_a=text_a, dep_head=head, dep_rel=deprel))
            else:
                lables = map_to_ids(line['det_label'], self.label_map)
                examples.append(InputExample(text_a=text_a, labels=lables, dep_head=head, dep_rel=deprel))
        return examples


def tokenize(sequence, tokenizer):  # Wordpiece-tokenize a list of tokens and return vocab ids.
    word_idxs, bert_tokens, subword_token_len = [], [], []
    idx = 0
    for s in sequence:
        tokens = tokenizer.tokenize(s)
        subword_token_len.append(len(tokens))
        word_idxs += [idx]
        bert_tokens += tokens
        idx += len(tokens)
    return bert_tokens, word_idxs, subword_token_len


def handle_long_sent_parses(dep_head, dep_rel, length):
    dep_rel = dep_rel[:length]
    dep_head, truncated_head = dep_head[:length], dep_head[length:]
    is_root = [True for x in truncated_head if x == 0]  # 检测ROOT是否在截取后剩余的字符串中
    if is_root:
        dep_head[-1] = 0  # 若ROOT在丢弃的字符串中将最后的字作为ROOT
    dep_root_ = [i for i, x in enumerate(dep_head) if x == 0]  # 每句中只有一个ROOT token
    assert len(dep_root_) == 1
    for i, head_word_index in enumerate(dep_head):  # 剩下token的head被截掉了 将此token的head设为它本身
        if head_word_index > len(dep_head):
            dep_head[i] = i + 1  # 设为最后一个token？ dep_head[i] = length
    return dep_head, dep_rel


def compute_alignment(word_idxs, subword_token_len):
    alignment = []  # 将未对齐的字在句中的实际index放在一个list中再放在此列表中
    for i, l in zip(word_idxs, subword_token_len):
        assert l > 0
        aligned_subwords = []
        for j in range(l):
            aligned_subwords.append(i + j)
        alignment.append(aligned_subwords)
    return alignment


def get_boundary_sensitive_alignment(word_pieces, raw_tokens, alignment):
    align_sizes = [0 for _ in range(len(word_pieces))]
    wp_rows = []
    for word_piece_slice in alignment:
        wp_rows.append(word_piece_slice)  # 此处wp_rows同alignment存放分词对应的index
        for i in word_piece_slice:
            align_sizes[i] += 1  # align_sizes除起始结束标签外都置为1长度同word_pieces的长度
    # To make this weighting work, we "align" the boundary tokens against every token in their sentence.
    # The boundary tokens are otherwise unaligned, which is how we identify them.
    offset = 0
    for i in range(len(word_pieces)):
        if align_sizes[offset + i] == 0:  # 标记[CLS] [SEP]开始结束标签
            align_sizes[offset + i] = len(raw_tokens)  # align_sizes开始和结束位置存放了raw_tokens的长度
            for j in range(len(raw_tokens)):
                wp_rows[j].append(offset + i)  # 此处wp_rows相当于alignment每项append开始和结束的标签在word_pieces中对应的id
    return wp_rows, align_sizes


def wp_aligned_dep_parse(word_idxs, subword_token_len, dep_head, dep_rel):
    wp_dep_head, wp_dep_rel = [], []
    # Default ROOT position is the [CLS] token
    root_pos = 0
    for i, (idx, slen) in enumerate(zip(word_idxs, subword_token_len)):
        if i == 0 or i == len(subword_token_len) - 1:  # 将[CLS]和[SEP]加上对应的依赖关系的边
            wp_dep_rel.append(DEPREL_TO_ID['special_rel'])
            wp_dep_head.append(idx + 1)  # 它们的根现在还不清楚，后面我们找到了会修改
        else:
            rel = dep_rel[i - 1]
            wp_dep_rel.append(rel)
            head = dep_head[i - 1]  # 由于前面有[CLS]所以这里是i-1
            if head == 0:  # 依存树中的ROOT token其他的词都将指向它(由于LTP的offset为1因此head为0的是ROOT token)
                root_pos = word_idxs[i]+1
                wp_dep_head.append(0)
            else:
                if head < max(word_idxs):  # Obtain the index of the displaced version of the same head
                    new_pos = word_idxs[head - 1 + 1]
                else:  # TODO: Fix this hack arising due to long lengths
                    new_pos = idx + 1  # self-connection
                wp_dep_head.append(new_pos + 1)

            for _ in range(1, slen):
                wp_dep_rel.append(DEPREL_TO_ID['subtokens'])  # 为一个词中除已建立边的第一个字外的字增加subwords依赖关系
                wp_dep_head.append(idx + 1)  # 将它们的head为设为下一个字
    wp_dep_head[0] = root_pos
    wp_dep_head[-1] = root_pos
    return wp_dep_head, wp_dep_rel


class FeaturizedDataLoader(DataLoader):
    def __init__(self, dataset, gpus, prefix='train', **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.prefix = prefix
        if gpus == 0:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):  # generate batch
        device = self.device
        batch = list(zip(*batch_data))
        maxlen = max(batch[5])
        tensorized = OrderedDict()
        tensorized['input_ids'] = torch.LongTensor(batch[0])[:, :maxlen].to(device)
        tensorized['wp_token_mask'] = torch.LongTensor(batch[1])[:, :maxlen].to(device)
        tensorized['token_type_ids'] = torch.LongTensor(batch[2])[:, :maxlen].to(device)
        tensorized['wp_rows'] = batch[3]
        tensorized['align_sizes'] = batch[4]
        tensorized['seq_len'] = batch[5]
        tensorized['dep_head'] = batch[6]
        tensorized['dep_rel'] = batch[7]
        if self.prefix != 'test':
            # tensorized['input_tokens'] = batch[8]
        # else:
            assert len(batch) == 10
            labels = [x for l in batch[8] for x in l]
            d_tags = [x for l in batch[9] for x in l]
            tensorized['labels'] = torch.LongTensor(labels).to(device)
            tensorized['d_tags'] = torch.LongTensor(d_tags).to(device)

        return tensorized


def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor, targets: torch.LongTensor, weights: torch.FloatTensor,
                                       average: str = "batch") -> torch.FloatTensor:
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.reshape(-1, 1).long()
    negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()
    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


def replace_swap_transforms(tokens):
    target_tokens = ' '.join(tokens)
    for i in range(len(tokens)):
        if tokens[i].startswith("$SWAP_"):  # $SWAP_{f_pos}_{f_len}
            head_index = tokens[i].split('_')[-2]
            if head_index != '$SWAP':
                head_index = int(head_index)
                replace_len = int(tokens[i].split('_')[-1])
            else:
                replace_len = 1  # 没有多替换情况
            old = ' '.join(tokens[head_index: i])
            # assert target_index in allowed_range
            # if target_index in allowed_range:
            swap1, swap2 = ' '.join(tokens[head_index:head_index+replace_len]), ' '.join(tokens[head_index+replace_len: i])
            old += f" {tokens[i]}"
            target_tokens = target_tokens.replace(old, swap2 + ' ' + swap1, 1)
    return target_tokens.split()
