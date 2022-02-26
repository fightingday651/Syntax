import torch
from torch import nn
import torch.nn.functional as F
from model.bert import BertModel, BertPreTrainedModel
from model.crf import CRF, SequenceCriteriaCRF
from utils import PAD_ID, Dtags_TO_ID, sequence_cross_entropy_with_logits


class SyntaxBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, label_map):
        super(SyntaxBertForTokenClassification, self).__init__(config)
        self.incorr_index = Dtags_TO_ID["INCORRECT"]
        self.num_labels_classes = config.num_labels_classes
        self.num_detect_classes = config.num_detect_classes
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.bert = BertModel(config)
        self.t_classifier = nn.Linear(config.hidden_size, config.num_labels_classes)
        self.d_classifier = nn.Linear(config.hidden_size, config.num_detect_classes)
        # self.loss_func = nn.CrossEntropyLoss()
        self.crf, self.loss_func = self.get_crf(config, label_map, 'labels')
        self.init_weights()

    def get_crf(self, config, label_map, type):
        if type == 'labels':  # 过渡参数矩阵，i,j是从j过渡到i的分数
            transitions = nn.Parameter(torch.randn(self.num_labels_classes, self.num_labels_classes))
        else:
            transitions = nn.Parameter(torch.randn(self.num_detect_classes, self.num_detect_classes))
        nn.init.xavier_uniform_(transitions.data)
        crf = CRF(label_map, config=config, transitions=transitions)
        loss_func = SequenceCriteriaCRF(crf)
        return crf, loss_func

    def forward(self, input_ids, wp_token_mask=None, token_type_ids=None, wp_rows=None, align_sizes=None, seq_len=None,
                dep_head=None, dep_rel=None, labels=None, d_tags=None, input_tokens=None):
        outputs = self.bert(input_ids, wp_token_mask, token_type_ids=token_type_ids, wp_rows=wp_rows,
                            align_sizes=align_sizes, seq_len=seq_len, dep_head=dep_head, dep_rel=dep_rel)
        encoded_text = outputs[0]
        batch_size, sequence_length, _ = encoded_text.size()
        encoded_text = F.dropout(encoded_text, p=self.hidden_dropout_prob, training=self.training)
        logits_labels = self.t_classifier(encoded_text)  # outputs[1]
        logits_d = self.d_classifier(encoded_text)
        class_probabilities_d = F.softmax(logits_d, dim=-1).view([batch_size, sequence_length, self.num_detect_classes])
        error_probs = class_probabilities_d[:, :, self.incorr_index] * wp_token_mask
        incorr_prob = torch.max(error_probs, dim=-1)[0]  # skip whole sentence if probability of correctness is not high
        # l = [len(row) for row in wp_rows]
        l = [row for row in seq_len]
        preds = []
        for feats, sequence_length in zip(logits_labels, l):
            _, tag_seq = self.crf(feats[: sequence_length])
            preds.extend(torch.LongTensor(tag_seq).cpu().numpy())
        preds = torch.LongTensor(preds)
        output_dict = {'predict': preds, 'max_error_probability': incorr_prob, 'lens': l}
        if d_tags is not None and labels is not None:  # lable:71 input:71 还是换为token级别
            batch_tags = nn.utils.rnn.pad_sequence(torch.split(d_tags, l), batch_first=True, padding_value=PAD_ID)
            loss_d = sequence_cross_entropy_with_logits(logits_d, batch_tags, wp_token_mask)
            batch_labels = nn.utils.rnn.pad_sequence(torch.split(labels, l), batch_first=True, padding_value=PAD_ID)
            loss_labels = self.loss_func(logits_labels, batch_labels, l)
            output_dict['loss'] = loss_labels+loss_d
        return output_dict
