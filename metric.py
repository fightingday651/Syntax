import json
import os
import numpy as np
import torch
from utils import UNK_TOKEN, PAD_TOKEN, replace_swap_transforms
from pytorch_transformers import BertTokenizer


def compute_metrics(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive + 1e-13)
    recall = true_positive / (true_positive + false_negative + 1e-13)
    f = (1 + 0.5 * 0.5) * precision * recall / (0.5 * 0.5 * precision + recall + 1e-13)
    return [precision, recall, f]


class TokenMetric(object):
    def __init__(self, labels):
        self.labels = labels
        self.min_error_probability = 0
        self.accs = []
        self.detection_metric = [0., 0., 0.]  # true_positive false_positive false_negative
        self.correct_metric = [0., 0., 0.]

    def update(self, pred_sequence_labels, gold_sequence_labels, training=False):
        if training:
            return
        gold_sequence_labels = gold_sequence_labels.detach().to("cpu").numpy().tolist()
        pred_sequence_labels = pred_sequence_labels.tolist()
        self.accs += [k == p for k, p in zip(gold_sequence_labels, pred_sequence_labels)]
        each_true_index, gold_edit, keep = [], [], self.labels.index('$KEEP')
        for i in range(len(pred_sequence_labels)):
            if gold_sequence_labels[i] != keep:  # src_token!=trg_token 有语法错误
                gold_edit.append(i)
                if pred_sequence_labels[i] == keep:  # false_negative有语法错误但判定其没有
                    self.detection_metric[2] += 1
                else:
                    self.detection_metric[0] += 1  # true_positive有语法错误且认为其有
                    each_true_index.append(i)  # 两者都认为有语法错误下标列表
            else:  # src_token==trg_token 没有语法错误
                if pred_sequence_labels[i] == keep:
                    # true_negative += 1  # 没有语法错误且也判定其没有
                    continue
                else:
                    self.detection_metric[1] += 1  # false_positive没有语法错误但判定其有
        predict_edits = []
        for j in each_true_index:
            predict_edits.append(pred_sequence_labels[j])
            if pred_sequence_labels[j] == gold_sequence_labels[j]:  # 两者都认为是这个edit_lable
                self.correct_metric[0] += 1
            else:  # 系统预测的是这个edit_lable但原本不是这个edit_lable
                self.correct_metric[1] += 1
        for j in gold_edit:  # gold中非keep的部分
            if gold_sequence_labels[j] in predict_edits:
                continue
            else:  # gold中的非keep纠错在系统纠错中没有
                self.correct_metric[2] += 1

    def reset(self):
        self.detection_metric = [0., 0., 0.]  # true_positive false_positive false_negative
        self.correct_metric = [0., 0., 0.]

    def get_stats(self, reset=False, training=False):
        if training:
            return
        all_metrics = {}
        det_metrics = compute_metrics(self.detection_metric[0], self.detection_metric[1], self.detection_metric[2])
        cor_metrics = compute_metrics(self.correct_metric[0], self.correct_metric[1], self.correct_metric[2])
        precision, recall, f = (det_metrics[0]+cor_metrics[0])/2, (det_metrics[1]+cor_metrics[1])/2, \
                                (det_metrics[2]+cor_metrics[2])/2
        all_metrics["precision"] = precision
        all_metrics["recall"] = recall
        all_metrics["f"] = f
        all_metrics["accuracy"] = np.mean(self.accs)
        if reset:
            self.reset()
        print("Precision: {:.3f}| Recall: {:.3f} | F1 (micro): {:.3f}".format(precision, recall, f))
        return all_metrics

    def save_predictions_to_file(self, pred_tag_lists, probs, raw_tokens):
        save_file_path = os.path.join('./out', "test_output.txt")
        tokenizer = BertTokenizer.from_pretrained('data_vocab/vocab.txt')
        # print(f"INFO -> write predictions to {save_file_path}")
        output = open(save_file_path, "a", encoding='utf-8')
        for pred_label_item, data_item, prob in zip(pred_tag_lists, raw_tokens, probs):
            data_item = tokenizer.convert_ids_to_tokens(data_item.tolist())[:len(pred_label_item)]
            assert len(pred_label_item) == len(data_item)
            pred_label_item = [self.labels[tmp] for tmp in pred_label_item]
            # skip whole sentences if there no errors
            if all(pred_label == '$KEEP' for pred_label in pred_label_item):
                output.write(''.join(data_item[1:-1]) + "\n")
                continue
            edits = []
            for i in range(len(pred_label_item)): # because of START token
                # skip if there is no error
                if pred_label_item[i] in [UNK_TOKEN, PAD_TOKEN, '$KEEP'] or prob < self.min_error_probability:
                    continue
                else:
                    edit = [(i, i+1), pred_label_item[i]]
                    edits.append(edit)
            output.write(''.join(get_target_sent_by_edits(data_item, edits)[1:-1]) + "\n")  # 将edits变成target_sent
        output.close()


def get_target_sent_by_edits(source_tokens, edits):
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        (start, end), label = edit
        target_pos = start + shift_idx
        if label == "$DELETE":
            del target_tokens[target_pos]
            shift_idx -= 1
        elif label.startswith("$APPEND_"):
            word = label.replace("$APPEND_", "")
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif label.startswith("$REPLACE_"):
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word
        elif label.startswith("$SWAP_"):
            indexs = label.split('_')[1:]
            start_index = int(indexs[0])
            e_str = source_tokens[start_index + int(indexs[1]) - 2: start_index + int(indexs[1]) + 1]
            if start_index > len(target_tokens) - 1 or target_tokens[start_index - 1: start_index + 2] != source_tokens[start_index - 1: start_index + 2]:
                if source_tokens[start_index] in target_tokens[start_index + shift_idx - 1: target_pos]:
                    i = target_tokens[start_index + shift_idx - 1: target_pos].index(source_tokens[start_index])
                    start_index += i + shift_idx - 1
                    while target_tokens[start_index - 1: start_index + 1] != source_tokens[int(indexs[0]) - 1: int(indexs[0]) + 1] and \
                            source_tokens[int(indexs[0])] in target_tokens[start_index + 1: target_pos]:
                        start_index = target_tokens[start_index + 1: target_pos].index(
                            source_tokens[int(indexs[0])]) + start_index + 1
                if target_tokens[start_index - 1: start_index + 1] != source_tokens[int(indexs[0]) - 1: int(
                        indexs[0]) + 1] and target_tokens[start_index - 1: start_index + 2] != source_tokens[int(indexs[0]) - 1: int(indexs[0]) + 2]:
                    start_index = target_tokens[0:target_pos].index(source_tokens[int(indexs[0])])
                    while target_tokens[start_index - 1: start_index + 2] != source_tokens[int(indexs[0]) - 1: int(indexs[0]) + 2] and \
                            source_tokens[int(indexs[0])] in target_tokens[start_index + 1: target_pos]:
                        start_index = target_tokens[start_index + 1: target_pos].index(source_tokens[int(indexs[0])]) + start_index + 1
            if int(indexs[1]) > 1 and (start_index + int(indexs[1]) > len(target_tokens) or target_tokens[start_index + int(indexs[1]) - 2: start_index + int(indexs[1])] != e_str[:-1]):
                if target_tokens[target_pos - 1] == e_str[1] and (target_tokens[target_pos - 2] == e_str[0] or target_tokens[target_pos] == e_str[2]):
                    shift = target_pos - start_index - int(indexs[1])
                else:
                    i = target_tokens[start_index:target_pos][::-1].index(e_str[1])
                    if target_tokens[start_index:target_pos][::-1][i + 1] != e_str[0] and target_tokens[start_index:target_pos][::-1][i - 1] != e_str[2]:
                        i = target_tokens[start_index:target_pos][::-1].index(e_str[1], i + 1)
                    # i = ''.join(target_tokens[start_index:target_pos][::-1]).index(''.join(e_str[::-1]))
                    shift = target_pos - start_index - i - int(indexs[1])
            else:
                shift = 0
            label = f"$SWAP_{start_index}_{int(indexs[1]) + shift}"
            target_tokens[target_pos + 1: target_pos + 1] = [label]
            shift_idx += 1
    leveled_tokens = target_tokens[:]
    target_tokens = replace_swap_transforms(leveled_tokens)
    return target_tokens


if __name__ == '__main__':  # ["$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$SWAP_5_1", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$SWAP_10_1", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$KEEP", "$REPLACE_。"]
    argmax_labels = [[2, 2, 2, 2, 2, 2, 185, 2, 2, 2, 2, 314, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6]]
    gold_labels = [[2, 2, 2, 2, 2, 2, 185, 2, 2, 2, 2, 314, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6]]
    with open(os.path.join('./data_vocab/', "label_map.json"), "r", encoding='utf-8') as f:
        label_map = json.load(f)
    TC_evaluation_metric = TokenMetric(label_map["id2lable"])
    TC_evaluation_metric.update(torch.tensor(argmax_labels), torch.tensor(gold_labels))
    metrics = TC_evaluation_metric.get_stats()
    raw_tokens = [["亏损", "的", "主要", "原因", "是", "年", "2014", "橡胶", "市场行情", "一路", ",", "下滑", "销售", "单价", "较", "上年", "大幅", "降低", ",", "相应", "销售收入", "减少", "，"]]
    TC_evaluation_metric.save_predictions_to_file(gold_labels, 0.1, raw_tokens)
    print(metrics)
