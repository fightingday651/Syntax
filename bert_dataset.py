import os, torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from utils import DataProcessor, tokenize, compute_alignment, wp_aligned_dep_parse, FeaturizedDataLoader,\
    handle_long_sent_parses, get_boundary_sensitive_alignment


class BertDataset(Dataset):
    def __init__(self, vocab_path, max_seq_length, prefix):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.vocab_file = os.path.join(vocab_path, 'vocab.txt')
        self.prefix = prefix
        self.sequence_a_segment_id = 0
        self.cls_token_segment_id = 0
        self.pad_token_segment_id = 0
        self.pad_token = 0
        self.mask_padding_with_zero = True
        self.examples = []

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        raw_tokens, est_len = example.text_a, 0
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file)
        # tokenizer.add_tokens(['‘', '’', '“', '”', '…', '—'])
        tokens, word_idxs, subword_token_len = tokenize(raw_tokens, tokenizer)
        # if len(tokens) > self.max_seq_length - 2:  # Account for [CLS] and [SEP] with "- 2"
        #     ratio = len(tokens) / len(raw_tokens)
        #     est_len = int((self.max_seq_length - 2) / ratio)-1
        #     raw_tokens = raw_tokens[:est_len]
        #     tokens, word_idxs, subword_token_len = tokenize(raw_tokens, tokenizer)
        assert len(tokens) <= self.max_seq_length - 2, (tokens, raw_tokens)
        alignment = compute_alignment(word_idxs, subword_token_len)
        # Add [SEP] token at ending
        tokens = tokens + [tokenizer.sep_token]
        segment_ids = [self.sequence_a_segment_id] * len(tokens)
        subword_token_len = subword_token_len + [1]
        word_idxs.append(len(tokens) - 1)
        # Add [CLS] token at beginning
        tokens = [tokenizer.cls_token] + tokens
        segment_ids = [self.cls_token_segment_id] + segment_ids
        subword_token_len = [1] + subword_token_len
        word_idxs = [0] + [i + 1 for i in word_idxs]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将句子转为ids
        # Increase alignment position by offset 1 i.e. because of the [CLS] token at the start
        alignment = [[val + 1 for val in list_] for list_ in alignment]
        wp_rows, align_sizes = get_boundary_sensitive_alignment(input_ids, raw_tokens, alignment)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to
        input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        input_ids = input_ids + ([self.pad_token] * padding_length)
        # dep_rel = dep_rel + ([self.pad_token] * (self.max_seq_length - len(dep_rel)))
        input_mask = input_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([self.pad_token_segment_id] * padding_length)
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        dep_head, dep_rel = example.dep_head, example.dep_rel
        if est_len:
            dep_head, dep_rel = handle_long_sent_parses(dep_head, dep_rel, est_len)
        dep_head, dep_rel = wp_aligned_dep_parse(word_idxs, subword_token_len, dep_head, dep_rel)
        if self.prefix == 'test':
            return input_ids, input_mask, segment_ids, wp_rows, align_sizes, len(tokens), dep_head, dep_rel, raw_tokens
        labels, d_tags = [0] + example.labels + [0], [0] + example.d_tags + [0]
        # padding_length = self.max_seq_length - len(labels)  # test,后面不需要
        # labels = labels + ([self.pad_token] * padding_length)
        # d_tags = d_tags + ([self.pad_token] * padding_length)
        # assert len(labels) == self.max_seq_length
        # assert len(d_tags) == self.max_seq_length
        assert len(dep_rel) == len(tokens)
        assert len(dep_head) == len(tokens)
        assert max(dep_head) <= len(tokens)
        return input_ids, input_mask, segment_ids, wp_rows, align_sizes, len(tokens), dep_head, dep_rel, labels, d_tags


def unit_test():
    prefix = 'train'
    processor = DataProcessor('./data_vocab')
    train_examples = processor.get_examples('dev')
    dataset = BertDataset('./data_vocab', max_seq_length=100, prefix=prefix)
    dataset.examples = train_examples
    if prefix == "train":
        data_generator = torch.Generator()
        data_generator.manual_seed(1000)
        data_sampler = RandomSampler(dataset, generator=data_generator)
    else:
        data_sampler = SequentialSampler(dataset)
    # sampler option is mutually exclusive with shuffle
    dataloader = FeaturizedDataLoader(dataset, gpus=1, batch_size=10, sampler=data_sampler)
    for batch in dataloader:
        d_list = b_len(**batch)
        assert all([x != 1 for x in d_list[9]])
        print(len(d_list))


def b_len(input_ids, wp_token_mask=None, token_type_ids=None, wp_rows=None, align_sizes=None, seq_len=None,
          labels=None, d_tags=None, dep_head=None, dep_rel=None):
    d_list = [input_ids, wp_token_mask, token_type_ids, wp_rows, align_sizes, seq_len, labels, d_tags, dep_head,
            dep_rel]
    return d_list


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    unit_test()
