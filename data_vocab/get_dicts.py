import json, random
from tqdm import tqdm
from utils import PAD_TOKEN, UNK_TOKEN


def get_dict(freqs, prefix, min_count=0, num=0):
    words = sorted(freqs.items(), key=lambda d: d[1], reverse=True)
    words = [word for word, count in words][:num]
    # words = [word for word, count in words if count >= min_count]
    print(f'Starting dump {prefix} dict to file...')
    if prefix != "token":
        word2id = {}  # Padding with <PAD> Unknown with <UNK>
        id2word = [PAD_TOKEN, UNK_TOKEN] + words
        for Index, word in enumerate(id2word):
            word2id[word] = Index
        with open(prefix+'_map.json', "w", encoding='utf-8') as f:  # 209706
            json.dump({"id2"+prefix: id2word, prefix + "2id": word2id}, f, indent=6, ensure_ascii=False)
    else:
        id2word = [PAD_TOKEN, UNK_TOKEN, '[CLS]', '[SEP]', '[MASK]'] + words
        with open('vocab.txt', "w", encoding='utf-8') as f:
            for word in id2word[:23236]:  # 445957
                f.write(word + '\n')
    print(f'dump {prefix} dict finished\n the dict size is {len(id2word)}')


def get_dicts(data_path, label_freqs):  # 制作字典
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = json.load(f)
    for line in tqdm(lines):
        for lable in line['det_label']:
            label_freqs[lable] = label_freqs.get(lable, 0) + 1
    return label_freqs


def split_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = json.load(f)
    # assert len(list(set([str(s) for s in lines]))) == len(lines), 'need to del repeats'
    print(len(lines))  # 999482 1219984
    sentences = [eval(s) for s in list(set([str(s) for s in lines]))]
    print(len(sentences))  # 998700 1219171
    random.seed(2022)
    random.shuffle(sentences)

    with open('./dev.json', 'w', encoding='utf-8') as f:
        json.dump(sentences[:10000], f, ensure_ascii=False)

    with open('./train.json', 'w', encoding='utf-8') as f:
        json.dump(sentences[10000:], f, ensure_ascii=False)  # [::-1]


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    label_freqs = {}
    # label_freqs = get_dicts('./synthesis_tag.json', label_freqs)  # 48480 300:1111
    label_freqs = get_dicts('../../Preprocess/total_down_100_tag.json', label_freqs)
    get_dict(label_freqs, "label", num=1095)  # 0:339549 200:2273 300：1716 400:1406 500:1232 600:1097(num=1095)
    # get_dict(token_freqs, "token", 0) num = 实际_num-2
    # split_dataset('./synthesis_tag.json')
    split_dataset('../../Preprocess/total_down_100_tag.json')
