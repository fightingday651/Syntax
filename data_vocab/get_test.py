import json
from ddparser import DDParser
ddp = DDParser()
dicts = []
with open('s_test.txt', mode='r', encoding='UTF-8') as f:
    for sentence in f:
        sentence = sentence.lstrip('\ufeff').rstrip('\n')
        assert '\ufeff' not in sentence and '\n' not in sentence
        r_dict = ddp.parse(sentence)[0]
        r_dict['original_text'] = r_dict.pop('word')
        dicts.append(r_dict)

with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(dicts, f, ensure_ascii=False)