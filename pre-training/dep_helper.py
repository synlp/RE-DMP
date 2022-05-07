import json

def get_word2id(train_data_path):
    word2id = {'<PAD>': 0}
    index = 1
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            word = splits[1]
            if word not in word2id:
                word2id[word] = index
                index += 1
    return word2id


def get_label_list(label_path):
    label_list = ['<UNK>']

    with open(label_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            label_list.append(line)

    assert 'amod' in label_list

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)


def load_vocab(file_path, min_freq=1):
    vocab2id = {}
    index = 0
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        split = line.split()
        w = split[0]
        c = int(split[1])

        if c >= min_freq and w not in vocab2id:
            vocab2id[w] = index
            index += 1

    return vocab2id
