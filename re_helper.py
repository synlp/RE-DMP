import json


def get_label_list(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) == 0 or line == '\n':
                continue
            splits = line.split('\t')
            relation = splits[2]
            if relation not in label_list:
                label_list.append(relation)

    return label_list


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)
