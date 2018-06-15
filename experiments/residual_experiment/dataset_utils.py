


def get_class_to_id():
    H = {}
    with open('class_to_id.csv', 'r') as f:
        for line in f:
            v, k = line.strip().split(',')
            H[k] = int(v)
    return H


def get_train_val():
    train_lst = []
    with open('train.csv', 'r') as f:
        for line in f:
            elems = line.strip().split(',')
            train_lst.append((elems[0], elems[1], elems[2], elems[3]))
    val_lst = []
    with open('validation.csv', 'r') as f:
        for line in f:
            elems = line.strip().split(',')
            val_lst.append((elems[0], elems[1], elems[2], elems[3]))
    return train_lst, val_lst


def get_labels(class_to_id, train_lst, val_lst):
    train_labels = [0] * len(train_lst)
    val_labels = [0] * len(val_lst)
    for el in train_lst:
        ID = int(el[0])
        sign = el[2]
        train_labels[ID] = class_to_id[sign]
    for el in val_lst:
        ID = int(el[0])
        sign = el[2]
        val_labels[ID] = class_to_id[sign]
    return train_labels, val_labels
