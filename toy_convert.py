import csv
import json
import random
import numpy as np
from spider import CATEGORYIES

N_EACH_CATE_TESTSET = 20


def gen_train_test_data(in_path):
    with open(in_path, 'r', encoding='utf-8') as fin:
        csvin = csv.reader(fin)
        x_train = [row for row in csvin]
    x_train = [[int(row[0]), ] + [float(x) if x else np.nan for x in row[1:]]
               for row in x_train[1:]]
    random.shuffle(x_train)
    print('Number of total samples: %d' % len(x_train))

    test_indices = []
    for cate_index in range(len(CATEGORYIES)):
        test_indices += [i for i, row in enumerate(x_train) if int(row[0]) == cate_index][
            :N_EACH_CATE_TESTSET]

    x_test = [x_train[i][1:] for i in test_indices]
    y_test = [x_train[i][0] for i in test_indices]
    y_train = [x_train[i][0]
               for i in range(len(x_train)) if i not in test_indices]
    x_train = [x_train[i][1:]
               for i in range(len(x_train)) if i not in test_indices]

    print('Number of train samples : %d' % len(x_train))
    print('Number of test samples : %d' % len(x_test))
    return x_train, y_train, x_test, y_test


def dump_data(dataset, save_path):
    with open(save_path, 'w', encoding='utf-8') as fout:
        json.dump(dataset, fout, indent=4)


def main():
    dataset = gen_train_test_data('toy.csv')
    dump_data(dataset, save_path='toy.json')


if __name__ == '__main__':
    main()
