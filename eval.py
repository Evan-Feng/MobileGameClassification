# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           eval.py                                                            #
#  Description:    convert binary label matrix into category names, combine them      #
#                  with original categories and package names for manual evaluation   #
#-------------------------------------------------------------------------------------#
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import pickle
import csv
import json
import argparse
import prettytable
import collections
from functools import reduce
from crawl import CATEGORIES

MODELS = ['well',
          'linear_well',
          'kfold_well',
          'nbsvm',
          'spysvm',
          'iterspysvm']


def write_result(result, pretty_print, field_names, savepath):
    if pretty_print:
        table = prettytable.PrettyTable()
        table.field_names = field_names
        for row in result:
            table.add_row(row)
        with open(savepath + '.txt', 'w', newline='') as fout:
            fout.write(str(table))
    else:
        with open(savepath + '.csv', 'w', newline='') as fout:
            csvout = csv.writer(fout)
            csvout.writerow(field_names)
            csvout.writerows(result)


def load_packages(dataset_id):
    """
    dataset_id: str

    Returns: np.array
    """
    with open('dataset/%s_package.json' % dataset_id, 'r') as fin:
        packages = json.load(fin)
    return np.array(packages['train'] + packages['test'])


def load_original_labels(dataset_id):
    """
    dataset_id: str

    Returns: dict{str: np.array, str: np.array}
    """
    with open('dataset/%s_complete' % dataset_id, 'rb') as fin:
        x_train, y_train, x_test, y_test = pickle.load(fin)
    labels = np.concatenate((y_train, y_test), axis=0)
    cates = np.array([CATEGORIES[x] for x in labels])
    return {'original_vec': labels, 'original_cates': cates}


def load_new_labels(model):
    """
    model: str

    Returns: dict{str: np.ndarray, str: np.array}
    """
    with open('model/%s.json' % model, 'r') as fin:
        labels = np.array(json.load(fin))
    cates = np.array([' '.join([CATEGORIES[i]
                                for i, j in enumerate(row) if j == 1]) for row in labels])
    return {'%s_mat' % model: labels, '%s_cates' % model: cates}


def load_result(dataset_id, model_list):
    """
    dataset_id: str
    model_list: list[str]

    Returns: dict
    """
    res = {'packages': load_packages(dataset_id)}
    res.update(load_original_labels(dataset_id))
    for model in model_list:
        res.update(load_new_labels(model))
    return res


def write_individual_result(dic, model, pretty_print, savepath):
    """
    dic: dict
    model: str
    pretty_print: bool
    savepath: str

    Returns: None
    """
    field_names = ['Package', 'Original_Label', model.upper()]
    result = list(
        zip(dic['packages'], dic['original_cates'], dic['%s_cates' % model]))
    write_result(result, pretty_print, field_names, savepath)


def write_merged_result(dic, model_list, pretty_print, savepath):
    """
    dic: dict
    model_list: list[str]
    pretty_print: bool
    savepath: str

    Returns: None
    """
    field_names = ['Package', 'Original_Label', *[s.upper()
                                                  for s in model_list]]
    result = list(
        zip(dic['packages'], dic['original_cates'], *[dic['%s_cates' % s] for s in model_list]))
    write_result(result, pretty_print, field_names, savepath)


def plot_lable_counts(dic, model_list):
    """
    dic: dict
    model_list: list[str]

    Returns: None
    """
    counts = [collections.Counter(np.count_nonzero(
        dic['%s_mat' % s], axis=1).tolist()) for s in model_list]
    maxcount = max(sum(map(list, counts), []))
    counts = [[x.get(n, 0) for n in range(1, maxcount + 1)] for x in counts]

    fig, ax = plt.subplots()
    index = np.arange(1, maxcount + 1)
    bar_width = 0.8 / len(model_list)

    colors = ['g', 'c', 'y', 'r', 'b']

    for i, model in enumerate(model_list):
        ax.bar(index + bar_width * i, counts[i], bar_width,
               color=colors[i],
               alpha=0.8,
               # linewidth=0.5,
               # edgecolor='k',
               label=model.upper())

    ax.set_xlabel('Number of Labels')
    ax.set_ylabel('Counts')
    ax.set_xticks(index + bar_width * (len(model_list) // 2))
    ax.set_xticklabels(list(range(1, maxcount + 1)))
    ax.legend()
    fig.tight_layout()
    plt.show()


def write_error_result(dic, model_list, pretty_print, savepath):
    """
    dic: dict
    model_list: list[str]
    pretty_print: bool
    savepath: str

    Returns: None
    """

    field_names = ['Package', 'Original_Label', *[s.upper()
                                                  for s in model_list]]
    original = LabelBinarizer().fit_transform(dic['original_vec'])
    new = reduce(np.logical_and, [dic['%s_mat' % s] for s in model_list])
    seletor = ~np.any(np.logical_and(original, new), axis=1)

    # print(np.count_nonzero(~np.any(np.logical_and(dic['nbsvm_mat'], original), axis=1)))
    # print(np.count_nonzero(~np.any(np.logical_and(dic['spysvm_mat'], original), axis=1)))

    # return

    result = list(zip(dic['packages'][seletor], dic['original_cates'][
                  seletor], *[dic['%s_cates' % s][seletor] for s in model_list]))
    write_result(result, pretty_print, field_names, savepath)


def main(args):
    """
    args: argparse.Namespace object

    Returns: None
    """
    model_list = [k for k in MODELS if getattr(args, k)]
    result = load_result(args.dataset_id, model_list)

    if args.count:
        plot_lable_counts(result, model_list)
    elif args.error:
        write_error_result(result, model_list, args.pretty_print,
                           'result/%s_error_result' % args.dataset_id)
    elif args.merge:
        write_merged_result(result, model_list, args.pretty_print,
                            'result/%s_merged_result' % args.dataset_id)
    else:
        for model in model_list:
            write_individual_result(
                result, model, args.pretty_print, 'result/%s_%s_result' % (args.dataset_id, model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_id',
                        help='specify the id of the dataset')
    parser.add_argument('-p', '--pretty_print',
                        action='store_true',
                        help='pretty-print the result to the target file')
    parser.add_argument('--merge',
                        action='store_true',
                        help='merge results of specified models')
    parser.add_argument('--error',
                        action='store_true',
                        help='generate mislabled result')
    parser.add_argument('--count',
                        action='store_true',
                        help='plot label counts')
    parser.add_argument('-l', '--linear_well',
                        action='store_true',
                        help='generate result from LinearWELL model')
    parser.add_argument('-k', '--kfold_well',
                        action='store_true',
                        help='generate result from KFoldWELL model')
    parser.add_argument('-w', '--well',
                        action='store_true',
                        help='generate result from WELL model')
    parser.add_argument('-n', '--nbsvm',
                        action='store_true',
                        help='generate result from NBSVM model')
    parser.add_argument('-s', '--spysvm',
                        action='store_true',
                        help='generate result from SpySVM model')
    parser.add_argument('-i', '--iterspysvm',
                        action='store_true',
                        help='generate result from Iter-SpySVM model')
    args = parser.parse_args()
    main(args)
