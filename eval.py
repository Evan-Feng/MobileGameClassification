# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           eval.py                                                            #
#  Description:    convert binary label matrix into category names, combine them      #
#                  with original categories and package names for manual evaluation   #
#  Author:         fyl                                                                #
#-------------------------------------------------------------------------------------#
import numpy as np
import pickle
import csv
import json
import argparse
from crawl import CATEGORIES


def main(args):
    """
    args: argparse.Namespace object

    Returns: None
    """
    model = 'linear_well' if args.linear else 'kfold_well'

    # load labels
    with open('dataset/%s_complete' % args.dataset_id, 'rb') as fin:
        x_train, y_train, x_test, y_test = pickle.load(fin)
    Y = np.concatenate((y_train, y_test), axis=0)

    # load package names
    with open('dataset/%s_package.json' % args.dataset_id, 'r') as fin:
        packages = json.load(fin)
    packages = packages['train'] + packages['test']

    # load new labels
    with open('model/%s.json' % model, 'r') as fin:
        multi_labels = json.load(fin)

    size = len(Y)

    Y = [CATEGORIES[i] for i in Y]
    multi_labels = [' '.join([CATEGORIES[i] for i, j in enumerate(row) if j == 1])
                    for row in multi_labels]

    result = list(zip(packages, Y, multi_labels))

    if args.pretty_print:
        import prettytable
        table = prettytable.PrettyTable()
        table.field_names = ['Package', 'Original_Label', 'New_Labels']
        for row in result:
            table.add_row(row)
        with open('result/%s_%s_result.txt' % (args.dataset_id, model), 'w', newline='') as fout:
            fout.write(str(table))
    else:
        with open('result/%s_%s_result.csv' % (args.dataset_id, model), 'w', newline='') as fout:
            csvout = csv.writer(fout)
            csvout.writerow(['Package', 'Original_Label', 'New_Labels'])
            csvout.writerows(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_id',
                        help='specify the id of the dataset')
    parser.add_argument('-p', '--pretty_print',
                        action='store_true',
                        help='pretty-print the result to the target file')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--linear',
                       action='store_true',
                       help='generate result from LinearWELL model')
    group.add_argument('-k', '--kfold',
                       action='store_true',
                       help='generate result from KFoldWELL model')
    args = parser.parse_args()
    main(args)
