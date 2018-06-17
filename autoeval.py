from sklearn import preprocessing
import csv
import collections
import numpy as np 
import argparse
import json
import pickle
from crawl import CATEGORIES
import prettytable

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

def eval_labeldistri(model):
    with open('model/%s.json' % model, 'r') as fin:
        multi_labels = json.load(fin)
    print(collections.Counter([row.count(1) for row in multi_labels]))
    return np.array([' '.join([CATEGORIES[i] for i, j in enumerate(row) if j == 1])
                        for row in multi_labels])

def eval_errorlabel(model, original):
    original = preprocessing.LabelBinarizer().fit_transform(original)
    with open('model/%s.json' % model, 'r') as fin:
        multi_labels = np.array(json.load(fin))
    error = ~np.any(np.logical_and(original, multi_labels), axis=1)
    return error.astype(int)


def main(args):
    model_list = ['linear_well', 'kfold_well', 'nbsvm', 'spysvm']

    # load labels
    with open('dataset/%s_complete' % args.dataset_id, 'rb') as fin:
        x_train, y_train, x_test, y_test = pickle.load(fin)
    original = np.concatenate((y_train, y_test), axis=0)
    ori_cates = np.array([CATEGORIES[i] for i in original])

    # load package names
    with open('dataset/%s_package.json' % args.dataset_id, 'r') as fin:
        packages = json.load(fin)
    packages = np.array(packages['train'] + packages['test'])

    columns = [packages, ori_cates]
    errors = np.zeros(len(original))
    for model in ['nbsvm', 'spysvm']:
        error = eval_errorlabel(model, original)
        columns.append(eval_labeldistri(model))
        errors = np.logical_or(errors, error)
    
    res = np.concatenate([col.reshape((-1, 1)) for col in columns], axis=1)[errors]

    write_result(res, False, ['Package', 'Original_labels', 'nbsvm', 'spysvm'], 'result/error')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_id',
                        help='specify the id of the dataset')
    args = parser.parse_args()
    main(args)