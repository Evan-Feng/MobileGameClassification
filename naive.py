# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           naive.py                                                           #
#  Description:    a naive method for multi-label learning with weak labels using     #
#                  One-vs-Rest SVM                                                    #
#-------------------------------------------------------------------------------------#
from sklearn import svm, preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from scipy.sparse import vstack, issparse
import numpy as np
import json
import random
import argparse
import pickle
from crawl import CATEGORIES
from pprint import pprint


def main(args):
    """
    args: argparse.Namespace object

    Returns: None
    """
    # load data
    with open(args.infile, 'rb') as fin:
        x_train, y_train, x_test, y_test = pickle.load(fin)

    # concatenate train samples and test samples
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    # convert label to one-hot encoding
    lb = preprocessing.LabelBinarizer()
    y_train = lb.fit_transform(y_train.astype('int64'))

    # extract tf-idf features
    tfidf = TfidfVectorizer()
    x_train = tfidf.fit_transform(x_train[:, -1])

    # learn the new label matrix using KFoldWELL
    clf = OneVsRestClassifier(svm.LinearSVC(C=0.3))

    res = clf.fit(x_train, y_train).predict(x_train)
    res = np.logical_or(res, y_train).astype(int)

    # write the label matrix to target file
    with open('model/naive.json', 'w') as fout:
        json.dump(res.tolist(), fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    main(args)
