# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           kfold_well.py                                                      #
#  Description:    a WEak-multi-Label-Learning algorithm based on k-fold              #
#                  valiadation and binary relevance                                   #
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


class KFoldWELLClassifer:
    """
    KFoldWELL is a multi-label learning algorithm that learns new labels from
    existing weak labels. During each iteration, the algorithm first split
    training data into k partitions, then learn the label matrix of a certain
    partition from the other k - 1 partitions using binary relavance strategy.

    Parameters
    ----------
    estimator: sklearn estimator object
        the base estimator to be used in the one-vs-rest classification
    max_iters: int
        number of iterations
    kfold: int
        number of partitions to split the training data
    verbose: bool
        increase verbosity
    """

    def __init__(self, estimator,  max_iters=20, kfold=3, verbose=False):
        self.estimator = estimator
        self.max_iters = max_iters
        self.kfold = kfold
        self.verbose = verbose

    def fit_transform(self, X, Y):
        """
        Calculate a new label matrix with the same size as Y based on a k-fold
        method.

        X: array-like, shape (m, d)
            supported types: python list, numpy.ndarray, scipy sparse matrix
        Y: array-like, shape (m, q)
            supported types: python list, numpy.ndarray, scipy sparse matrix

        Returns: numpy.ndarray, shape (m, q)
        """
        if isinstance(Y, np.ndarray):
            Y = Y.tolist()
        elif issparse(Y):
            Y = Y.toarray().tolist()

        is_sparse = issparse(X)
        xlen = X.shape[0] if is_sparse else len(X)

        tmp = list(zip(range(xlen), X, Y))
        random.shuffle(tmp)
        old_index, X, Y = zip(*tmp)
        old_to_new = sorted(list(enumerate(old_index)), key=lambda x: x[1])
        old_to_new = [t[0] for t in old_to_new]
        X, Y = list(X), list(Y)

        X = vstack(X, format='csr') if is_sparse else np.array(X)
        Y = np.array(Y)

        self.q = len(Y[0])
        Y_next = Y.copy()
        kf = KFold(self.kfold, shuffle=False)
        clf = OneVsRestClassifier(self.estimator)

        for iteration in range(self.max_iters):
            print('[%d]' % iteration, end='', flush=True)

            for train_index, test_index in kf.split(X):
                clf.fit(X[train_index], Y[train_index])
                Y_next[test_index] = clf.predict(X[test_index])
                print('.', end='', flush=True)
            Y = np.logical_or(Y, Y_next).astype(int)
            print()

        return Y[old_to_new]


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
    y_train = lb.fit_transform(y_train.astype('int64')).tolist()

    # extract tf-idf features
    tfidf = TfidfVectorizer()
    x_train = tfidf.fit_transform(x_train[:, -1])

    # learn the new label matrix using KFoldWELL
    clf = KFoldWELLClassifer(svm.LinearSVC(
        C=0.3), max_iters=args.iters, kfold=args.kfold)

    res = clf.fit_transform(x_train, y_train)

    # write the label matrix to target file
    with open('model/kfold_well.json', 'w') as fout:
        json.dump(res.tolist(), fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    parser.add_argument('-k', '--kfold',
                        type=int,
                        default=3,
                        help='specify parameter kfold to be used')
    parser.add_argument('-t', '--iters',
                        type=int,
                        default=1,
                        help='specify the number of iterations')
    args = parser.parse_args()
    main(args)
