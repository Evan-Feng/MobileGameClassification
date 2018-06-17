# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           nbsvm.py                                                           #
#  Description:    a PU-learning algorithm using Naive Baysian and SVM                #
#-------------------------------------------------------------------------------------#
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm, preprocessing
from scipy.sparse import issparse, vstack
import numpy as np
import argparse
import json
import pickle


class NBSVMClassifier:
    """
    A Classifier that learns from positive and unlabeled samples using Naive Bayasian
    and SVM.

    Only binary classification is supported. 

    This algorithm is based on a two-phase approach in traditional PU-learning. Denote
    the positive samples as P and unlabled samples as U. The algorithm consists of the 
    following steps:

    1. Build a Multinomial Naive Bayasian classifier using P and U.
    2. Use the classifier to classify U, those samples in U the are classified as
       negative form RN (reliable negative).
    3. Build a linear SVM classifier using P and RN.
    4. Use the SVM classifier to classify all samples.

    See https://www-new.comp.nus.edu.sg/~leews/publications/ICDM-03.pdf for details.

    Parameters
    ----------
    alpha: float, optional (default: 1.0)
        parameter alpha for the naive bayasian classifier
    C: float, optional (default: 0.3)
        parameter C for the linear svm classifer
    verbose: bool, optional (defult: False)
        increase verbosity
    """

    def __init__(self, alpha=1.0, C=0.3, verbose=False):
        self.alpha = alpha
        self.C = C
        self.verbose = verbose

    def fit_transform(self, X, y, X_clf=None):
        """
        Learn a new label vector using Naive Bayasian + SVM.

        X: array-like, shape (n_samples, n_features)
            supported types: python list, numpy.ndarray, scipy sparse matrix
        y: array-like, shape (n_samples,)
            supported types: python list, numpy.ndarray, scipy sparse matrix
        X_clf: array-like, shape (n_samples, n_features), optional (default: None)
            supported types: python list, numpy.ndarray, scipy sparse matrix

        Returns: np.array, shape (n_samples,)
        """

        # step 1: identify Reliable-Negative (RN) samples
        X = X if issparse(X) else np.array(X)
        y = y.toarray().reshape(-1) if issparse(y) else np.array(y)

        nb = MultinomialNB(alpha=self.alpha)
        pred = nb.fit(X, y).predict(X)
        pred = np.logical_or(pred, y)

        p = [i for i, b in enumerate(y) if b == 1]
        rn = [i for i, b in enumerate(pred) if b == False]

        if self.verbose:
            print('p=%d, rn=%d, sum=%d' % (len(p), len(rn), len(p) + len(rn)))

        # step2: train a classifier based on P and RN
        train_x = X[p + rn]
        train_y = np.concatenate((np.ones(len(p)), np.zeros(len(rn))), axis=0)
        return svm.LinearSVC(C=self.C).fit(train_x, train_y).predict(X_clf)


class OneVsRestWELL:
    """
    Given a binary classifier, form a multi-label classifer using binary relavance 
    strategy. 

    Parameters
    ----------
    estimator: an binary classifier that supports fit_transform method
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit_transform(self, X, y, X_clf=None):
        """
        Build N classifiers and perform multi-label classification using binary 
        relavance strategy. 

        X: array-like, shape (n_samples, n_features)
            supported types: python list, numpy.ndarray, scipy sparse matrix
        y: array-like, shape (n_samples)
            supported types: python list, numpy.ndarray, scipy sparse matrix

        Returns: numpy.ndarray, shape (n_samples, n_classes)
        """
        y = y.toarray() if issparse(y) else np.array(y)
        n_classes = np.max(y) + 1
        n_samples = X.shape[0] if issparse(X) else len(X)
        res = np.zeros((n_samples, n_classes))
        for label in range(n_classes):
            curr_y = np.equal(y, label, dtype=int)
            res[:, label] = self.estimator.fit_transform(X, curr_y, X_clf)
        return res.astype(int)


def main(args):
    """
    args: argparse.Namespace object

    Returns: None
    """
    # load data
    with open(args.infile, 'rb') as fin:
        x_train, y_train, x_test, y_test = pickle.load(fin)
    y_train, y_test = y_train.astype(int), y_test.astype(int)

    # concatenate train samples and test samples
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    # extract features
    X = CountVectorizer().fit_transform(x_train[:, -1])
    X_clf = TfidfVectorizer().fit_transform(x_train[:, -1])

    clf = OneVsRestWELL(NBSVMClassifier(verbose=True))

    res = clf.fit_transform(X, y_train, X_clf)
    not_classified = ~np.any(res, axis=1)
    Y = preprocessing.LabelBinarizer().fit_transform(y_train)
    res[not_classified] = Y[not_classified]

    # write the label matrix to target file
    with open('model/nbsvm.json', 'w') as fout:
        json.dump(res.tolist(), fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    main(args)
