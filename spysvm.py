# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           nbsvm.py                                                           #
#  Description:    a PU-learning algorithm using spy technique and SVM                #
#-------------------------------------------------------------------------------------#
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm, preprocessing
from scipy.sparse import issparse, vstack
import numpy as np
import random
import argparse
import json
import pickle


class SpySVMClassifier:
    """
    A Classifier that learns from positive and unlabeled samples using Spy technique
    and SVM.

    Only binary classification is supported. 

    This algorithm is based on a two-phase approach in traditional PU-learning. Denote
    the positive samples as P and unlabled samples as U. The algorithm consists of the 
    following steps:

    1. Randomly select a set S (spies) from P.
    2. Build a Multinomial Naive Bayasian classifier using P and (S ∪ U).
    3. Determine a probability threshold using the probabilities assigned to S.
    4. The samples in U with probability smaller than the threshold form RN (reliable 
       negative).
    5. Build a linear SVM classifier using P and RN.
    6. Use the SVM classifier to classify all samples.

    See https://www-new.comp.nus.edu.sg/~leews/publications/ICDM-03.pdf for details.

    Parameters
    ----------
    alpha: float, optional (default: 1.0)
        parameter alpha for the naive bayasian classifier
    C: float, optional (default: 0.3)
        parameter C for the linear svm classifer
    S: float, optional (default 0.15)
        S% of positive samples will be selected as spies
    noise: float, optional (default 0.2)
        noise% of positive samples are considered noise, used to determine the threshold

    verbose: bool, optional (defult: False)
        increase verbosity
    """

    def __init__(self, alpha=1.0, C=0.3, S=0.15, noise=0.2, verbose=False):
        self.alpha = alpha
        self.C = C
        self.S = S
        self.verbose = verbose
        self.noise = noise

    def fit_transform(self, X, y, X_clf=None):
        """
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

        p = [i for i, c in enumerate(y) if c == 1]
        spies = random.sample(p, int(len(p) * self.S))

        y_tmp = y.copy()
        y_tmp[spies] = 0
        nb = MultinomialNB(alpha=self.alpha).fit(X, y_tmp)
        pred = nb.predict_proba(X)[:, 1]
        threshold = np.sort(pred[spies])[int(len(spies) * self.noise)]
        pred = pred >= threshold
        rn = [i for i, b in enumerate(pred) if b == False and i not in p]

        if self.verbose:
            print('p=%d, rn=%d, sum=%d, conflict=%d' %
                  (len(p), len(rn), len(p) + len(rn), len(set(rn) & set(p))))

        # step2: train a classifier based on P and RN
        if issparse(X_clf):
            train_x = vstack((X_clf[p], X_clf[rn]), format='csr')
        else:
            X_clf = np.array(X_clf)
            train_x = np.concatenate((X_clf[p], X_clf[rn]), axis=0)
        train_y = [1] * len(p) + [0] * len(rn)

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
        res = []
        n_classes = np.max(y) + 1
        n_samples = X.shape[0] if issparse(X) else len(X)
        for label in range(n_classes):
            curr_y = np.equal(y, label, dtype=int)
            res.append(self.estimator.fit_transform(
                X, curr_y, X_clf).reshape((n_samples, 1)))

        return np.concatenate(res, axis=1)


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

    clf = OneVsRestWELL(SpySVMClassifier(verbose=True))

    res = clf.fit_transform(X, y_train, X_clf)
    not_classified = ~np.any(res, axis=1)
    Y = preprocessing.LabelBinarizer().fit_transform(y_train)
    res[not_classified] = Y[not_classified]

    # write the label matrix to target file
    with open('model/spysvm.json', 'w') as fout:
        json.dump(res.tolist(), fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    main(args)
