# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           linear_well.py                                                     #
#  Description:    a weak-multi-label learning algorithm based on a linear method     #
#  Author:         fyl                                                                #
#-------------------------------------------------------------------------------------#
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse, vstack
import cvxpy as cvx
import numpy as np
import argparse
import json
import pickle


class LinearWELL:
    """
    LinearWELL is a  multi-label learning machine based on a linear method.
    This algorithm uses a first-order strategy, which means it does not consider
    the corelation between different labels.

    Given a matrix of shape (m, d) and a label matrix of shape (m, q), a new
    label matrix of shape (m, q) will be calculated. Denote the new matrix
    as F. The algorithm solve each column of F seperately.

    For each label, the classifier first partition samples into UNLABLED
    and LABLED. Thus the solution can be represented by a m0-dimension vector
    f, where m0 = len(UNLABLED). The pairwise similarity matrix between UNLABLED
    and LABLED is calculated using cosine similaity
    ( SIM(x, y) = <x, y> / (||x||*||y||) ), denoted W. f would be the solution
    to the following optimization problem:

                    min   -||f * W|| + gamma * ||f||
                     f
                            s.t.   0 <= f <= 1

    where gamma is a hyper-parameter. f tends to be sparse if gamma is large,
    while f is likely to be an all-ones vector if gamma is zero.

    Parameters
    ----------
    gamma: float, optional (default 10)
        The only hyper-parameter of this classification model. Output tends to
        be sparse if gamma is large. On the contrary, output is likely to be a
        all-ones matrix if gamma is zero.

    verbose: bool, optional (default False)
        Passed to cvxpy solver. Prints runtime information when solving
        optimization problem.
    """

    def __init__(self, gamma=10, verbose=False):
        self.gamma = gamma
        self.verbose = verbose

    def fit_transform(self, X, Y):
        """
        Calculate a new label matrix with the same size as Y based on the cosine
        similarity matrix of X.

        X: array-like, shape (m, d)
            supported types: python list, numpy.ndarray, scipy sparse matrix
        Y: array-like, shape (m, q)
            supported types: python list, numpy.ndarray, scipy sparse matrix

        Returns: numpy.ndarray, shape (m, q)
        """
        q = len(Y[0])
        m = len(X) if isinstance(X, list) else X.shape[0]

        is_sparse = issparse(X)
        if issparse(Y):
            Y = Y.toarray()

        F = []
        for label in range(q):
            labeled = [x for x, y in zip(X, Y) if y[label] == 1]
            index, unlabeled = zip(
                *[(j, x) for j, (x, y) in enumerate(zip(X, Y)) if y[label] == 0])
            m0 = len(unlabeled)
            m1 = len(labeled)

            if is_sparse:
                unlabeled = vstack(unlabeled, format='csr')
                labeled = vstack(labeled, format='csr')

            if m1 == 0 or m0 == 0:
                print('Runtime Warning: label %d all %s'(
                    label, 'positive' if m0 == 0 else 'negative'))
                F.append([int(m0 == 0)] * m)
                continue

            W = cosine_similarity(unlabeled, labeled)
            x = cvx.Variable(m0)
            objective = cvx.Minimize(-cvx.sum(x.T * W) +
                                     self.gamma * (cvx.sum(x)))
            constraints = [x >= 0, x <= 1]

            problem = cvx.Problem(objective, constraints)
            problem.solve(verbose=self.verbose)

            ans = np.ones(m, dtype='int')
            for i in range(m0):
                ans[index[i]] = x.value[i]
            F.append(ans.reshape(m, 1))

        return np.concatenate(F, axis=1)


def main(args):
    """
    Load training data from args.infile, extract tf-idf features from train
    sample, and learn a new label matrix based an samples and the weak label
    matrix. The result will be written to "./model/linear_well.json".

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

    # learn the new label matrix using LinearWELL
    param_grid = [10, 14, 16, 18, 12]
    for gamma in param_grid:
        clf = LinearWELL(gamma=gamma, verbose=(args.verbose >= 1))
        res = clf.fit_transform(x_train[:], y_train[:])
        print('gamma: %.2f   nonzero rate: %.2f' %
              (gamma, np.count_nonzero(res) / res.size))

    # write the matrix to target path
    with open('model/linear_well.json', 'w') as fout:
        json.dump(res.tolist(), fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    parser.add_argument('-v', '--verbose',
                        action='count',
                        default=0,
                        help='increase verbosity')
    args = parser.parse_args()
    main(args)
