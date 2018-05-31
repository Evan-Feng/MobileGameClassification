# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           linear_well.py                                                     #
#  Description:    multi-label learning with weak label based on a linear method      #
#  Author:         fyl                                                                #
#-------------------------------------------------------------------------------------#
from sklearn.metrics.pairwise import pairwise_kernels, cosine_similarity
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse
import cvxpy as cvx
import numpy as np
import argparse
import json
import pickle


class LinearWELL:
    """
    LinearWELL is a  multi-label learning machine based on a linear method.
    This algorithm use a first-order strategy and does not consider the
    corelation between different labels.

    Given a matrix of shape (m, d) and a label matrix of shape (m, q), a new
    label matrix of shape (m, q) will be calculated. Denote the new matrix
    as F. We solves each column of F seperately.


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
    while f is likely to be a all-ones vector if gamma is zero.

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
        Calculates a new label matrix with the same size as Y based on the cosine
        similarity matrix of X.

        X: array-like, shape (m, d)
            supported types: python list, numpy.ndarray, scipy sparse matrix
        Y: array-like, shape (m, q)
            supported types: python list, numpy.ndarray, scipy sparse matrix

        Returns: numpy.ndarray, shape (m, q)
        """
        q = len(Y[0])
        m = len(X) if isinstance(X, list) else X.shape[0]
        to_nparray = lambda x: x.toarray() if issparse(x) else np.array(x)
        X = to_nparray(X)
        Y = to_nparray(Y)

        F = []
        for label in range(q):
            labeled = [x for x, y in zip(X, Y) if y[label] == 1]
            index, unlabeled = zip(
                *[(j, x) for j, (x, y) in enumerate(zip(X, Y)) if y[label] == 0])
            m0 = len(unlabeled)
            m1 = len(labeled)

            if m1 == 0 or m0 == 0:
                print('Runtime Warning: label %d all %s'(
                    label, 'positive' if m0 == 0 else 'negative'))
                F.append([int(m0 == 0)] * m)
                continue
            W = cosine_similarity(unlabeled, labeled)

            x = cvx.Variable(m0)
            objective = cvx.Minimize(-cvx.sum(x.T * W) +
                                     self.gamma * (np.ones(m0) * x))
            constraints = [x >= 0, x <= 1]

            problem = cvx.Problem(objective, constraints)
            problem.solve(verbose=(self.verbose >= 1))

            ans = np.ones(m, dtype='int')
            for i in range(m0):
                ans[index[i]] = x.value[i]
            F.append(ans.reshape(m, 1))

        return np.concatenate(F, axis=1)


def main(args):
    """
    Load training data from args.infile, extract tf-idf features from train
    sample, and learn a new label matrix based an samples and the weak label
    matrix. The result would be written to "./model/linear_well.json".

    args: argparse.Namespace object

    Returns: None
    """
    with open(args.infile, 'rb') as fin:
        x_train, y_train, x_test, y_test = pickle.load(fin)

    # convert label to one-hot encoding
    lb = preprocessing.LabelBinarizer()
    y_train = lb.fit_transform(y_train.astype('int64')).tolist()
    y_test = lb.transform(y_test.astype('int64')).tolist()

    # extract tf-idf features
    tfidf = TfidfVectorizer()
    x_train = tfidf.fit_transform(x_train[:, -1])
    x_test = tfidf.transform(x_test[:, -1])

    param_grid = [14]
    for gamma in param_grid:
        clf = LinearWELL(gamma=gamma, verbose=args.verbose)
        res = clf.fit_transform(x_train[:], y_train[:])
        print('gamma: %.2f   nonzero rate: %.2f' %
              (gamma, np.count_nonzero(res) / res.size))

    with open('model/linear_well.json', 'w') as fout:
        json.dump(res.tolist(), fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    parser.add_argument('-v', '--verbose',
                        action='count',
                        help='increase verbosity')
    args = parser.parse_args()
    main(args)
