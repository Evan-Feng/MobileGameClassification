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


class KWELLClassifer:

    def __init__(self, estimator,  max_iters=20, kfold=3, verbose=0):
        self.estimator = estimator
        self.max_iters = max_iters
        self.kfold = kfold
        self.verbose = verbose

    def _split_k(self, X, Y, k):
        is_sparse = issparse(X)

        if not is_sparse:
            X = np.array(X)

        if X.shape[0] < k:
            raise ValueError(
                'not enough samples to be split into %d partitions' % k)
        elif X.shape[0] != len(Y):
            raise ValueError(
                'number of samples do not match, %d in X and %d in Y' % (X.shape[0], len(Y)))
        div, mod = divmod(X.shape[0], k)
        n_each = [div + 1] * (mod) + [div] * (k - mod)
        ans = 0

        X_, Y_, X_c = [], [], []
        for i in range(k):
            X_.append(X[ans:ans + n_each[i]])
            Y_.append(np.array(Y[ans:ans + n_each[i]]))
            if is_sparse:
                X_c.append(
                    vstack((X[:ans], X[ans + n_each[i]:]), format='csr'))
            else:
                X_c.append(np.concatenate(
                    (X[:ans], X[ans + n_each[i]:]), axis=0))
            ans += n_each[i]

        return X_, Y_, X_c

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
        X, Y = list(X), list(Y)
        
        if is_sparse:
            X = vstack(X, format='csr')

        self.q = len(Y[0])
        X, Y, Xc = self._split_k(X, Y, self.kfold)
        Y_next = [None] * self.kfold

        clf = OneVsRestClassifier(self.estimator)

        for iteration in range(self.max_iters):
            if self.verbose >= 1:
                print('[%d]' % iteration, end='', flush=True)

            for k in range(self.kfold):
                Yc = np.concatenate([Y[i] for i in range(self.kfold) if i != k], axis=0)
                clf.fit(Xc[k], np.array(Yc))
                Y_next[k] = clf.predict(X[k])
            
            for k in range(self.kfold):
                Y[k] = np.logical_or(Y[k], Y_next[k]).astype(int)

            Y_tmp = np.concatenate(Y, axis=0)
            Y_old = [None] * len(Y_tmp)
            for i in range(len(Y_tmp)):
                Y_old[old_index[i]] = Y_tmp[i]
            Y_old = np.array(Y_old)

            with open('tmp/%d_%d.json' % (xlen, iteration), 'w') as fout:
                json.dump(Y_old.tolist(), fout, indent=4)

            if self.verbose >= 1:
                print()

        return Y_old


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

    clf = KWELLClassifer(svm.LinearSVC(C=0.3), max_iters=1)

    res = clf.fit_transform(x_train, y_train)

    with open('model/kfold_well.json', 'w') as fout:
        json.dump(res.tolist(), fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    main(args)
