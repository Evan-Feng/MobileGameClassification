from sklearn import svm, base, decomposition, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack, issparse
import numpy as np
import json
import random
import argparse
import pickle
from pprint import pprint
from crawl import CATEGORIES

from pprint import pprint

class RankJudge:

    def __init__(self, binary_classifier, q, verbose=0):
        self.base_clf = binary_classifier
        self.q = q
        self.verbose = verbose

    def fit(self, X, Y):
        is_sparse = issparse(X)

        self.clfs = [[base.clone(self.base_clf)
                      for _ in range(self.q - 1 - i)] for i in range(self.q)]

        for i in range(self.q):
            for j in range(i + 1, self.q):
                X_tmp, Y_tmp = [], []
                for xi, yi in zip(X, Y):
                    if yi[i] != yi[j]:
                        X_tmp.append(xi)
                        Y_tmp.append(int(yi[i] > yi[j]))
                    else:
                        X_tmp += [xi, xi]
                        Y_tmp += [0, 1]
                if is_sparse:
                    X_tmp = vstack(X_tmp, format='csr')
                if self.verbose >= 1:
                    print('.', end='', flush=True)
                if 0 in Y_tmp and 1 in Y_tmp:
                    self.clfs[i][j - i - 1].fit(X_tmp, Y_tmp)
                else:
                    self.clfs[i][j - i - 1] = Y_tmp[0]

    def predict(self, X):
        xlen = X.shape[0] if issparse(X) else len(X)

        res = [[0] * self.q for _ in range(xlen)]
        for i in range(self.q):
            for j in range(i + 1, self.q):
                if self.clfs[i][j - i - 1] in {0, 1}:
                    pred = [self.clfs[i][j - i - 1]] * xlen
                else:
                    pred = self.clfs[i][j - i - 1].predict(X)
                for index in range(xlen):
                    if pred[index] == 1:
                        res[index][i] += 1
                    else:
                        res[index][j] += 1
        return res


class MultiRankClassifier:

    def __init__(self, binary_classifier,  max_iters=20, kfold=3, verbose=0):
        self.clf = binary_classifier
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
            Y_.append(Y[ans:ans + n_each[i]])
            if is_sparse:
                X_c.append(
                    vstack((X[:ans], X[ans + n_each[i]:]), format='csr'))
            else:
                X_c.append(np.concatenate(
                    (X[:ans], X[ans + n_each[i]:]), axis=0))
            ans += n_each[i]

        return X_, Y_, X_c

    def fit(self, X, Y):
        if isinstance(Y, np.ndarray):
            Y = Y.tolist()

        xlen = X.shape[0] if issparse(X) else len(X)

        is_sparse = issparse(X)

        tmp = list(zip(range(xlen), X, Y))
        random.shuffle(tmp)
        old_index, X, Y = zip(*tmp)
        X, Y = list(X), list(Y)
        
        if is_sparse:
            X = vstack(X, format='csr')
        

        self.q = len(Y[0])
        X, Y, Xc = self._split_k(X, Y, self.kfold)
        rankj = RankJudge(self.clf, self.q, self.verbose)
        Y_next = [None] * self.kfold

        for iteration in range(self.max_iters):
            if self.verbose >= 1:
                print('[%d]' % iteration, end='', flush=True)
            for k in range(self.kfold):
                Yc = sum([Y[i] for i in range(self.kfold) if i != k], [])
                rankj.fit(Xc[k], Yc)
                Y_next[k] = rankj.predict(X[k])
            Y = Y_next

            Y_tmp = sum(Y, [])
            Y_old = [None] * len(Y_tmp)
            for i in range(len(Y_tmp)):
                Y_old[old_index[i]] = Y_tmp[i]
            with open('tmp/%d_%d.json' % (xlen, iteration), 'w') as fout:
                json.dump(sum(Y_old, []), fout, indent=4)

            if self.verbose >= 1:
                print()

        self.rankj = rankj

        return Y_old

    def predict(self, X):
        return self.rankj.predict(X)


def load_dataset(path):
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
    return data


def main(args):
    """
    args: argparse.Namespace object

    Returns: None
    """
    x_train, y_train, x_test, y_test = load_dataset(args.infile)

    # convert label to one-hot encoding
    lb = preprocessing.LabelBinarizer()
    y_train = lb.fit_transform(y_train.astype('int64')).tolist()
    y_test = lb.transform(y_test.astype('int64')).tolist()

    # extract tf-idf features
    tfidf = TfidfVectorizer()
    x_train = tfidf.fit_transform(x_train[:, -1])
    x_test = tfidf.transform(x_test[:, -1])

    clf = MultiRankClassifier(svm.LinearSVC(C=0.3), verbose=4)

    train_rank = clf.fit(x_train, y_train)
    test_rank = clf.predict(x_test)

    print(len(train_rank), len(y_train))
    print(len(test_rank), len(y_test))

    pprint(train_rank[:10])
    pprint(y_train[:10])

    pprint(test_rank[:10])
    pprint(y_test[:10])

    with open('model/rank', 'wb') as fout:
        pickle.dump([train_rank, test_rank], fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    main(args)
