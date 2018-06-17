# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           well.py                                                            #
#  Description:    a WEak-multi-Label-Learning algorithm                              #
#-------------------------------------------------------------------------------------#
from sklearn.metrics.pairwise import pairwise_kernels, cosine_similarity
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import cvxpy as cvx
import numpy as np
import argparse
import json
import pickle


class WELLClassifier:

    def __init__(self, alpha=100, beta=10, verbose=False):
        """
        alpha: int/float, optional (default 100)
        beta: int/float, optional (default 10)
        verbose: bool, optional (default False)

        Returns: None
        """
        self.alpha = alpha
        self.beta = beta

    def fit_transform(self, X, Y):
        """
        X: array-like, shape (m, d)
        Y: array-like, shape (m, q)

        Returns: numpy.ndarray, shape (m, q)
        """
        Y = np.array(Y)
        W = cosine_similarity(X)
        W = pairwise_kernels(X, metric='rbf')                       # size: (m, m)
        # print(W[:20, :20])
        # print(np.all(np.linalg.eigvals(W) > 0), '***')
        L = np.diag(np.sum(W, axis=1)) - W                 # size: (m, m)
        q = len(Y[0])                                      # size: ()
        m = len(X) if isinstance(X, list) else X.shape[0]  # size: ()

        F = []

        for k in range(q):
            Y_k = np.diag(Y[:, k])                         # size: (m,)
            mat0 = self.alpha * L + self.beta * Y_k        # size: (m, m)
            # print(mat0)
            # print(mat0.dtype)
            vec0 = 2 * self.beta * Y[:, k] - np.ones(m)    # size: (m,)
            # print(vec0.dtype)

            x = cvx.Variable(m)
            objective = cvx.Minimize(cvx.quad_form(x, mat0) - vec0 * x)
            constraints = [x >= 0., x <= 1.]
            problem = cvx.Problem(objective, constraints)
            problem.solve()

            candidates = sorted(x.value.tolist())[: -1]
            ans = float('inf')
            cal_obj = lambda x: (x.reshape((1, m)) @ mat0 @ x.reshape((m, 1))) - np.dot(vec0, x)
            for threshold in candidates:
                f = (x.value > threshold).astype('int')
                obj = float(cal_obj(f))
                if obj < ans:
                    res = f
                    ans = obj
            F.append(res.reshape(m, 1))
            print('.', end='', flush=True)
        print()
        F = np.concatenate(F, axis=1)
        F = np.logical_or(F, Y)
        return F.astype(int)


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

    candidates = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]  # 32 38
    target_rate = 0.08
    curr_min = 1

    print('------------------------------------------------')
    print('fitting for each of %d candidates' % len(candidates))
    print('target: %.6f' % target_rate)
    print()

    for alpha in candidates:
        clf = WELLClassifier(alpha=alpha, beta=10000)
        curr_res = clf.fit_transform(x_train, y_train)
        score = np.count_nonzero(curr_res) / curr_res.size
        print()
        print('[CV] alpha=%.2f   score=%.6f' % (alpha, score))

        if abs(score - target_rate) < curr_min and score >= 0.06:
            curr_min = abs(score - target_rate)
            res = curr_res
            best_param = alpha
    print()
    print('the best parameter is %s' % {'alpha': best_param})
    print('------------------------------------------------')
    print(res[:20])

    with open('model/well.json', 'w') as fout:
        json.dump(res.tolist(), fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    main(args)
