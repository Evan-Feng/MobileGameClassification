from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from scipy.sparse import issparse, vstack


class NBSVMClassifier:

    def __init__(self, alpha=1.0, C=0.3, verbose=False):
        self.alpha = alpha
        self.C = C
        self.verbose = verbose

    def fit(self, X, y):
        """
        X: array-like, shape (n_samples, n_features)
            supported types: python list, numpy.ndarray, scipy sparse matrix
        Y: array-like, shape (n_samples)
            supported types: python list, numpy.ndarray, scipy sparse matrix
        """
        is_sparse = issparse(X)
        X = X if is_sparse else np.array(X)
        y = y.toarray().reshape(-1) if issparse(y) else np.array(y)
        
        nb = MultinomialNB(alpha=self.alpha)
        pred = nb.fit(X, y).pred(X)
        pred = np.logical_or(pred, y)

        p = [i for i, b in enumerate(y) if b == 1]
        rn = [i for i, b in enumerate(pred) if b == False]

        if self.verbose:
            print('p=%d, rn=%d' % (len(p), len(rn)))

        if is_sparse:
            train_x = vstack((X[p], X[rn]), format='csr')
        else:
            train_x = np.concatenate((X[p], X[rn]), axis=0)

        train_y = [1] * len(p) + [0] * len(rn)

        svc = svm.LinearSVC(C=self.C)
        svc.fit(train_x, train_y)
        return svc.predict(X)


def main():
    pass



if __name__ == '__main__':
    main()