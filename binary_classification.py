# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           classify.py                                                        #
#  Description:    single-label classification based on linear SVM                    #
#-------------------------------------------------------------------------------------#
from sklearn import metrics, preprocessing, svm, decomposition
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse.csr import csr_matrix
import numpy as np
import pickle
import random
import argparse
import time


class MyCountVectorizer(CountVectorizer):
    """
    Extract the last column of the input matrix (which is app description) and
    simply pass it to CountVectorizer.

    Parameters
    ----------
    (identical to the parameters of CountVectorizer)
    """

    def __init__(self, input='content', encoding='utf-8', decode_error='strict',
                 strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r'(?u)\b\w\w +\b', ngram_range=(1, 1),
                 analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
                 binary=False, dtype=np.int64):

        return super().__init__(input=input, encoding=encoding, decode_error=decode_error,
                                strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor,
                                tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern,
                                ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df,
                                max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype)

    def fit(self, x, *args, **kwargs):
        return super().fit(x[:, -1].tolist(), *args, **kwargs)

    def fit_transform(self, x, *args, **kwargs):
        return super().fit_transform(x[:, -1].tolist(), *args, **kwargs)

    def transform(self, x, *args, **kwargs):
        return super().transform(x[:, -1].tolist(), *args, **kwargs)


class MyTfidfVectorizer(TfidfVectorizer):
    """
    Extract the last column of the input matrix (which is app description) and
    simply pass it to TfidfVectorizer.

    Parameters
    ----------
    (identical to the parameters of TfidfVectorizer)
    """

    def __init__(self, input='content', encoding='utf-8', decode_error='strict',
                 strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
                 analyzer='word', stop_words=None, token_pattern=r'(?u)\b\w\w+\b',
                 ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64, norm='l2', use_idf=True,
                 smooth_idf=True, sublinear_tf=False):

        return super().__init__(input=input, encoding=encoding, decode_error=decode_error,
                                strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor,
                                tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words,
                                token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df,
                                min_df=min_df, max_features=max_features, vocabulary=vocabulary,
                                binary=binary, dtype=dtype, norm=norm, use_idf=use_idf,
                                smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

    def fit(self, x, *args, **kwargs):
        return super().fit(x[:, -1].tolist(), *args, **kwargs)

    def fit_transform(self, x, *args, **kwargs):
        return super().fit_transform(x[:, -1].tolist(), *args, **kwargs)

    def transform(self, x, *args, **kwargs):
        return super().transform(x[:, -1].tolist(), *args, **kwargs)


class MyScaler(preprocessing.StandardScaler):
    """
    Standardize features by removing the mean and scaling to unit variance. Both
    fit_tranform and tranform method return sparse matrix in csr format.

    Parameters
    ----------
    (null)

    Attributes
    ----------
    mean_: numpy.array, shape (n_features)
        the mean value for each features in the input matrix
    std_: numpy.array, shape (n_features)
        the standard deviation for each features in the input matrix
    """

    def __init__(self, identical=False):
        self.mean_ = None
        self.std_ = None
        self.epsilon = 0.000001
        self.identical = identical

    def fit(self, X, y=None):
        X = X[:, :-1].astype('float64')
        self.mean_ = np.mean(X, axis=0) if not self.identical else 0
        self.std_ = np.std(X, axis=0) if not self.identical else 1
        return self

    def fit_tranform(self, X, y=None):
        X = X[:, :-1].astype('float64')
        self.mean_ = np.mean(X, axis=0) if not self.identical else 0
        self.std_ = np.std(X, axis=0) if not self.identical else 1
        res = (X - self.mean_) / (self.std_ + self.epsilon)
        return csr_matrix(res)

    def transform(self, X, y=None):
        X = X[:, :-1].astype('float64')
        res = (X - self.mean_) / (self.std_ + self.epsilon)
        return csr_matrix(res)


def main(args):
    """
    Grid-search over different parameters for a linear SVM classifier using
    a 3-fold cross valiadation.

    args: argparse.Namespace object

    Returns: None
    """

    # load dataset
    with open(args.infile, 'rb') as fin:
        x_train, y_train, x_test, y_test = pickle.load(fin)

    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    random_index = np.random.permutation(len(x_train))
    x_train = np.array(x_train[random_index])
    y_train = np.array(y_train[random_index])


    # combined different features
    feature_extractors = [
        # ('general', MyScaler(False)),
        ('wordcount', MyCountVectorizer(ngram_range=(1, 1))),
        # ('tfidf', MyTfidfVectorizer(stop_words='english')),
    ]
    combined_feature = FeatureUnion(feature_extractors)

    for clf in [MultinomialNB(alpha=1.0)]:
        print(clf)
        for c in range(17):
            estimators = [('feature', combined_feature),
                          ('clf', clf)]
            pipeline = Pipeline(estimators)

            y_train_tmp = (y_train == c).astype(int)
            y_test_tmp = (y_test == c).astype(int)

            pipeline.fit(x_train, y_train_tmp)
            print('Category: %d    Accuracy: %.4f' % (c, pipeline.score(x_test, y_test_tmp)))

            print(metrics.confusion_matrix(y_test_tmp, pipeline.predict(x_test)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    main(args)
