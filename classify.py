# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           classify.py                                                        #
#  Description:    single-label classification based on linear SVM                    #
#  Author:         fyl                                                                #
#-------------------------------------------------------------------------------------#
from sklearn import metrics, preprocessing, svm, decomposition
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pickle
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
    Standardize features by removing the mean and scaling to unit variance.

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

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.epsilon = 0.0001

    def fit(self, X, y=None):
        X = X[:, :-1].astype('float64')
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def fit_tranform(self, X, y=None):
        X = X[:, :-1].astype('float64')
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return (X - self.mean_) / (self.std_ + self.epsilon)

    def transform(self, X, y=None):
        X = X[:, :-1].astype('float64')
        return (X - self.mean_) / (self.std_ + self.epsilon)


def main(args):
    """
    Grid-search over different parameters for a linear SVM classifier using
    a 3-fold cross valiadation.

    args: argparse.Namespace object

    Returns: None
    """

    # load dataset
    with open(path, 'rb') as fin:
        x_train, y_train, x_test, y_test = lpickle.load(fin)

    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    # combined different features
    feature_extractors = [
        # ('general', MyScaler()),
        # ('wordcount', MyCountVectorizer(ngram_range=(1, 1))),
        ('tfidf', MyTfidfVectorizer()),
    ]
    combined_feature = FeatureUnion(feature_extractors)

    estimators = [('feature', combined_feature),
                  ('clf', svm.LinearSVC())]
    pipeline = Pipeline(estimators)

    # parameters to search
    param_grid = [
        {
            'clf': [svm.LinearSVC()],
            'clf__C': [1, 0.5, 0.3, 0.1, 0.03],
        },
    ]

    # start training
    t0 = time.time()
    grid = GridSearchCV(pipeline, param_grid=param_grid, verbose=4)
    grid.fit(x_train, y_train)

    print()
    print('done in %.2f seconds' % (time.time() - t0))
    print()
    print('train accuracy: %.2f%%' % (100 * grid.score(x_train, y_train)))
    print('test accuracy: %.2f%%' % (100 * grid.score(x_test, y_test)))
    print()
    print('the best parameters are:', grid.best_params_)
    print()
    print('confusion matrix:')
    print(metrics.confusion_matrix(y_test, grid.predict(x_test)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='file path of the dataset')
    args = parser.parse_args()
    main(args)
