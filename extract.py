# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           extract.py                                                         #
#  Description:    clean raw data, remove unicode characters and punctuations,        #
#                  extract app description and split it into train and test           #
#                  subsets                                                            #
#  Author:         fyl                                                                #
#-------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import collections
import pickle
import csv
import json
import random
import argparse
from crawl import CATEGORIES


MAX_DESCRIPTION_LENGTH = 3000
MIN_DESCRIPTION_LENGTH = 1200  # 800


def clean(df):
    """
    df: pandas.DataFrame object

    Returns: pandas.DataFrame object
    """
    df.drop_duplicates('Package', keep='first', inplace=True)
    df.index = range(len(df))

    puncs = ':/\\,.[]}{()*-_$"\'+?:@#^&!='
    df['Description'] = df['Description'].apply(lambda x: x.encode(
        'utf-8').decode('ascii', 'ignore').lower().translate({ord(c): None for c in puncs}))
    bad_lines = [i for i, row in enumerate(
        df.loc[:, 'Description']) if row.count(' ') > len(row) // 4]
    df.drop(bad_lines, axis=0, inplace=True)
    df.index = range(len(df))

    df['Description'] = df['Description'].str.replace(r' +', ' ').str.replace(r' http.*? ', ' ').apply(lambda x: x if x.find(
        ' ', MAX_DESCRIPTION_LENGTH) == -1 else x[:x.find(' ', MAX_DESCRIPTION_LENGTH)])
    too_short = [i for i, row in enumerate(
        df.loc[:, 'Description']) if len(row) < MIN_DESCRIPTION_LENGTH]
    df.drop(too_short, axis=0, inplace=True)
    df.index = range(len(df))
    return df


EXTRACTORS = collections.OrderedDict({
    'Package': lambda x: x,
    'Category': lambda x: x.apply(CATEGORIES.index).astype('int64'),
    'Installs': lambda x: x.str.replace(',', '').str.extract(r'(?P<Installs>\d+)').astype('float64'),
    'Description': lambda x: x,
})


def extract(df):
    """
    df: pandas.DataFrame object

    Returns: pandas.DataFrame object
    """
    df = pd.concat([EXTRACTORS[k](df[k])
                    for k in EXTRACTORS if k in df.columns], axis=1)
    df.fillna(0.0, inplace=True)
    df.sort_values('Installs', ascending=False, inplace=True)
    return df


def sample(df, n_each_class=0):
    """
    df: pandas.DataFrame object

    Returns: pandas.DataFrame object
    """
    grouped = df.groupby('Category')
    if n_each_class == 0:
        n_each_class = min([len(grp[1]) for grp in grouped]) // 10 * 10
    df = pd.concat([grp[1][:n_each_class] for grp in grouped], axis=0)
    df.index = range(len(df))
    return df


def split(df):
    """
    df: pandas.DataFrame object

    Returns: numpy.array, numpy.array, numpy.ndarray, numpy.array, numpy.ndarray, numpy.array
        train package list, test package list,
        train samples, train labels, 
        test samples, test labels
    """
    grouped = df.groupby('Category')
    train = pd.concat([grp[1][20:] for grp in grouped], axis=0).values
    test = pd.concat([grp[1][:20] for grp in grouped], axis=0).values
    return train[:, 0], test[:, 0], train[:, 2:], train[:, 1].astype('int32'), test[:, 2:], test[:, 1].astype('int32')


def main(args):
    """
    args: argparse.Namespace object

    Returns: None
    """
    df = pd.read_csv(args.infile)
    if args.verbose:
        print(df.count())
        print(df.groupby('Category').size())
    print('cleaning data...')
    df = clean(df)
    print('extracting features...')
    df = extract(df)
    if args.balance or args.n > 0:
        print('sampling...')
        df = sample(df, args.n)
    if args.verbose:
        print(df.groupby('Category').size())
    train_pkg, test_pkg, *res = split(df)
    print('dumping...')
    with open('dataset/%d_complete' % len(res[0]), 'wb') as fout:
        pickle.dump(res, fout)
    with open('dataset/%d_package.json' % len(res[0]), 'w') as fout:
        json.dump({'train': train_pkg.tolist(),
                   'test': test_pkg.tolist()}, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        help='the csv file that contains raw package info')
    parser.add_argument('-v', '--verbose',
                        help='increase verbosity level',
                        action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-n',
                       help='samples each category',
                       type=int,
                       default=0)
    group.add_argument('--balance',
                       help='each category',
                       action='store_true')

    args = parser.parse_args()
    main(args)
