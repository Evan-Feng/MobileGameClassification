# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           extract.py                                                         #
#  Description:    clean raw data, remove unicode characters and punctuations,        #
#                  extract app description and split it into train and test           #
#                  subsets                                                            #
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
from langdetect import detect


MAX_DESCRIPTION_LENGTH = 3000
MIN_DESCRIPTION_LENGTH = 800  # 800 1200


def clean(df):
    """
    df: pandas.DataFrame object

    Returns: pandas.DataFrame object
    """
    df.drop_duplicates('Package', keep='first', inplace=True)
    df.index = range(len(df))

    foreign = []
    for i, row in enumerate(df['Description']):
        print(i, len(foreign))
        try:
            if detect(row) != 'en':
                foreign.append(i)
        except Exception:
            foreign.append(i)
    print(len(foreign), 'not english')
    df.drop(foreign, axis=0, inplace=True)
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
    too_short = [index for index in too_short if df.Package[
        index] not in IMPORTANT_PACKAGES]
    df.drop(too_short, axis=0, inplace=True)
    df.index = range(len(df))
    return df

CONTENT_FEATURE_MAP = {
    "Alcohol Reference": "Alcohol",
    "Alcohol and Tobacco Reference": "Alcohol, Tobacco",
    "Blood": "Blood",
    "Blood and Gore": "Blood",
    "Cartoon Violence": "Cartoon_Violence",
    "Comic Mischief": "Mischief",
    "Crude Humor": "Crude_Humor",
    "Diverse Content: Discretion Advised": "Discretion_Advised",
    "Drug Reference": "Drug",
    "Drug and Alcohol Reference": "Alcohol, Drug",
    "Extreme Violence": "Extreme_Violence",
    "Fantasy Violence": "Cartoon_Violence",
    "Fear": "Horror",
    "Gambling": "Gambling",
    "Horror": "Horror",
    "Implied Violence": "Implied_Violence",
    "Intense Violence": "Extreme_Violence",
    "Language": "Language",
    "Learn More": "Unrated",
    "Mild Blood": "Mild_Blood",
    "Mild Fantasy Violence": "Mild_Violence",
    "Mild Language": "Language",
    "Mild Swearing": "Language",
    "Mild Violence": "Mild_Violence",
    "Moderate Violence": "Mild_Violence",
    "Nudity": "Sexual",
    "Partial Nudity": "Sexual",
    "Sexual Content": "Sexual",
    "Sexual Innuendo": "Sexual",
    "Sexual Themes": "Sexual",
    "Simulated Gambling": "Gambling",
    "Strong Language": "Language",
    "Strong Sexual Content": "Sexual",
    "Strong Violence": "Extreme_Violence",
    "Suggestive Themes": "Suggestive_Themes",
    "Tobacco Reference": "Tobacco",
    "Use of Alcohol": "Alcohol",
    "Use of Alcohol and Tobacco": "Alcohol, Tobacco",
    "Use of Alcohol/Tobacco": "Alcohol, Tobacco",
    "Use of Drugs": "Drug",
    "Use of Drugs and Alcohol": "Drug, Alcohol",
    "Use of Tobacco": "Tobacco",
    "Violence": "Violence",
    "Violent References": "Violence",
    "Warning \u2013 content has not yet been rated. Unrated apps may potentially contain content appropriate for mature audiences only.": "Unrated"
}

CONTENT_FEATURE_KEYS = {'Alcohol', 'Blood', 'Cartoon_Violence', 'Crude_Humor',
                        'Discretion_Advised', 'Drug', 'Extreme_Violence', 'Gambling', 'Horror',
                        'Implied_Violence', 'Language', 'Mild_Blood', 'Mild_Violence',
                        'Mischief', 'Sexual', 'Suggestive_Themes', 'Tobacco', 'Unrated',
                        'Violence'}

PERMISSION_KEYS = {
    'Google Play license check', 'access Bluetooth settings',
    'access USB storage filesystem', 'approximate location (network-based)',
    'change network connectivity', 'change your audio settings',
    'connect and disconnect from Wi-Fi', 'control vibration',
    'disable your screen lock', 'draw over other apps',
    'find accounts on the device', 'install shortcuts',
    'modify or delete the contents of your USB storage',
    'modify system settings', 'pair with Bluetooth devices',
    'precise location (GPS and network-based)',
    'prevent device from sleeping', 'read Google service configuration',
    'read phone status and identity',
    'read the contents of your USB storage', 'receive data from Internet',
    'record audio', 'retrieve running apps', 'run at startup',
    'take pictures and videos', 'use accounts on the device',
    'view Wi-Fi connections', 'view network connections',
}

AGE_MAP = {
    'Everyone':      0,
    'Everyone 10+':  3,
    'Mature 17+':    7,
    'Rated for 12+': 4,
    'Rated for 16+': 6,
    'Rated for 18+': 8,
    'Rated for 3+':  1,
    'Rated for 7+':  2,
    'Teen':          5,
    'Unrated':       0,
}


def extract_content_feature(series):
    def cf_map(s):
        s = ', %s,' % s
        for k, v in CONTENT_FEATURE_MAP.items():
            s = s.replace(', %s,' % k, ', %s,' % v)
        return s[2:-1]

    extracted = series.apply(cf_map).str.get_dummies(sep=', ')
    extracted.drop(set(extracted.columns) -
                   CONTENT_FEATURE_KEYS, axis=1, inplace=True)
    return extracted.astype('int8')


def extract_permission(series):
    extracted = series.str.get_dummies(sep=';')
    extracted.drop(set(extracted.columns) -
                   PERMISSION_KEYS, axis=1, inplace=True)
    return extracted.astype('int8')


EXTRACTORS = collections.OrderedDict({
    'Package': lambda x: x,
    'Category': lambda x: x.apply(CATEGORIES.index).astype('int64'),
    'Installs': lambda x: x.str.replace(',', '').str.extract(r'(?P<Installs>\d+)').astype('float64').fillna(0.0),
    'Content_Feature': extract_content_feature,
    'Permission': extract_permission,
    'Name': lambda x: x.apply(lambda x: ''.join(list(filter(lambda x: x.isalpha() or x == ' ', str(x).lower())))),
    'Description': lambda x: x,
})


def extract(df):
    """
    df: pandas.DataFrame object

    Returns: pandas.DataFrame object
    """
    df = pd.concat([EXTRACTORS[k](df[k])
                    for k in EXTRACTORS if k in df.columns], axis=1)
    df['Name'].fillna('', inplace=True)
    df['Description'].fillna('', inplace=True)
    df['Description'] = (df['Name'] + ' ') * 5 + df['Description']
    df.drop(['Name'], inplace=True, axis=1)
    return df


IMPORTANT_PACKAGES = ['com.yodo1.rodeo.safari',
                      'com.ubisoft.hungrysharkworld',
                      'com.devolver.downwell',
                      'com.nianticlabs.pokemongo',
                      'com.yodo1.rodeo.safari',
                      'com.eline.neveralonemobile',
                      'com.turbochilli.rollingsky',
                      'com.blizzard.wtcg.hearthstone',
                      'com.devolver.reigns',
                      'com.yodo1.rodeo.safari',
                      'com.outerminds.tubular',
                      'com.devolver.downwell',
                      'com.Seriously.Forever',
                      'com.Zeeppo.GuitarBand',
                      'com.youmusic.magictiles',
                      'com.gismart.realpianofree',
                      'game.puzzle.blockpuzzle',
                      'com.protey.doors_challenge2',
                      'game2048.b2048game.twozerofoureight2048.game',
                      'com.yodo1.rodeo.safari',
                      'com.FireproofStudios.TheRoom3',
                      'com.martinmagni.mekorama',
                      'com.combineinc.streetracing.driftthreeD',
                      'com.silevel.mountainclimbstunt',
                      'com.wordsmobile.RealBikeRacing',
                      'com.outerminds.tubular',
                      'role-com.nexon.hit.global',
                      'com.springloaded.thelastvikings',
                      'com.ea.game.starwarscapital_row',
                      'com.scopely.headshot ',
                      'com.martinmagni.mekorama',
                      'com.martinmagni.mekorama',
                      'com.kongregate.mobile.thetrail.google',
                      'com.outerminds.tubular',
                      'com.ea.gp.nbamobile',
                      'com.supercell.clashroyale&uo=4&ct=appcards',
                      'com.epicwaronline.ms',
                      'com.igg.android.lordsmobile',
                      'com.tencent.ig',
                      'com.netease.ko',
                      'com.robtopx.geometryjumplite',
                      'com.robtopx.geometryjump',
                      'com.ea.game.pvzfree_row',
                      'com.zeptolab.ctr.ads',
                      'com.mojang.minecraftpe',
                      'com.ngame.allstar.eu',
                      'com.cmplay.dancingline',
                      'com.rovio.angrybirds',
                      'com.mobile.legends',
                      'com.ustwo.monumentvalley',
                      'com.tomorrowcorporation.humanresourcemachine']


def sample(df, n_each_class=0, important=IMPORTANT_PACKAGES):
    """
    df: pandas.DataFrame object

    Returns: pandas.DataFrame object
    """
    grouped = df.groupby('Category')
    if n_each_class == 0:
        n_each_class = min([len(grp[1]) for grp in grouped]) // 10 * 10
    df = pd.concat([grp[1].sort_values('Installs', ascending=False)[
                   :n_each_class] for grp in grouped] + [df[df.Package.isin(important)]], axis=0)
    df.drop_duplicates('Package', inplace=True)
    df.drop(['Installs'], inplace=True, axis=1)
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
