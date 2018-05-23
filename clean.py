import pandas as pd
import numpy as np
import collections
import pickle
import csv
import random
import argparse
from spider import CATEGORYIES

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


def extract_inap_prod(series):
    series = series.str.replace('???', '$0.0 - $0.0', regex=False)
    extracted = series.str.extract(
        r'(?P<Inapp_Lower>\d+\.\d+|\d+).*?(?P<Inapp_Upper>\d+\.\d+|\d+)')
    return extracted.astype('float32')


def extract_description(series):
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    extracted = vectorizer.fit_transform(series).toarray()
    extracted = pd.DataFrame(extracted, dtype='int8')
    return extracted


MAX_DESCRIPTION_LENGTH = 1000
MIN_DESCRIPTION_LENGTH = 200


def clean(df):
    df.drop_duplicates('Package', keep='first', inplace=True)

    puncs = ':/\\,.[]}{()*-_$"\'+?:@#^&!='
    df['Description'] = df['Description'].apply(lambda x: x.encode(
        'utf-8').decode('ascii', 'ignore').lower().translate({ord(c): None for c in puncs}))
    bad_lines = [i for i, row in enumerate(
        df.loc[:, 'Description']) if row.count(' ') > len(row) // 4]
    df.drop(bad_lines, axis=0, inplace=True)
    df.index = range(len(df))

    df['Description'] = df['Description'].str.replace(r' +', ' ').apply(lambda x: x if x.find(
        ' ', MAX_DESCRIPTION_LENGTH) == -1 else x[:x.find(' ', MAX_DESCRIPTION_LENGTH)])
    too_short = [i for i, row in enumerate(
        df.loc[:, 'Description']) if len(row) < MIN_DESCRIPTION_LENGTH]
    df.drop(too_short, axis=0, inplace=True)

    df.index = range(len(df))
    df.fillna('0.0', inplace=True)
    return df


EXTRACTORS = collections.OrderedDict({
    'Category': lambda x: x.apply(CATEGORYIES.index).astype('int64'),
    'Updated': lambda x: ((pd.Timestamp('2018-5-19') - pd.to_datetime(x)) // pd.Timedelta('1d')).astype('int64'),
    'Price': lambda x: x.str.replace('(Install|Free)', '0.0').str.extract(r'(?P<Price>\d+\.\d+|\d+)').astype('float64'),
    'Requires_Android': lambda x: x.str.extract(r'(?P<Requires_Android>\d+\.\d+|\d+)').astype('float64'),
    'Size': lambda x: x.str.replace(r'.*?k', '0.8M').str.extract(r'(?P<Size>\d+\.\d+|\d+)').astype('float64'),
    'Installs': lambda x: x.str.replace(',', '').str.extract(r'(?P<Installs>\d+)').astype('int64'),
    'Age': lambda x: x.apply(lambda x: AGE_MAP[x]).astype('int64'),
    'Version': lambda x: x.str.extract(r'(?P<Version>\d+\.\d+)').astype('float64'),
    'Rating': lambda x: x.astype('float64'),
    'Rating_Total': lambda x: x.str.replace(',', '').str.extract(r'(?P<Rating_Total>\d+)').astype('float64'),
    'Rating_1': lambda x: x.str.replace(',', '').str.extract(r'(?P<Rating_1>\d+)').astype('float64'),
    'Rating_2': lambda x: x.str.replace(',', '').str.extract(r'(?P<Rating_2>\d+)').astype('float64'),
    'Rating_3': lambda x: x.str.replace(',', '').str.extract(r'(?P<Rating_3>\d+)').astype('float64'),
    'Rating_4': lambda x: x.str.replace(',', '').str.extract(r'(?P<Rating_4>\d+)').astype('float64'),
    'Rating_5': lambda x: x.str.replace(',', '').str.extract(r'(?P<Rating_5>\d+)').astype('float64'),
    'Content_Feature': extract_content_feature,
    'Permission': extract_permission,
    'Inapp_Products': extract_inap_prod,
    'Description': lambda x: x,
})


def extract(df):
    df = pd.concat([EXTRACTORS[k](df[k])
                    for k in EXTRACTORS if k in df.columns], axis=1)
    df.fillna(df.mean(), inplace=True)
    return df


def sample(df):
    grouped = df.groupby('Category')
    n_each_class = min([len(grp[1]) for grp in grouped]) // 10 * 10
    df = pd.concat([grp[1].sample(n_each_class) for grp in grouped], axis=0)
    df.index = range(len(df))
    return df


def split(df):
    grouped = df.groupby('Category')
    train = pd.concat([grp[1][20:] for grp in grouped], axis=0).values
    test = pd.concat([grp[1][:20] for grp in grouped], axis=0).values
    return train[:, 1:], train[:, 0].astype('int32'), test[:, 1:], test[:, 0].astype('int32')


def gen_fasttext_dataset(csvpath):
    with open(csvpath, 'r', encoding='utf-8') as fin:
        csvin = csv.DictReader(fin)
        dataset = ['__label__' + CATEGORYIES[int(row['Category'])] +
                   ' ' + row['Description']for row in csvin]
    random.shuffle(dataset)
    with open('dataset/train.txt', 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(dataset[1000:]))
    with open('dataset/test.txt', 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(dataset[:1000]))


def concat_csv(csv_paths, savepath):
    df = pd.concat([pd.read_csv(f) for f in csv_paths])
    df.drop_duplicates('Package', keep='first', inplace=True)
    df.to_csv(savepath, index=False)


def main(args):
    df = pd.read_csv(args.infile)
    print('cleaning data...')
    df = clean(df)
    print('extracting features...')
    df = extract(df)
    if args.balance:
        print('sampling...')
        df = sample(df)
    res = split(df)
    print('dumping...')
    with open('dataset/%d_complete' % len(res[0]), 'wb') as fout:
        pickle.dump(res, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile',
                        help='the csv file that contains raw package info',
                        default='raw/complete.csv')
    parser.add_argument('--balance',
                        help='each category',
                        action='store_true')
    args = parser.parse_args()
    main(args)