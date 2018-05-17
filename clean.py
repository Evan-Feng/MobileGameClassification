import csv
import collections
from spider import CATEGORYIES

AGES = ['Unrated',
        'Everyone',
        'Rated for 3+',
        'Everyone 10+',
        'Teen',
        'Rated for 16+',
        'Mature 17+',
        ]


def inspect_csv_by_category(directory, category):
    with open(directory + '/' + category + '.csv', 'r', encoding='utf-8') as fin:
        csvin = csv.DictReader(fin)
        dics = [row for row in csvin]

    is_missing_value = lambda dic: '' in dic.values()
    is_unicode = lambda dic: any(ord(c) >= 128 for c in ''.join(dic.values()))

    total = len(dics)
    missing_info = len([row for row in dics if is_missing_value(row)])
    contains_unicode = len([row for row in dics if is_unicode(row)])
    complete = total - \
        len([row for row in dics if is_missing_value(row) or is_unicode(row)])

    # missing_counts = collections.defaultdict(int)
    # missing_key = collections.defaultdict(int)
    # for row in dics:
    #     missing_counts[len([v for v in row.values() if v == ''])] += 1
    #     for key in row:
    #         if row[key] == '':
    #             missing_key[key] += 1

    print('Category: %s, Total: %d, Missing Info: %d, Contains Unicode: %d, Complete: %d' %
          (category, total, missing_info, contains_unicode, complete))
    # print(missing_counts, '\n', missing_key, '\n')


def main():
    for cate in CATEGORYIES:
        inspect_csv_by_category('raw', cate)

if __name__ == '__main__':
    main()
