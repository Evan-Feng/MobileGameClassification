import csv
import json
from spider import HEADERS, CATEGORYIES


def main():
    dic = {}

    for cate in CATEGORYIES:
        with open('raw/' + cate + '.csv', 'w', encoding='utf-8', newline='') as fout:
            csvout = csv.writer(fout)
            csvout.writerows([HEADERS['new']])
        dic[cate] = [0, 0]

    with open('spider_history.json', 'w', encoding='utf-8') as fout:
        json.dump(dic, fout, indent=4)

if __name__ == '__main__':
    main()
