import csv
from spider import HEADERS, CATEGORYIES


def main():
    for cate in CATEGORYIES:
        with open('raw/' + cate + '.csv', 'w', encoding='utf-8', newline='') as fout:
            csvout = csv.writer(fout)
            csvout.writerows([HEADERS])

if __name__ == '__main__':
    main()