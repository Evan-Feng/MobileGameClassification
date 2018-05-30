import json
import argparse
from pprint import pprint


def main(args):
    iteration = 0
    res = []
    while True:
        try:
            with open('tmp/%d_%d' % (args.dataset_id, iteration), 'r') as fin:
                res.append(json.load(fin)[args.index])
        except OSError:
            break
    pprint(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_id', type=int)
    parser.add_argument('index', type=int)
    args = parser.parse_args()
    main(args)