# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           cat.py                                                             #
#  Description:    concatenate app info into csv files, multi-processing is used to   #
#                  to speed up the procedure                                          #
#-------------------------------------------------------------------------------------#
import re
import os
import csv
import collections
import time
import multiprocessing as mp
from spider import CATEGORYIES
from pprint import pprint

N_PROCESS = 8

PATH = 'D:/Study/算法设计与分析/metadata_full/metadata_full/app_data/'

KEY_MAP = collections.OrderedDict({
    'Category': 'Category',
    'Package': 'Package',
    'Age': 'Age',
    'Developer': 'Developer',
    'Download': 'Installs',
    'Edition': 'Version',
    'System': 'Requires_Android',
    'Rating': 'Rating',
    'Rating_Num': 'Rating_Total',
    '1-Star_Rating_Num': 'Rating_1',
    '2-Star_Rating_Num': 'Rating_2',
    '3-Star_Rating_Num': 'Rating_3',
    '4-Star_Rating_Num': 'Rating_4',
    '5-Star_Rating_Num': 'Rating_5',
    'Name': 'Name',
    'Update_Time': 'Updated',
    'Price': 'Price',
    'Permission': 'Permission',
    'Description': 'Description',
})


def parse_information(path):
    info = {}
    with open(path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    for i, row in enumerate(lines):
        if not row.startswith('\t'):
            key = row.strip()
            if key in KEY_MAP:
                info[KEY_MAP[key]] = lines[i + 1].strip()
    return info


def parse_permission(path):
    with open(path, 'r', encoding='utf-8') as fin:
        res = fin.read().strip().replace('\n', ';')
    return res


def parse_description(path):
    with open(path, 'r', encoding='utf-8') as fin:
        res = fin.read().replace('\n', ' ')
    return res


def doit(package_list, pid):
    print('Process %d started...' % pid)
    print('Number of packages:', len(package_list))
    t0 = time.time()
    time.sleep(10)

    buffer = []
    for i, package in enumerate(package_list):
        try:
            target = PATH + package + '/' + os.listdir(PATH + package)[0]
            dic = parse_information(target + '/Information(eng).txt')
            if dic['Category'] not in CATEGORYIES and dic['Category'] != 'Role Playing':
                continue
            dic['Category'] = dic['Category'].replace(' ', '_')
            dic['Permission'] = parse_permission(
                target + '/Permission(eng).txt')
            dic['Description'] = parse_description(
                target + '/Description(eng).txt')
            dic['Package'] = package
        except Exception as e:
            print(e)
            continue
        print('[%d]%d' % (pid, i))
        buffer.append(dic)

    with open('raw/%d_complete_%d.csv' % (len(buffer), pid), 'w', encoding='utf-8', newline='') as fout:
        csvout = csv.DictWriter(fout, fieldnames=list(KEY_MAP.values()))
        csvout.writeheader()
        csvout.writerows(buffer)

    print('Process %d finished in %.2f seconds, terminating...' % (pid, time.time() - t0))


def main():
    package_list=os.listdir(PATH)
    neach=len(package_list) // N_PROCESS
    for i in range(N_PROCESS - 1):
        mp.Process(target=doit, args=(
            package_list[i * neach:(i + 1) * neach], i)).start()
    mp.Process(target=doit, args=(
        package_list[(N_PROCESS - 1) * neach:], N_PROCESS - 1)).start()


if __name__ == '__main__':
    main()
