#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters')
parser.add_argument('-csv', type=str, help='the file of metrics.csv')

args = parser.parse_args()


def bin_mean_std(csv_file, step=100, epochs=500):
    assert epochs % step == 0
    data = pd.read_csv(csv_file)
    count = epochs // step

    print('epoch', 'precision_mean', 'precision_std',
          'recall_mean', 'recall_std', 'f1_mean', 'f1_std',
          'f1-00_mean', 'f1-10_mean', 'f1-01_mean', 'f1-11_mean', sep=',')
    for i in range(count):
        df = data[step * i:step * (i + 1)]
        avg = df.mean()
        std = df.std()
        print(step * (i + 1),
              format(avg['precision'], '.4f'),
              format(std['precision'], '.4f'),
              format(avg['recall'], '.4f'),
              format(std['recall'], '.4f'),
              format(avg['f1'], '.4f'),
              format(std['f1'], '.4f'),
              format(avg['f1-00'], '.4f'),
              format(avg['f1-10'], '.4f'),
              format(avg['f1-01'], '.4f'),
              format(avg['f1-11'], '.4f'),
              sep=',')


if __name__ == '__main__':
    bin_mean_std(args.csv, 100)
