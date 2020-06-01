'''
Comine predictions from training base classifiers based on one feature set.
Author: Linhua (Alex) Wang
Date:  1/02/2019
'''

from glob import glob
import gzip
from os.path import abspath, exists, isdir
from os import listdir
from sys import argv
from common import load_arff_headers, load_properties
from pandas import concat, read_csv


def combine_individual(path):
    dirnames = sorted(filter(isdir, glob('%s/weka.classifiers.*' % path)))
    # for fold in range(fold_count):
    for fold in fold_values:
        dirname_dfs = []
        for dirname in dirnames:
            classifier = dirname.split('.')[-1]
            nested_fold_dfs = []
            for nested_fold in range(nested_fold_count):
                bag_dfs = []
                for bag in range(bag_count):
                    filename = '%s/validation-%s-%02i-%02i.csv.gz' % (dirname, fold, nested_fold, bag)
                    try:
                        df = read_csv(filename, skiprows=1, index_col=[0, 1], compression='gzip', engine='python')
                        df = df[['prediction']]
                        df.rename(columns={'prediction': '%s.%s' % (classifier, bag)}, inplace=True)
                        bag_dfs.append(df)
                    except:
                        print('file not existed or crashed %s' % filename)
                nested_fold_dfs.append(concat(bag_dfs, axis=1))
            dirname_dfs.append(concat(nested_fold_dfs, axis=0))
        concat(dirname_dfs, axis=1).sort_index().to_csv('%s/validation-%s.csv.gz' % (path, fold), compression='gzip')

    # for fold in range(fold_count):
    for fold in fold_values:
        dirname_dfs = []
        for dirname in dirnames:
            classifier = dirname.split('.')[-1]
            bag_dfs = []
            for bag in range(bag_count):
                filename = '%s/predictions-%s-%02i.csv.gz' % (dirname, fold, bag)
                try:
                    df = read_csv(filename, skiprows=1, index_col=[0, 1], compression='gzip', engine='python')
                    df = df[['prediction']]
                    df.rename(columns={'prediction': '%s.%s' % (classifier, bag)}, inplace=True)
                    bag_dfs.append(df)
                except:
                    print('file not existed or crashed %s' % filename)
            dirname_dfs.append(concat(bag_dfs, axis=1))
        concat(dirname_dfs, axis=1).sort_index().to_csv('%s/predictions-%s.csv.gz' % (path, fold), compression='gzip')


data_folder = abspath(argv[1])
data_name = data_folder.split('/')[-1]
fns = listdir(data_folder)
fns = [fn for fn in fns if fn != 'analysis']
fns = [data_folder + '/' + fn for fn in fns]
feature_folders = [fn for fn in fns if isdir(fn)]

p = load_properties(data_folder)
# fold_count = int(p['foldCount'])
if 'foldAttribute' in p:
    input_fn = '%s/%s' % (feature_folders[0], 'data.arff')
    assert exists(input_fn)
    headers = load_arff_headers(input_fn)
    fold_values = headers[p['foldAttribute']]
else:
    fold_values = range(int(p['foldCount']))

nested_fold_count = int(p['nestedFoldCount'])
bag_count = max(1, int(p['bagCount']))
for path in feature_folders:
    combine_individual(path)
