import pandas as pd
import datetime
import numpy as np
import os
from os.path import exists, abspath, isdir
from os import mkdir
from sys import argv
from glob import glob
from multiprocessing import Pool
from itertools import product
import arff
# from soft_impute import SoftImpute
from scipy.sparse import coo_matrix, csr_matrix, eye, load_npz, save_npz
# from rwr_from_jeff import *
from sklearn.model_selection import KFold, StratifiedKFold
# import networkx as nx

txt_dir = 'not_on_github/edge_list/'
edge_txt_format = 'human_string_{}_adjacency.txt'

csv_path = 'not_on_github/csv/'
features = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']

nz_rows_list = []
for f in features:
    df = pd.read_csv('{}{}.csv'.format(csv_path, f), index_col=0)
    print(f, 'original shape:', df.shape)

    np_a = df.values
    # print(np_a)
    # zero_rows = (np.all(np_a == 0, axis=0))
    zero_rows = (np.all(np_a == 0, axis=1))
    nz_rows_list.append(~zero_rows)
    # print(np.sum(zero_rows))
    # print(np.sum(zero_cols))
    # zero_rows_df = (df == 0).all(axis=0)
    # zero_cols_df = (df == 0).all(axis=1)
    # print(np.sum(zero_rows_df))
    # print(np.sum(zero_cols_df))
    # print(np_a == 0)
    # print('{} missing count: {}%'.format(f, sum(df.isnull().values)))

nz_all = np.ones_like(nz_rows_list[0])

for nz_rows in nz_rows_list:
    nz_all = np.logical_and(nz_all, nz_rows)

print(np.sum(nz_all))



