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


csv_path = 'not_on_github/csv/'
features = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']

for f in features:
    df = pd.read_csv('{}{}.csv'.format(csv_path, f))
    print('{} missing count: {}%'.format(f, sum(df.isnull().values)))

