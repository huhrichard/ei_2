import common
import pandas as pd
import argparse
from time import time
from os import mkdir
from os.path import abspath, exists
from sys import argv
from numpy import array, column_stack, append
from numpy.random import choice, seed
from sklearn.cluster import MiniBatchKMeans
# from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.linear_model import SGDClassifier  # SGD
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.linear_model import LogisticRegression  # Logistic regression
from sklearn.ensemble import AdaBoostClassifier  # Adaboost
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.ensemble import GradientBoostingClassifier  # Logit Boost with parameter(loss='deviance')
from sklearn.neighbors import KNeighborsClassifier  # K nearest neighbors (IBk in weka)
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier

from xgboost import XGBClassifier
from sklearn.svm import SVC
# XGBoost?

import sklearn
import warnings
from common import load_arff_headers, load_properties
from os.path import abspath, isdir
from os import remove, system, listdir
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import product

# def reshape_base_pred_to_tensor(base_pred_df):
#     base_pred_cols = base_pred_df.columns
#     new_df = pd.DataFrame({'pred': 0.0,
#                            'base_data':'', 'base_cls':'', 'base_bag': '', 'idx': ''
#                            })
#     base_pred_df['idx'] = base_pred_cols.index
#     melt_base_pred_df = pd.melt(base_pred_df, id_vars=['idx'],
#                                 value_vars=base_pred_cols,
#                                 var_name='data_cls_bag')
#
#     melt_base_pred_df['base_data'] = ''
#     melt_base_pred_df['base_cls'] = ''
#     # melt_base_pred_df['base_bag'] = ''
#
#     melt_base_pred_df['base_data'] = melt_base_pred_df['data_cls_bag'].str.split('.')[0]
#     melt_base_pred_df['base_cls'] = melt_base_pred_df['data_cls_bag'].str.split('.')[1]
#     # melt_base_pred_df['base_bag'] = melt_base_pred_df['value'].str.split('.')[2]
#
#     # gpby_df = pd.group_by(['base_data', 'base_cls'])
#     pivoted_df = pd.pivot_table(melt_base_pred_df, values='value',
#                                  index=['idx'], columns=['base_data', 'base_cls'],
#                                 aggfunc=np.mean)
#
#     dim0 = len(pivoted_df.columns.get_level_values(0).unique())
#     dim1 = len(pivoted_df.columns.get_level_values(1).unique())
#     base_pred_tensor = pivoted_df.values.reshape((dim0, dim1, pivoted_df.shape[1]))
#
#     return base_pred_tensor

import tensor_cca
def tcca_projection(X, rDim=3):
    var_mats, cov_t = tensor_cca.var_cov_ten_calculation(X)
    H, Z = tensor_cca.tcca(X, var_mats=var_mats, cov_ten=cov_t, rDim=rDim)
    return H, Z

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', '-P', type=str, required=True, help='data path')
    parser.add_argument('--fold', '-F', type=int, default=5, help='cross-validation fold')
    parser.add_argument('--aggregate', '-A', type=int, default=1, help='if aggregate is needed, feed bagcount, else 1')
    parser.add_argument('--rdim', '-R', type=int, default=10, help='desired reduced dimension')
    args = parser.parse_args()
    data_path = abspath(args.path)

    fns = listdir(data_path)

    excluding_folder = ['analysis']
    fns = [fn for fn in fns if not fn in excluding_folder]
    fns = [fn for fn in fns if not 'tcca' in fn]

    fns = [data_path + '/' + fn for fn in fns]
    feature_folders = [fn for fn in fns if isdir(fn)]
    if len(feature_folders) == 0:
        feature_folders.append('./')
    assert len(feature_folders) > 0
    ### get weka properties from weka.properties
    p = load_properties(data_path)
    # fold_values = range(int(p['foldCount']))
    assert ('foldAttribute' in p) or ('foldCount' in p)
    if 'foldAttribute' in p:
        # input_fn = '%s/%s' % (feature_folders[0], 'data.arff')
        # assert exists(input_fn)
        # headers = load_arff_headers(input_fn)
        # fold_values = headers[p['foldAttribute']]
        fold_values = ['67890']
    else:
        fold_values = range(int(p['foldCount']))
    testing_bool = ('67890' in fold_values and 'foldAttribute' in p)
    list_of_rdim = np.array(range(args.rdim))+1
    for rdim in list_of_rdim:
        tcca_path = os.path.join(data_path, 'tcca{}/'.format(rdim))
        if not os.path.exists(tcca_path):
            os.mkdir(tcca_path)
        os.system('cp {} {}'.format(os.path.join(data_path, 'classifiers.txt'),tcca_path))
        os.system('cp {} {}'.format(os.path.join(data_path, 'weka.properties'),tcca_path))

        for fold in fold_values:
            train_base_preds = []
            test_base_preds = []
            train_labels = []
            test_labels = []
            train_id, test_id = None, None
            for view_path in feature_folders:
                train_df, train_labels, test_df, test_labels = common.read_fold(view_path, fold)
                train_df = common.unbag(train_df, args.aggregate)

                test_df = common.unbag(test_df, args.aggregate)
                train_base_preds.append(train_df.values)
                test_base_preds.append(test_df.values)
                train_id = train_df.index
                test_id = test_df.index

            H_train, Z_train = tcca_projection(train_base_preds, rDim=rdim)
            Z_test = []
            feat_col_name = []

            for view_path in feature_folders:
                for r in range(rdim):
                    feat_col_name.append('{}.tcca{}.0'.format(view_path.split('/')[-1], r))

            for v in range(len(H_train)):
                Z_test.append(np.matmul(test_base_preds[v], H_train[v]))

            project_train_array = np.hstack(Z_train)
            project_test_array = np.hstack(Z_test)
            Z_test = np.array(Z_test)
            print('rDim = {}, number of complex: {} out of {}'.format(rdim, np.sum(np.iscomplex(project_test_array)),
                                                                      project_test_array.size))
            train_fn = '%s/validation-%s.csv.gz' % (tcca_path, fold)
            test_fn = '%s/predictions-%s.csv.gz' % (tcca_path, fold)

            projected_train_df = pd.DataFrame(data=project_train_array,
                                              columns=feat_col_name,
                                              index=train_id)

            projected_test_df = pd.DataFrame(data=project_test_array,
                                              columns=feat_col_name,
                                              index=test_id)

            projected_train_df.to_csv(train_fn, compression='gzip')
            projected_test_df.to_csv(test_fn, compression='gzip')






    # main(args.path, args.fold, args.aggregate)