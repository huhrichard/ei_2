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

def read_pca_arff(df_fn, v_path):
    pca_df = common.read_arff_to_pandas_df(df_fn)
    # print(pca_df)
    pca_df.rename(columns={p['idAttribute']: 'id', p['classAttribute']: 'label'}, inplace=True)
    pca_df.loc[:,'label'] = pca_df['label'].replace({'pos': '1', 'neg':
                                                     '0'}).astype(int)
    pca_df.drop(columns=[p['foldAttribute']], inplace=True)
    pca_df.set_index(['id', 'label'], inplace=True)
    # print(pca_df)
    v = v_path.split('/')[-1]
    pca_df = pca_df.add_prefix(v+'.')
    return pca_df

def mkdir_as_method(method_path):
    if not os.path.exists(method_path):
        os.mkdir(method_path)
    os.system('cp {} {}'.format(os.path.join(data_path, 'classifiers.txt'), method_path))
    os.system('cp {} {}'.format(os.path.join(data_path, 'weka.properties'), method_path))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def project(X, rDim=3):
    var_mats, cov_t = tensor_cca.var_cov_ten_calculation(X)
    H, Z = tensor_cca.tcca(X, var_mats=var_mats, cov_ten=cov_t, rDim=rDim)
    return H.real, Z.real

def EI_tcca_v0(dest_path, f_list, rdim=10):
    """
    Only base predicted scores put into TCCA
    :param dest_path:
    :param f_list:
    :param rdim:
    :return:
    """
    for fold in f_list:
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

        # if args.clf_as_view:
        #     train_base_preds = np.swapaxes(np.array(train_base_preds), 0, -1)
        #     test_base_preds = np.swapaxes(np.array(test_base_preds), 0, -1)
            #     H_train, Z_train = project(train_base_preds, rDim=rdim)
            # else:
        H_train, Z_train = project(train_base_preds, rDim=rdim)
        Z_test = []
        feat_col_name = []

        for view_path in feature_folders:
            for r in range(rdim):
                feat_col_name.append('{}.tcca{}.0'.format(view_path.split('/')[-1], r))

        for v in range(len(H_train)):
            Z_test.append(np.matmul(test_base_preds[v], H_train[v]))

        tcca_project_train_array = np.hstack(Z_train)
        tcca_project_test_array = np.hstack(Z_test)
        Z_test = np.array(Z_test)
        print('rDim = {}, number of complex: {} out of {}'.format(rdim, np.sum(np.iscomplex(tcca_project_train_array)),
                                                                  tcca_project_train_array.size))

        train_fn = '%s/validation-%s.csv.gz' % (dest_path, fold)
        test_fn = '%s/predictions-%s.csv.gz' % (dest_path, fold)

        projected_train_df = pd.DataFrame(data=tcca_project_train_array,
                                          columns=feat_col_name,
                                          index=train_id)

        projected_test_df = pd.DataFrame(data=tcca_project_test_array,
                                         columns=feat_col_name,
                                         index=test_id)

        projected_train_df.to_csv(train_fn, compression='gzip')
        projected_test_df.to_csv(test_fn, compression='gzip')

def EI_tcca_v1(dest_path, f_list, rdim=10):
    """
    Perform TCCA with data concatenated base predicted score and PCA
    :param dest_path:
    :param f_list:
    :param rdim:
    :return:
    """
    for fold in f_list:
        train_base_preds = []
        test_base_preds = []
        train_labels = []
        test_labels = []
        train_id, test_id = None, None
        for view_path in feature_folders:
            pca_df_name = os.path.join(view_path, 'data_pca_{}.arff'.format(fold))
            pca_df = read_pca_arff(pca_df_name, view_path)
            # print(pca_df)

            train_df, train_labels, test_df, test_labels = common.read_fold(view_path, fold)
            train_df = common.unbag(train_df, args.aggregate)
            v = view_path.split('/')[-1]
            train_df = train_df.add_prefix(v + '.')
            train_with_pca_df = pd.concat([train_df, pca_df], axis=1, join='inner')
            test_df = common.unbag(test_df, args.aggregate)
            test_df = test_df.add_prefix(v + '.')
            test_with_pca_df = pd.concat([test_df, pca_df], axis=1, join='inner')
            # print(test_df)

            train_base_preds.append(train_with_pca_df.values)
            test_base_preds.append(test_with_pca_df.values)
            train_id = train_with_pca_df.index
            test_id = test_with_pca_df.index

            feat_col_name = feat_col_name + train_with_pca_df.columns.tolist()

        H_train, Z_train = project(train_base_preds, rDim=rdim)
        Z_test = []
        feat_col_name = []



        for view_path in feature_folders:
            for r in range(rdim):
                # if args.clf_as_view:
                feat_col_name.append('{}.tcca{}.0'.format(view_path.split('/')[-1], r))
        #
        for v in range(len(H_train)):
            Z_test.append(np.matmul(test_base_preds[v], H_train[v]))

        tcca_project_train_array = np.hstack(Z_train)
        tcca_project_test_array = np.hstack(Z_test)
        print('rDim = {}, number of complex: {} out of {}'.format(rdim, np.sum(np.iscomplex(tcca_project_train_array)),
                                                                  tcca_project_train_array.size))

        train_fn = '%s/validation-%s.csv.gz' % (dest_path, fold)
        test_fn = '%s/predictions-%s.csv.gz' % (dest_path, fold)

        projected_train_df = pd.DataFrame(data=tcca_project_train_array,
                                          columns=feat_col_name,
                                          index=train_id)

        projected_test_df = pd.DataFrame(data=tcca_project_test_array,
                                         columns=feat_col_name,
                                         index=test_id)

        projected_train_df.to_csv(train_fn, compression='gzip')
        projected_test_df.to_csv(test_fn, compression='gzip')

def EI_pca_only(dest_path, f_list):
    """
    Move base score from pca to 'dest_path'
    :param dest_path:
    :param f_list:
    :param rdim:
    :return:
    """
    for fold in f_list:
        os.system('cp {} {}'.format(os.path.join(data_path, 'predictions-pca_{}.csv.gz'.format(fold)),
                                    os.path.join(dest_path, 'predictions-{}.csv.gz'.format(fold))))
        os.system('cp {} {}'.format(os.path.join(data_path, 'validation-pca_{}.csv.gz'.format(fold)),
                                    os.path.join(dest_path, 'validation-{}.csv.gz'.format(fold))))

def EI_base_cat_pca(dest_path, f_list):
    """
    Perform TCCA with data concatenated base predicted score and PCA
    :param dest_path:
    :param f_list:
    :param rdim:
    :return:
    """
    for fold in f_list:
        train_base_preds = []
        test_base_preds = []
        train_labels = []
        test_labels = []
        train_id, test_id = None, None
        feat_col_name = []
        for view_path in feature_folders:
            pca_df_name = os.path.join(view_path, 'data_pca_{}.arff'.format(fold))
            pca_df = read_pca_arff(pca_df_name, view_path)

            train_df, train_labels, test_df, test_labels = common.read_fold(view_path, fold)
            train_df = common.unbag(train_df, args.aggregate)
            v = view_path.split('/')[-1]
            train_df = train_df.add_prefix(v+'.')
            # feat_col_name = feat_col_name + train_df.columns
            # feat_col_name = feat_col_name + ['{}.pca_projected_feat.{}'.format(view_path.split('/')[-1], i) for i in range(pca_df.shape[1])]
            train_with_pca_df = pd.concat([train_df, pca_df], axis=1, join='inner')

            print(train_with_pca_df)
            test_df = common.unbag(test_df, args.aggregate)
            test_df = test_df.add_prefix(v + '.')
            test_with_pca_df = pd.concat([test_df, pca_df], axis=1, join='inner')

            train_base_preds.append(train_with_pca_df)
            test_base_preds.append(test_with_pca_df)
            train_id = train_with_pca_df.index
            test_id = test_with_pca_df.index
            feat_col_name = feat_col_name + train_with_pca_df.columns.tolist()




        tcca_project_train_array = np.hstack(train_base_preds)
        tcca_project_test_array = np.hstack(test_base_preds)

        train_fn = '%s/validation-%s.csv.gz' % (dest_path, fold)
        test_fn = '%s/predictions-%s.csv.gz' % (dest_path, fold)

        projected_train_df = pd.DataFrame(data=tcca_project_train_array,
                                          columns=feat_col_name,
                                          index=train_id)

        projected_test_df = pd.DataFrame(data=tcca_project_test_array,
                                         columns=feat_col_name,
                                         index=test_id)

        projected_train_df.to_csv(train_fn, compression='gzip')
        projected_test_df.to_csv(test_fn, compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', '-P', type=str, required=True, help='data path')
    parser.add_argument('--fold', '-F', type=int, default=5, help='cross-validation fold')
    parser.add_argument('--aggregate', '-A', type=int, default=1, help='if aggregate is needed, feed bagcount, else 1')
    parser.add_argument('--rdim', '-R', type=int, default=10, help='desired reduced dimension')
    # parser.add_argument('--clf_as_view', '-cav', type=str2bool, default='false', help='desired reduced dimension')
    args = parser.parse_args()
    data_path = abspath(args.path)

    fns = listdir(data_path)

    excluding_folder = ['analysis']
    feature_folders = common.data_dir_list(data_path)
    if len(feature_folders) == 0:
        feature_folders.append('./')
    assert len(feature_folders) > 0
    ### get weka properties from weka.properties
    p = load_properties(data_path)
    # fold_values = range(int(p['foldCount']))
    assert ('foldAttribute' in p) or ('foldCount' in p)
    if 'foldAttribute' in p:
        df = common.read_arff_to_pandas_df(feature_folders[0] + '/data.arff')
        fold_values = df[p['foldAttribute']].unique()
    else:
        fold_values = range(int(p['foldCount']))
    pca_fold_values = ['pca_{}'.format(fv) for fv in fold_values]
    testing_bool = ('67890' in fold_values and 'foldAttribute' in p)
    # list_of_rdim = np.array(range(args.rdim))+1
    # list_of_rdim = [10]
    # list_of_rdim = np.array(range(args.rdim))+1
    # list_of_rdim = np.array(range(args.rdim))+1
    # for rdim in list_of_rdim:

    tcca_path = os.path.join(data_path, 'tcca_{}/'.format(10))
    # tcca_pca_path = os.path.join(data_path, 'tcca_{}_with_pca_feat/'.format(10))
    pca_EI_path = os.path.join(data_path, 'pca_only_EI/')
    base_pca_EI_path = os.path.join(data_path, 'base_cat_pca_EI/')
    mkdir_as_method(tcca_path)
    # mkdir_as_method(tcca_pca_path)
    mkdir_as_method(pca_EI_path)
    mkdir_as_method(base_pca_EI_path)

    EI_tcca_v0(tcca_path, fold_values)
    # EI_tcca_v1(tcca_pca_path, fold_values)
    EI_pca_only(pca_EI_path, fold_values)
    EI_base_cat_pca(base_pca_EI_path, fold_values)










    # main(args.path, args.fold, args.aggregate)