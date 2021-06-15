from numpy import argmax, argmin, argsort, corrcoef, mean, nanmax, sqrt, triu_indices_from, where
from pandas import DataFrame, concat, read_csv
from scipy.io.arff import loadarff
import sklearn.metrics
import numpy as np
import os
from os.path import exists,abspath,isdir,dirname
from sys import argv
from os import listdir,environ
import pandas as pd
from common import load_arff_headers, load_properties, read_arff_to_pandas_df
from xgboost import XGBClassifier
from os import mkdir
import common

def read_arff_to_pandas_df(arff_path):
    # loadarff doesn't support string attribute
    # data = arff.loadarff(arff_path)
    df = pd.read_csv(arff_path, comment='@')
    num_col = df.shape[1]
    columns = []
    file1 = open(arff_path, 'r')
    Lines = file1.readlines()

    count = 0
    # Strips the newline character
    for line_idx, line in enumerate(Lines):
        # if line_idx > num_col
        if '@attribute' in line:
            columns.append(line.strip().split(' ')[1])

    df.columns = columns
    return df

def xgboost_predictions_result(outcome_path):


    data_source_dir = outcome_path.split('/')[-2]
    data_name = outcome_path.split('/')[-1]
    working_dir = dirname(abspath(argv[0]))

    ### get weka properties from weka.properties
    p = load_properties(outcome_path)
    df = read_arff_to_pandas_df(os.path.join(outcome_path,
                                             'concatenated/data.arff'))



    fold_values = list(df[p['foldAttribute']].unique())
    fold_col = p['foldAttribute']
    id_col = p['idAttribute']
    label_col = p['classAttribute']
    column_non_feature = [fold_col, label_col, id_col]
    feature_columns = df.columns.tolist()
    for nf in column_non_feature:
        feature_columns.remove(nf)
    print(feature_columns)

    df.replace(to_replace="pos", value="1", inplace=True)
    df.replace(to_replace="neg", value="0", inplace=True)
    df[label_col] = pd.to_numeric(df[label_col])

    # print(feature_columns)
    test_labels = []
    test_predictions = []
    test_dfs = []

    for outer_fold in fold_values:
        test_bool = df[fold_col] == outer_fold
        train_bool = df[fold_col] != outer_fold
        test_split_list = df.loc[test_bool]
        train_split_list = df.loc[test_bool]
        test_nf = test_split_list[column_non_feature]
        train_nf = train_split_list[column_non_feature]

        test_feat = test_split_list[feature_columns]
        train_feat = train_split_list[feature_columns]

        test_label = test_nf[label_col]
        # test_label.replace(to_replace="pos",value="1")
        # test_label.replace(to_replace="neg",value="1")
        # test_labels.append(test_label)

        train_label = train_nf[label_col]

        xgb_clf = XGBClassifier()
        xgb_clf.fit(train_feat, train_label)
        test_prediction = xgb_clf.predict_proba(test_feat)[:, 1]
        # test_predictions.append(test_prediction)
        test_df = pd.DataFrame(
            {'label': test_label, 'prediction': test_prediction})
        test_dfs.append(test_df)

    test_df_cat = pd.concat(test_dfs)
    print(test_df_cat)
    fmax = common.fmeasure_score(test_df_cat.label, test_df_cat.prediction, None)
    auc = sklearn.metrics.roc_auc_score(test_df_cat.label, test_df_cat.prediction)
    cols = ['data_name', 'fmax', 'method', 'auc']
    performance_df = pd.DataFrame(data=[['xgboost', fmax['F'], 'XGB_base', auc]], columns=cols, index=[0])
    analysis_folder = os.path.join(outcome_path, 'analysis')
    if not exists(analysis_folder):
        mkdir(analysis_folder)
    performance_df.to_csv(os.path.join(analysis_folder, "performance.csv"), index=False)

data_path = '/sc/arion/scratch/liy42/covid19_DECEASED_INDICATOR/xgboost'
for outcome in os.walk(data_path, topdown=True):
    root, dirs, files = outcome
    num_sep = data_path.count(os.path.sep)
    num_sep_this = root.count(os.path.sep)
    print(root)
    # print(cat_dir)
    if (root == data_path):
        # print(dirs)
        for dir in dirs:
            go_scratch_dir = os.path.join(root, dir)
            if go_scratch_dir != data_path:
                print(go_scratch_dir)
                xgboost_predictions_result(go_scratch_dir)
    else:
        break

