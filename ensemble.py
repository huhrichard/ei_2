import common
import pandas as pd
import argparse
from time import time
from os import mkdir
import os
from os.path import abspath, exists
from sys import argv
from numpy import array, column_stack, append
from numpy.random import choice, seed
from sklearn.cluster import MiniBatchKMeans
# from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Random Forest
from sklearn.linear_model import SGDClassifier  # SGD
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.linear_model import LogisticRegression, LinearRegression  # Logistic regression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor  # Adaboost
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # Decision Tree
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # Logit Boost with parameter(loss='deviance')
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # K nearest neighbors (IBk in weka)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import (
                            LinearRegression,
                            TheilSenRegressor,
                            RANSACRegressor,
                            HuberRegressor,
                            ElasticNetCV, ElasticNet,
                            LassoCV, Lasso,
                            BayesianRidge,
                            LarsCV, Lars,
                            LassoLarsCV,
                            ARDRegression,
                            RidgeCV, Ridge
                            )

from sklearn.metrics import fbeta_score, make_scorer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVR
from beat_pd_scoring_function import weighted_mse, \
    measurement_subject_df, measurement_id_col, subject_id_col
from sklearn.preprocessing import StandardScaler

# XGBoost?

import sklearn
import warnings
from common import load_arff_headers, load_properties
from os.path import abspath, isdir
from os import remove, system, listdir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
from itertools import product
from joblib import Parallel

from sklearn.metrics import check_scoring
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
# from sklearn.utils import

warnings.filterwarnings("ignore")

wmse_sklearn = make_scorer(weighted_mse, greater_is_better=False)
fmax_sklearn = make_scorer(common.f_max, greater_is_better=True, needs_proba=True)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def checkFolder(path, fold_count=5):
    for fold in range(fold_count):
        if not exists('%s/predictions-%d.csv.gz' % (path, fold)):
            return False
        if not exists('%s/validation-%d.csv.gz' % (path, fold)):
            return False
    return True


def get_performance(df, ensemble, fold, seedval, regression=False):
    labels = df.index.get_level_values('label').values
    predictions = df[ensemble].mean(axis=1)
    if regression:
        score = weighted_mse(labels, predictions)
    else:
        score = common.fmeasure_score(labels, predictions)['F']

    return {'fold': fold, 'seed': seedval, 'score': score, 'ensemble': ensemble[-1],
            'ensemble_size': len(ensemble)}


def get_predictions(df, ensemble, fold, seedval):
    ids = df.index.get_level_values('id')
    labels = df.index.get_level_values('label')
    predictions = df[ensemble].mean(axis=1)
    diversity = common.diversity_score(df[ensemble].values)
    return pd.DataFrame(
        {'fold': fold, 'seed': seedval, 'id': ids, 'label': labels, 'prediction': predictions, 'diversity': diversity,
         'ensemble_size': len(ensemble)})




def select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i, scoring_func):
    initial_ensemble_size = 2
    max_candidates = 50
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace=False)
        # candidate_scores = [common.score(train_labels, train_df[ensemble + [candidate]].mean(axis=1)) for candidate in
        #                     candidates]
        candidate_scores = [scoring_func(train_labels, train_df[ensemble + [candidate]].mean(axis=1)) for candidate in
                            candidates]
        best_candidate = candidates[common.argbest(candidate_scores)]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


# def selection(fold, seedval, path, agg):
#     seed(seedval)
#     initial_ensemble_size = 2
#     max_ensemble_size = 50
#     max_candidates = 50
#     max_diversity_candidates = 5
#     accuracy_weight = 0.5
#     max_clusters = 20
#     train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
#     # print(train_df)
#     train_df = common.unbag(train_df, agg)
#     test_df = common.unbag(test_df, agg)
#     best_classifiers = train_df.apply(lambda x: common.fmeasure_score(train_labels, x)['F']).sort_values(
#         ascending=not common.greater_is_better)
#     # best_classifiers = train_df.apply(lambda x: weighted_mse(train_labels, x)).sort_values(
#     #
#     train_performance = []
#     test_performance = []
#     ensemble = []
#     for i in range(min(max_ensemble_size, len(best_classifiers))):
#         best_candidate = select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i, scoring_func=common.f_max)
#         ensemble.append(best_candidate)
#         train_performance.append(get_performance(train_df, ensemble, fold, seedval))
#         test_performance.append(get_performance(test_df, ensemble, fold, seedval))
#     train_performance_df = pd.DataFrame.from_records(train_performance)
#     best_ensemble_size = common.get_best_performer(train_performance_df).ensemble_size.values
#     best_ensemble = train_performance_df.ensemble[:best_ensemble_size.item(0) + 1]
#     return get_predictions(test_df, best_ensemble, fold, seedval), \
#            pd.DataFrame.from_records(test_performance), \
#            get_predictions(train_df, best_ensemble, fold, seedval)

def selection(fold, seedval, path, agg, subject_model=False,
                           regression=False, greater_is_better=False,
                           scoring_func=common.f_max):
    seed(seedval)
    initial_ensemble_size = 2
    max_ensemble_size = 50
    max_candidates = 50
    max_diversity_candidates = 5
    accuracy_weight = 0.5
    max_clusters = 20
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    # print(train_df)
    train_df = common.unbag(train_df, agg)

    test_df = common.unbag(test_df, agg)
    if regression:
        scoring_func = weighted_mse
    if subject_model:
        measurement_id_train = train_df.index.get_level_values('id')
        measurement_id_test = test_df.index.get_level_values('id')
        # print(measurement_id_test)
        subset_ms_df = measurement_subject_df.loc[measurement_subject_df[measurement_id_col].isin(measurement_id_train)]
        subject_id_list = subset_ms_df[subject_id_col].unique()
        # print(subject_id_list)
        # m_count_per_subject = subset_ms_df_train[subject_id_col].value_counts()
        # print(measurement_id_train)

        test_pred_df_list = []
        test_record_list = []
        train_pred_df_list = []

        for subject_id in subject_id_list:
            # print(subject_id)

            measurements_of_subject = measurement_subject_df.loc[(measurement_subject_df[subject_id_col] == subject_id), measurement_id_col]

            measurements_train_bool_of_subject = measurement_id_train.isin(measurements_of_subject)
            measurements_test_bool_of_subject = measurement_id_test.isin(measurements_of_subject)

            train_df_per_subject = train_df.loc[measurements_train_bool_of_subject]
            # print(train_df_per_subject)
            train_label_per_subject = train_labels[measurements_train_bool_of_subject]
            test_df_per_subject = test_df.loc[measurements_test_bool_of_subject]
            # print(test_df_per_subject)
            test_label_per_subject = test_labels[measurements_test_bool_of_subject]

            best_classifiers_per_subject = train_df_per_subject.apply(lambda x: weighted_mse(train_label_per_subject, x)).sort_values(
                ascending=greater_is_better)
            # print(best_classifiers)
            train_performance = []
            test_performance = []
            ensemble = []
            for i in range(min(max_ensemble_size, len(best_classifiers_per_subject))):
                best_candidate = select_candidate_enhanced(train_df_per_subject, train_label_per_subject,
                                                           best_classifiers_per_subject, ensemble, i,
                                                           scoring_func=scoring_func)
                ensemble.append(best_candidate)
                train_performance.append(get_performance(train_df_per_subject, ensemble, fold, seedval, regression=regression))
                test_performance.append(get_performance(test_df_per_subject, ensemble, fold, seedval, regression=regression))
            train_performance_df = pd.DataFrame.from_records(train_performance)
            best_ensemble_size = common.get_best_performer(train_performance_df, _greater_is_better=greater_is_better).ensemble_size.values
            # print(best_ensemble_size)
            best_ensemble = train_performance_df.ensemble[:best_ensemble_size.item(0) + 1]

            test_pred_df = get_predictions(test_df_per_subject, best_ensemble, fold, seedval)
            test_record = pd.DataFrame.from_records(test_performance)
            train_pred_df = get_predictions(train_df_per_subject, best_ensemble, fold, seedval)
            # print(test_pred_df)

            test_pred_df_list.append(test_pred_df)
            test_record_list.append(test_record)
            train_pred_df_list.append(train_pred_df)

        return pd.concat(test_pred_df_list), pd.concat(test_record_list), pd.concat(train_pred_df_list)

        # return get_predictions(test_df, best_ensemble, fold, seedval), \
        #        pd.DataFrame.from_records(test_performance), \
        #        get_predictions(train_df, best_ensemble, fold, seedval)

    else:
        # best_classifiers = train_df.apply(lambda x: common.fmeasure_score(train_labels, x)['F']).sort_values(
        best_classifiers = train_df.apply(lambda x: scoring_func(train_labels, x)).sort_values(
            ascending=greater_is_better)
        # print(best_classifiers)

        train_performance = []
        test_performance = []
        ensemble = []
        for i in range(min(max_ensemble_size, len(best_classifiers))):
            best_candidate = select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i,
                                                       scoring_func=scoring_func)
            ensemble.append(best_candidate)
            train_performance.append(get_performance(train_df, ensemble, fold, seedval,
                                                     regression=regression))
            test_performance.append(get_performance(test_df, ensemble, fold, seedval,
                                                    regression=regression))
        train_performance_df = pd.DataFrame.from_records(train_performance)
        best_ensemble_size = common.get_best_performer(train_performance_df,
                                                       _greater_is_better=greater_is_better).ensemble_size.values
        # print(best_ensemble_size)
        best_ensemble = train_performance_df.ensemble[:best_ensemble_size.item(0) + 1]
        return get_predictions(test_df, best_ensemble, fold, seedval), \
               pd.DataFrame.from_records(test_performance), \
               get_predictions(train_df, best_ensemble, fold, seedval)

def thres_fmax(train_label_df, train_pred_df):
    if testing_bool:
        fmax_training = common.fmeasure_score(train_label_df, train_pred_df)
        thres = fmax_training['thres']
    else:
        thres = None

    return thres

def CES_fmax(path, fold_count=range(5), agg=1):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    method = 'enhanced'
    select_candidate = eval('select_candidate_' + method)
    method_function = selection
    initial_ensemble_size = 2
    max_ensemble_size = 50
    max_candidates = 50
    max_diversity_candidates = 5
    accuracy_weight = 0.5
    max_clusters = 20
    predictions_dfs = []
    train_predictions_dfs = []
    performance_dfs = []
    seeds = range(agg)

    for seedval in seeds:
        # for fold in range(fold_count):
        for fold in fold_count:
            # if '67890' in fold:
            # if testing_bool or (not 'foldAttribute' in p):
            pred_df, perf_df, train_pred_df = method_function(fold, seedval, path, agg)
            predictions_dfs.append(pred_df)
            train_predictions_dfs.append(train_pred_df)
            performance_dfs.append(perf_df)
            thres = thres_fmax(train_pred_df.label, train_pred_df.prediction)
    performance_df = pd.concat(performance_dfs)
    performance_df.to_csv('%s/analysis/selection-%s-%s-iterations.csv' % (path, method, 'fmax'), index=False)
    predictions_df = pd.concat(predictions_dfs)
    predictions_df['method'] = method
    predictions_df['metric'] = 'fmax'
    predictions_df.to_csv('%s/analysis/selection-%s-%s.csv' % (path, method, 'fmax'), index=False)
    auc = sklearn.metrics.roc_auc_score(predictions_df.label, predictions_df.prediction)
    auprc = common.auprc(predictions_df.label, predictions_df.prediction)
    # auprc = sklearn.metrics.pre(predictions_df.label, predictions_df.prediction)

    # if ('67890' in fold_count and 'foldAttribute' in p):
    #     train_predictions_df = pd.concat(train_predictions_dfs)
    #
    #
    # else:
    print(thres)
    fmax = (common.fmeasure_score(predictions_df.label, predictions_df.prediction, thres=thres))
    return {'f-measure':fmax, 'auc':float(auc), 'auprc':auprc}

def CES(path, fold_count=range(5), agg=1,
                     subject_model=False, inference_only=False,
                     regression=False):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    method = 'enhanced'
    select_candidate = eval('select_candidate_' + method)
    method_function = selection
    initial_ensemble_size = 2
    max_ensemble_size = 50
    max_candidates = 50
    max_diversity_candidates = 5
    accuracy_weight = 0.5
    max_clusters = 20
    predictions_dfs = []
    train_predictions_dfs = []
    performance_dfs = []
    seeds = range(agg)

    for seedval in seeds:
        # for fold in range(fold_count):
        for fold in fold_count:
            # if '67890' in fold:
            # if testing_bool or (not 'foldAttribute' in p):
            if regression:
                greater_better = False
            else:
                greater_better = True
            pred_df, perf_df, train_pred_df = method_function(fold, seedval, path, agg,
                                                              subject_model=subject_model,
                                                              greater_is_better=greater_better,
                                                              regression=regression)
            predictions_dfs.append(pred_df)
            train_predictions_dfs.append(train_pred_df)
            performance_dfs.append(perf_df)
            if not regression:
                thres = thres_fmax(train_pred_df.label, train_pred_df.prediction)
    performance_df = pd.concat(performance_dfs)
    performance_df.to_csv('%s/analysis/selection-%s-%s-iterations.csv' % (path, method, 'weighted_mse'), index=False)

    predictions_df = pd.concat(predictions_dfs)
    predictions_df['method'] = method
    if regression:
        predictions_df['metric'] = 'weighted_mse'
    else:
        predictions_df['metric'] = 'fmax'
    predictions_df.to_csv('%s/analysis/selection-%s-%s.csv' % (path, method, 'weighted_mse'), index=False)
    # auc = '%.3f' % (sklearn.metrics.roc_auc_score(predictions_df.label, predictions_df.prediction))

    # if ('67890' in fold_count and 'foldAttribute' in p):
    #     train_predictions_df = pd.concat(train_predictions_dfs)
    #
    #
    # else:
    # print(thres)
    # print(predictions_df.index.loc[:, 'label'])
    # print(predictions_df['label'].values)
    # print(predictions_df)
    predictions_df_wo_label = predictions_df.drop(columns=['label', 'id'])
    # print(predictions_df_wo_label)
    predictions_df_wo_label = predictions_df_wo_label.reset_index()
    predictions_df_wo_label.drop(columns=['label'], inplace=True)
    predictions_df_wo_label.set_index('id', inplace=True)
    predictions_scores = predictions_df_wo_label.loc[:,'prediction']
    # print(predictions_scores)
    if regression:
        if inference_only is True:
            predictions_scores.to_csv("{}/{}/{}".format(path, "analysis",'CES_test_predictions'))
            return {'weighted_mse': np.nan}
        else:
            weighted_mse_score = weighted_mse(predictions_df['label'].values, predictions_scores)
            return {'weighted_mse': weighted_mse_score}
    else:
        auc = '%.3f' % (sklearn.metrics.roc_auc_score(predictions_df.label, predictions_df.prediction))
        fmax = (common.fmeasure_score(predictions_df.label, predictions_df.prediction, thres=thres))
        return {'f-measure': fmax, 'auc': float(auc)}


# m

def aggregating_ensemble(path, fold_count=range(5), agg=1, median=False,
                         subject_model=False, inference_only=False,
                         z_scoring=False, regression=False, test_set=False):
    def _unbag_mean(df, agg=agg):
        df = common.unbag(df, agg)
        return df.mean(axis=1)

    def _unbag_median(test_df_temp, agg=agg, z_scoring=False, train_df_temp=None):
        test_df_temp = common.unbag(test_df_temp, agg)
        # print(test_df_temp.values.shape)
        train_df_temp = common.unbag(train_df_temp, agg)
        # print(train_df_temp.values.shape)
        if z_scoring:
            z_scaler = StandardScaler()
            train_df_temp[:] = z_scaler.fit_transform(train_df_temp.values)
            test_df_z = test_df_temp.copy()
            test_df_z[:] = z_scaler.transform(test_df_temp.values)
            test_df_median = test_df_z.median(axis=1)
            for idx, (id, tz) in enumerate(test_df_z.iterrows()):
                tz_median = test_df_median[idx]
                median_bool = (tz == tz_median)
                # print(median_bool)
                # print(test_df_temp.iloc[idx][median_bool])
                # print(test_df_temp.iloc[idx][median_bool])
                test_df_median[idx] = test_df_temp.iloc[idx][median_bool]
            # print(test_df_median)
            return test_df_median
        else:
            return test_df_temp.median(axis=1)

    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    predictions = []
    labels = []
    train_dfs = []
    train_labels = []
    # for fold in range(fold_count):
    for fold in fold_count:

        # if testing_bool or (not 'foldAttribute' in p):

        train_df, train_label, test_df, test_label = common.read_fold(path, fold)
        # if subject_model:
        #     measurement_id_train = train_df.index.get_level_values('id')
        #     measurement_id_test = test_df.index.get_level_values('id')
        #     # print(measurement_id_test)
        #     subset_ms_df = measurement_subject_df.loc[
        #         measurement_subject_df[measurement_id_col].isin(measurement_id_train)]
        #     subject_id_list = subset_ms_df[subject_id_col].unique()
        #
        #     for subject_id in subject_id_list:
        #         # print(subject_id)
        #
        #         measurements_of_subject = measurement_subject_df.loc[
        #             (measurement_subject_df[subject_id_col] == subject_id), measurement_id_col]
        #
        #         measurements_train_bool_of_subject = measurement_id_train.isin(measurements_of_subject)
        #         measurements_test_bool_of_subject = measurement_id_test.isin(measurements_of_subject)
        #
        #         train_df_per_subject = train_df.loc[measurements_train_bool_of_subject]
        #         # print(train_df_per_subject)
        #         train_label_per_subject = train_label[measurements_train_bool_of_subject]
        #         test_df_per_subject = test_df.loc[measurements_test_bool_of_subject]
        #         # print(test_df_per_subject)
        #         test_label_per_subject = test_label[measurements_test_bool_of_subject]
        #         if median:
        #             predict = _unbag_median(test_df_per_subject, agg)
        #         else:
        #             predict = _unbag_mean(test_df_per_subject, agg)
        #         predictions.append(predict)
        #         labels.append(test_label_per_subject)
        # else:

        # thres = thres_fmax(train_label, _unbag_mean(train_df))

        if median:
            train_agg = _unbag_median(train_df, agg, z_scoring=z_scoring, train_df_temp=train_df)
            predict = _unbag_median(test_df, agg, z_scoring=z_scoring, train_df_temp=train_df)
        else:
            train_agg = _unbag_mean(train_df, agg)
            predict = _unbag_mean(test_df, agg)
        predictions.append(predict)
        labels.append(test_label)
        train_dfs.append(train_agg)
        train_labels.append(train_label)

    predictions = pd.concat(predictions)
    labels = np.concatenate(labels, axis=None)
    train_dfs = pd.concat(train_dfs)
    train_labels = np.concatenate(train_labels, axis=None)

    if inference_only is True:
        if median:
            output_name = "median"
        else:
            output_name = "mean"
        if z_scoring:
            z_str = '_Z_scoring'
        else:
            z_str = ''
        output_path = "{}/{}/{}{}{}".format(path, "analysis",
                                            output_name, z_str, '_test_predictions.csv')
        print(output_path)
        # print
        output_df = predictions.to_frame()
        output_df.columns = ['prediction']
        output_df.reset_index(inplace=True)
        print(output_df.columns)
        # output_df.reset_index(inplace=True)
        output_df.rename(columns={'id': 'measurement_id'}, inplace=True)
        output_df = output_df[['measurement_id', 'prediction']]
        output_df.to_csv(output_path, index=False)
        return {'weighted_mse': np.nan}
    else:
        if regression:
            weighted_mse_score = weighted_mse(labels, predictions)
            return {'weighted_mse': weighted_mse_score}
        else:
            # thres = thres_fmax(train_label, _unbag_mean(train_df))
            thres = thres_fmax(train_labels, train_dfs)

            fmax = common.fmeasure_score(labels, predictions, thres)
            auc = sklearn.metrics.roc_auc_score(labels, predictions)
            auprc = common.auprc(labels, predictions)

            return {'f-measure': fmax, 'auc': auc, 'auprc': auprc}



def bestbase_fmax(path, fold_count=range(5), agg=1):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    predictions = []
    labels = []

    # for fold in range(fold_count):
    for fold in fold_count:
        # if '67890' in fold:
        # if testing_bool or (not 'foldAttribute' in p):
        train_df, train_label, test_df, label = common.read_fold(path, fold)
        test_df = common.unbag(test_df, agg)
        predictions.append(test_df)
        labels = append(labels, label)
            # thres = thres_fmax(train_label, common.unbag(train_df, agg))
    predictions = pd.concat(predictions)

    # need to be changed
    fmax_list = [common.fmeasure_score(labels, predictions.iloc[:, i])['F'] for i in range(len(predictions.columns))]
    auc_list = [sklearn.metrics.roc_auc_score(labels, predictions.iloc[:, i]) for i in range(len(predictions.columns))]
    auprc_list = [common.auprc(labels, predictions.iloc[:, i]) for i in range(len(predictions.columns))]

    return {'f-measure':max(fmax_list), 'auc':max(auc_list), 'auprc':max(auprc_list)}

def best_base_predictors(path, fold_count=range(5), agg=1, regression=False):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    predictions = []
    labels = []

    # for fold in range(fold_count):
    for fold in fold_count:
        # if '67890' in fold:
        # if testing_bool or (not 'foldAttribute' in p):
        train_df, train_label, test_df, label = common.read_fold(path, fold)
        test_df = common.unbag(test_df, agg)
        predictions.append(test_df)
        labels = append(labels, label)
            # thres = thres_fmax(train_label, common.unbag(train_df, agg))
    predictions = pd.concat(predictions)

    # need to be changed
    if regression:
        weighted_mse_score_list = [weighted_mse(labels, predictions.iloc[:, i]) for i in
                                   range(len(predictions.columns))]
        min_score = min(weighted_mse_score_list)
        name_cls = predictions.columns.tolist()[np.argmin(weighted_mse_score_list)]
        return {'weighted_mse': min_score, 'name': name_cls}
    else:
        thres = thres_fmax(train_label, common.unbag(train_df, agg))
        fmax_list = [common.fmeasure_score(labels, predictions.iloc[:, i], thres=thres)['F'] for i in
                     range(len(predictions.columns))]
        auc_list = [sklearn.metrics.roc_auc_score(labels, predictions.iloc[:, i]) for i in
                    range(len(predictions.columns))]

        return {'f-measure': max(fmax_list), 'auc': max(auc_list)}

def reshape_base_pred_to_tensor(base_pred_df):
    base_pred_cols = base_pred_df.columns
    new_df = pd.DataFrame({'pred': 0.0,
                           'base_data':'', 'base_cls':'', 'base_bag': '', 'idx': ''
                           })
    base_pred_df['idx'] = base_pred_cols.index
    melt_base_pred_df = pd.melt(base_pred_df, id_vars=['idx'],
                                value_vars=base_pred_cols,
                                var_name='data_cls_bag')

    melt_base_pred_df['base_data'] = ''
    melt_base_pred_df['base_cls'] = ''
    # melt_base_pred_df['base_bag'] = ''

    melt_base_pred_df['base_data'] = melt_base_pred_df['data_cls_bag'].str.split('.')[0]
    melt_base_pred_df['base_cls'] = melt_base_pred_df['data_cls_bag'].str.split('.')[1]
    # melt_base_pred_df['base_bag'] = melt_base_pred_df['value'].str.split('.')[2]

    # gpby_df = pd.group_by(['base_data', 'base_cls'])
    pivoted_df = pd.pivot_table(melt_base_pred_df, values='value',
                                 index=['idx'], columns=['base_data', 'base_cls'],
                                aggfunc=np.mean)

    dim0 = len(pivoted_df.columns.get_level_values(0).unique())
    dim1 = len(pivoted_df.columns.get_level_values(1).unique())
    base_pred_tensor = pivoted_df.values.reshape((dim0, dim1, pivoted_df.shape[1]))

    return base_pred_tensor


def stacked_generalization(path, stacker_name, stacker, fold, agg, stacked_df,
                           z_scoring=False, subject_model=False,
                           regression=False):
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    # print('number of complex: {} out of {}'.format(np.sum(np.iscomplex(train_df.values)),
    #                                                           train_df.values.size))
    # train_df_cols = train_df.columns
    # f_train_base = [common.fmeasure_score(train_labels, train_df[c].values) for c in train_df_cols]
    # thres_train_base = [f['thres'] for f in f_train_base]
    # fscore_train_base = [f['F'] for f in f_train_base]
    # f_test_base = [common.fmeasure_score(test_labels, test_df[c].values, thres_train_base[idx]) for idx, c in enumerate(train_df_cols)]
    # fscore_test_base = [f['F'] for f in f_test_base]
    if z_scoring:
        z_scaler = StandardScaler()
        # print(test_df)
        train_df[:] = z_scaler.fit_transform(train_df.values)
        test_df[:] = z_scaler.transform(test_df.values)
        # print(test_df)

    if subject_model:
        measurement_id_train = train_df.index.get_level_values('id')
        measurement_id_test = test_df.index.get_level_values('id')
        # print(measurement_id_test)
        subset_ms_df = measurement_subject_df.loc[
            measurement_subject_df[measurement_id_col].isin(measurement_id_train)]
        subject_id_list = subset_ms_df[subject_id_col].unique()
        test_predictions = []
        train_predictions = []
        test_labels_dummy = []
        train_labels_dummy = []
        for subject_id in subject_id_list:
            # print(subject_id)

            measurements_of_subject = measurement_subject_df.loc[
                (measurement_subject_df[subject_id_col] == subject_id), measurement_id_col]

            measurements_train_bool_of_subject = measurement_id_train.isin(measurements_of_subject)
            measurements_test_bool_of_subject = measurement_id_test.isin(measurements_of_subject)

            train_df_per_subject = train_df.loc[measurements_train_bool_of_subject]
            # print(train_df_per_subject)
            train_label_per_subject = train_labels[measurements_train_bool_of_subject]
            test_df_per_subject = test_df.loc[measurements_test_bool_of_subject]
            # print(test_df_per_subject)
            test_label_per_subject = test_labels[measurements_test_bool_of_subject]
            stacker = stacker.fit(train_df_per_subject, train_label_per_subject)

            test_predictions.append(stacker.predict(test_df_per_subject))
            train_predictions.append(stacker.predict(train_df_per_subject))
            test_labels_dummy.append(test_label_per_subject)
            train_labels_dummy.append(train_label_per_subject)

        test_predictions = np.concatenate(test_predictions, axis=None)
        train_predictions = np.concatenate(train_predictions, axis=None)
        test_labels = np.concatenate(test_labels_dummy, axis=None)
        train_labels = np.concatenate(train_labels_dummy, axis=None)

    else:
        stacker = stacker.fit(train_df, train_labels)
        # feat_imp = []
        # if hasattr(stacker, 'feature_importances_'):
        #     feat_imp = stacker.feature_importances_
        # elif hasattr(stacker, 'coef_'):
        #     feat_imp = stacker.coef_
        # elif hasattr(stacker, 'theta_'):
        #     feat_imp = stacker.theta_
        # if len(feat_imp)>0:
        #     # feat_imp = np.squeeze(feat_imp)
        #     if len(feat_imp.shape) > 1:
        #         feat_imp = feat_imp[-1,:]
        #     # if not fold in stacked_df['fold']:
        #     new_df = pd.DataFrame({'f_train_base':fscore_train_base,
        #                            'f_test_base': fscore_test_base,
        #                            # 'base': train_df_cols,
        #                            'feat_imp': feat_imp,
        #                            'base_data':'', 'base_cls':'', 'base_bag': ''
        #                            })
        #
        #     split_str = pd.Series(train_df_cols).str.split('.',expand=True)
        #     # print(split_str[0])
        #     # new_df.loc[:,['base_data', 'base_cls', 'base_bag']] = ''
        #     # new_df.loc[:,['base_data', 'base_cls', 'base_bag']] =
        #     new_df.loc[:,'base_data'] = split_str[0]
        #     new_df.loc[:,'base_cls'] = split_str[1]
        #     new_df.loc[:,'base_bag'] = split_str[2]
        #     new_df['fold'] = fold
        #     new_df['stacker'] = stacker_name
        #     # print(new_df.to_string())
        #     stacked_df = pd.concat([stacked_df, new_df])
        if hasattr(stacker, "predict_proba") and (not regression):
            test_predictions = stacker.predict_proba(test_df)[:, 1]
            train_predictions = stacker.predict_proba(train_df)[:, 1]
        else:
            test_predictions = stacker.predict(test_df)
            train_predictions = stacker.predict(train_df)
            if regression is False:
                test_predictions = test_predictions[:, 1]
                train_predictions = train_predictions[:, 1]

    # print(stacker.coef_)
    if stacker_name == "LR.S":
        print(train_df.columns)
        print(stacker.coef_)
        coefs = pd.DataFrame(data=list(stacker.coef_), columns=train_df.columns, index=[0])
        coefs['fold'] = fold
    else:
        coefs = None

    df = pd.DataFrame(
        {'fold': fold, 'id': test_df.index.get_level_values('id'), 'label': test_labels, 'prediction': test_predictions,
         'diversity': common.diversity_score(test_df.values)})
    return {'testing_df':df, "training": [train_labels, train_predictions], 'train_dfs': [train_df, train_labels],
            'stacked_df':stacked_df, 'coefs':coefs}

def plot_scatter(df, path, x_col, y_col, hue_col, fn, title):
    fig, ax = plt.subplots(1,1)
    ax = sns.scatterplot(ax=ax, data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.7)
    ax.set_title(title)
    fig.savefig(path+'/'+fn, bbox_inches="tight")


def main_classification(path, f_list, agg=1):
    #
    dn = abspath(path).split('/')[-1]
    # cols = ['data_name', 'fmax', 'method']
    cols = ['data_name', 'fmax', 'method', 'auc', 'auprc']

    dfs = []
    aggregated_dict = {'CES': CES_fmax,
                       'Mean': aggregating_ensemble,
                       'best base': bestbase_fmax}

    for key, val in aggregated_dict.items():
        print('[{}] Start building model #################################'.format(key))
        perf = val(path, fold_values, agg)
        if key != 'best base':
            fmax_perf = perf['f-measure']['F']
        else:
            fmax_perf = perf['f-measure']
        auc_perf = perf['auc']
        auprc_perf = perf['auprc']
        print('[{}] Finished evaluating model ############################'.format(key))
        print('[{}] F-max score is {}.'.format(key, fmax_perf))
        print('[{}] AUC score is {}.'.format(key, auc_perf) )
        print('[{}] AUPRC score is {}.'.format(key, auprc_perf))
        dfs.append(pd.DataFrame(data=[[dn, fmax_perf, key, auc_perf, auprc_perf]], columns=cols, index=[0]))


    # print('[CES] Start building model #################################')
    # ces = CES_fmax(path, fold_values, agg)
    # print('[CES] Finished evaluating model ############################')
    # print('[CES] F-max score is %s.' % ces['f-measure']['F'])
    # print('[CES] AUC score is %s.' % ces['auc'])
    # print('[Mean] Start building model ################################')
    # mean = aggregating_ensemble(path, fold_values, agg)
    # print('[Mean] Finished evaluating model ###########################')
    # print('[Mean] F-max score is %s.' % mean['f-measure']['F'])
    # print('[Mean] AUC score is %s.' % mean['auc'])
    # print('[Best Base] Start building model ###########################')
    # bestbase = bestbase_fmax(path, fold_values, agg)
    # print('[Best Base] Finished evaluating model ######################')
    # print('[Best Base] F-max score is %s.' % bestbase['f-measure'])
    # print('[Best Base] AUC score is %s.' % bestbase['auc'])
    # dfs.append(pd.DataFrame(data=[[dn, ces['f-measure']['F'], 'CES', ces['auc']]], columns=cols, index=[0]))
    # dfs.append(pd.DataFrame(data=[[dn, mean['f-measure']['F'], 'Mean', mean['auc']]], columns=cols, index=[0]))
    # dfs.append(pd.DataFrame(data=[[dn, bestbase['f-measure'], 'best base', bestbase['auc']]], columns=cols, index=[0]))
    print('Saving results #############################################')
    analysis_path = '%s/analysis' % path
    if not exists(analysis_path):
        mkdir(analysis_path)
    # Get Stacking Fmax scores
    # stackers = [RandomForestClassifier(n_estimators=200, max_depth=2, bootstrap=False, random_state=0),
    #             SVC(C=1.0, cache_size=10000, class_weight=None, coef0=0.0,
    #                 decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', probability=True,
    #                 max_iter=1e8, random_state=None, shrinking=True,
    #                 tol=0.001, verbose=False), GaussianNB(), LogisticRegression(), AdaBoostClassifier(),
    #             DecisionTreeClassifier(), GradientBoostingClassifier(loss='deviance'), KNeighborsClassifier(),
    #             XGBClassifier(),
    #             # MLPClassifier(), GaussianProcessClassifier()
    #             ]
    stackers_dict = {
                     "RF.S": RandomForestClassifier(),
                     # "RF_CV.S": GridSearchCV(RandomForestClassifier(),
                     #                      param_grid={'max_features':['auto','sqrt','log2'],
                     #                                  'criterion': ['gini', 'entropy'],
                     #                                  'bootstrap': [True, False],
                     #                                  'oob_score': [True, False],
                     #                                  'class_weight':['balanced', 'balanced_subsample']}, scoring=fmax_sklearn),
                     "SVM.S": SVC(kernel='linear', probability=True),
                     # "SVM_CV.S": GridSearchCV(SVC(kernel='linear', probability=True),
                     #                       param_grid={'C': [0.25, 0.5, 1, 2, 5]},
                     #                          scoring=fmax_sklearn),
                     # "KernelSVM.S": SVC(kernel='rbf', probability=True),
                     # "KernelSVM_CV.S": GridSearchCV(SVC(kernel='rbf', probability=True),
                     #                         param_grid={'C': [0.25, 0.5, 1, 2, 5],
                     #                                     'gamma': ['scale', 'auto']},
                     #                                scoring=fmax_sklearn),
                     "NB.S": GaussianNB(),
                     # "NB_CV.S": GridSearchCV(GaussianNB(),
                     #                         param_grid={'var_smoothing': [1e-5, 1e-7, 1e-9, 1e-11, 1e-13]},
                     #                         scoring=fmax_sklearn),
                     "LR.S": LogisticRegression(),
                     # "LR_CV.S": GridSearchCV(LogisticRegression(),
                     #                      param_grid={'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                     #                                  'C': [0.25, 0.5, 1, 2, 5],
                     #                                  },
                     #                         scoring=fmax_sklearn),
                     "AdaBoost.S": AdaBoostClassifier(),
                     # "AdaBoost_CV.S": GridSearchCV(AdaBoostClassifier(),
                     #                               param_grid={'learning_rate':[0.1, 0.25, 0.5, 0.75, 1]},
                     #                               scoring=fmax_sklearn),
                     "DT.S": DecisionTreeClassifier(),
                     # "DT_CV.S": GridSearchCV(DecisionTreeClassifier(),
                     #                         param_grid={'criterion': ['gini', 'entropy'],
                     #                                     'splitter': ["best", "random"],
                     #                                     'max_features': ['auto', 'sqrt', 'log2'],
                     #                                     },
                     #                         scoring=fmax_sklearn),
                     "GradientBoosting.S": GradientBoostingClassifier(),
                     # "GradientBoosting_CV.S": GridSearchCV(GradientBoostingClassifier(),
                     #                                    param_grid={'loss':['deviance', 'exponential'],
                     #                                                'criterion':['friedman_mse', 'mse', 'mae'],
                     #                                                'max_features': ['auto', 'sqrt', 'log2']},
                     #                                    scoring=fmax_sklearn),
                     "KNN.S": KNeighborsClassifier(),
                     # "KNN_CV.S": GridSearchCV(KNeighborsClassifier(),
                     #                          param_grid={'weights':['uniform', 'distance'],
                     #                    'p':[1, 2],
                     #                    'n_neighbors': [2,3,5,10,15],
                     #                    },
                     #                    scoring=fmax_sklearn),
                     "XGB.S": XGBClassifier(),
                     # "XGB_CV.S": GridSearchCV(XGBClassifier(),
                     #                          param_grid={"objective": ['reg:squarederror',
                     #                                                'reg:squaredlogerror',
                     #                                                'reg:pseudohubererror',
                     #                                                'reg:gamma',
                                                                    # 'reg:tweedie'
                                                                    # ]}
                                              # )

                        # MLPClassifier(), GaussianProcessClassifier()
                    }
    # stacker_names = ["RF.S", "SVM.S", "NB.S", "LR.S", "AB.S", "DT.S", "LB.S", "KNN.S",
    #                  "XGB.S",
    #                  # ,"MLP.S", "GP.S"
    #                  ]
    # stacker_names_feat_imp = ['{}_stacked_feat_imp'.format(s) for s in stacker_names]
    df_cols = ['f_train_base','f_test_base', 'fold', 'stacker',
               'feat_imp', 'base_data', 'base_cls', 'base_bag']
    stacked_df = pd.DataFrame(columns= df_cols)
    # for i, (stacker_name, stacker) in enumerate(zip(stacker_names, stackers)):
    for i, (stacker_name, stacker) in enumerate(stackers_dict.items()):
        print('[%s] Start building model ################################' % (stacker_name))


        if (not testing_bool):
            stacking_output = []
            for fold in f_list:
                stack = stacked_generalization(path, stacker_name, stacker, fold, agg, stacked_df)
                stacked_df = stack.pop('stacked_df')
                # if fold == 1:
                #     print(fold)
                stacking_output.append(stack)
        else:
            stacking_output = [stacked_generalization(path, stacker_name, stacker, '67890', agg, stacked_df)]
            stacked_df = stacking_output[0].pop('stacked_df')
        predictions_dfs = [s['testing_df'] for s in stacking_output]
        if stacker_name == 'LR.S':
            # coef_dfs = [s['coefs'] for s in stacking_output]
            # coef_cat_df = pd.concat(coef_dfs)
            # coef_cat_df.to_csv(os.path.join(analysis_path, 'coefs_lr.csv'))
            # training_dfs = pd.concat([s['train_dfs'][0] for s in stacking_output])
            # training_labels = pd.concat([pd.DataFrame({'label':s['train_dfs'][1]}) for s in stacking_output])
            training_dfs = stacking_output[0]['train_dfs'][0]
            training_labels = pd.DataFrame({'label': stacking_output[0]['train_dfs'][1]})

            # training_dfs_diff_to_label = training_dfs
            # training_dfs_diff_to_label[:] = abs(training_dfs_diff_to_label.values - training_labels)
            print(training_dfs.shape)

            # stacker.fit(training_dfs, training_labels)
            stacker.fit(training_dfs, training_labels)
            predict_scores = stacker.predict_proba(training_dfs)[:,1]
            fmax_train_c = common.fmeasure_score(training_labels.values, predict_scores, thres=None)
            fmax_train_sk = fmax_sklearn(stacker, training_dfs, training_labels)
            auc_train = sklearn.metrics.roc_auc_score(training_labels.values, predict_scores)
            print('fmax_sk of the whole training set:', fmax_train_sk)
            print('fmax_c of the whole training set:', fmax_train_c)
            print('auc of the whole training set:', auc_train)
            n_repeats = 100
            stacker_pi = permutation_importance(estimator=stacker,
                                               X=training_dfs,
                                               y=training_labels,
                                           n_repeats=n_repeats,
                                            random_state=0,
                                               scoring = fmax_sklearn
                                                )
            print(stacker_pi.importances_mean)
            print(stacker_pi.importances_mean.shape)
            print(stacker.coef_.shape)
            # pi_df = pd.DataFrame(data=stacker_pi.importances.T, columns=training_dfs.columns, index=range(n_repeats))
            pi_df = pd.DataFrame(data=[stacker_pi.importances_mean], columns=training_dfs.columns, index=[0])
            coefs = pd.DataFrame(data=stacker.coef_, columns=training_dfs.columns, index=[0])
            # coef_cat_df = pd.concat(coef_dfs)
            coefs.to_csv(os.path.join(analysis_path, 'coefs_lr.csv'))
            pi_df.to_csv(os.path.join(analysis_path, 'coefs_lr_pi.csv'))



        # _dfs = [s['testing_df'] for s in stacking_output]
        _training = stacking_output[0]['training']
        thres = thres_fmax(_training[0], _training[1])

        predictions_df = pd.concat(predictions_dfs)
        print(thres)
        fmax = common.fmeasure_score(predictions_df.label, predictions_df.prediction, thres)
        print(fmax)
        auc = sklearn.metrics.roc_auc_score(predictions_df.label, predictions_df.prediction)
        auprc = common.auprc(predictions_df.label, predictions_df.prediction)
        print('[%s] Finished evaluating model ###########################' % (stacker_name))
        print('[%s] F-measure score is %s.' % (stacker_name, fmax['F']))
        if 'P' in fmax:
            print('[%s] Precision score is %s.' % (stacker_name, fmax['P']))
            print('[%s] Recall score is %s.' % (stacker_name, fmax['R']))
        print('[%s] AUC score is %s.' % (stacker_name, auc))
        print('[%s] AUPRC score is %s.' % (stacker_name, auprc))
        df = pd.DataFrame(data=[[dn, fmax['F'], stacker_name, auc, auprc]], columns=cols, index=[0])
        dfs.append(df)
    dfs = pd.concat(dfs)

    # hue_list = ['stacker','base_data', 'base_cls']
    # y_list = ['f_train_base','f_test_base']
    # x_list = ['feat_imp']
    # plot_path = './plot/feat_imp_'+path.split('/')[-1]
    # common.check_dir_n_mkdir(plot_path)
    # params_list = list(product(x_list, y_list, hue_list))
    # # print(stacked_df)
    # for params in params_list:
    #     x, y, hue = params
    #     fn = 'scatter_{}_by_{}'
    #     title = 'F measure of {} base classifier VS Feature Importance of stackers (by {})'
    #     if 'train' in y:
    #         fn = fn.format('train', hue)
    #         title = title.format('train', hue)
    #     else:
    #         fn = fn.format('test', hue)
    #         title = title.format('test', hue)
    #     plot_scatter(df=stacked_df, x_col=x, y_col=y, hue_col=hue, fn=fn, path=plot_path, title=title)

    # Save results

    dfs.to_csv(os.path.join(analysis_path, "performance.csv"), index=False)

def main_regression(path, f_list, agg=1, inference_only=False):
    #
    dn = abspath(path).split('/')[-1]
    # cols = ['data_name', 'fmax', 'method']
    # cols = ['data_name', 'fmax', 'method', 'auc']
    # cols = ['data_name', 'weighted_mse', 'method']
    cols = ['data_name', 'weighted_mse', 'method_name', 'by_subject', 'Z_scoring','params_optimized' , 'cls_name']
    dfs = []


    # print('[CES] Start building model #################################')
    # ces = CES(path, fold_values, agg, regression=True)
    # print('[CES] Finished evaluating model ############################')
    # print('[CES] Weighted MSE score is %s.' % ces['weighted_mse'])
    # print('[Mean] Start building model ################################')
    # mean = aggregating_ensemble(path, fold_values, agg, regression=True)
    # print('[Mean] Finished evaluating model ###########################')
    # print('[Mean] Weighted MSE score is %s.' % mean['weighted_mse'])

    # print('[Median.Z_scoring] Start building model ################################')
    # median = aggregate_weighted_mse(path, fold_values, agg, median=True, inference_only=inference_only)
    # median_Z = aggregating_ensemble(path, fold_values, agg,
    #                               median=True, inference_only=inference_only, regression=True,
    #                               z_scoring=True)
    # print('[Median.Z_scoring] Finished evaluating model ###########################')
    # print('[Median.Z_scoring] Weighted MSE score is %s.' % median_Z['weighted_mse'])

    print('[Median] Start building model ################################')
    # median = aggregate_weighted_mse(path, fold_values, agg, median=True, inference_only=inference_only)
    median = aggregating_ensemble(path, fold_values, agg,
                                  median=True, inference_only=inference_only, regression=True)
    print('[Median] Finished evaluating model ###########################')
    print('[Median] Weighted MSE score is %s.' % median['weighted_mse'])
    # print('[Best Base] Start building model ###########################')
    # bestbase = best_base_predictors(path, fold_values, agg, regression=True)
    # print('[Best Base] Finished evaluating model ######################')
    # print('[Best Base] Weighted MSE score is %s.' % bestbase['weighted_mse'])
    # dfs.append(pd.DataFrame(data=[[dn, ces['weighted_mse'], 'CES', False, False, False,'CES']], columns=cols, index=[0]))
    # dfs.append(pd.DataFrame(data=[[dn, mean['weighted_mse'], 'Mean', False, False, False,'Mean']], columns=cols, index=[0]))
    # dfs.append(pd.DataFrame(data=[[dn, median['weighted_mse'], 'Median', False, False,False, 'Median']], columns=cols, index=[0]))
    # dfs.append(pd.DataFrame(data=[[dn, median_Z['weighted_mse'], 'Median.Z_scoring', False, True,False, 'Median']], columns=cols, index=[0]))
    # dfs.append(pd.DataFrame(data=[[dn, bestbase['weighted_mse'], 'best base ({})'.format(bestbase['name']), False, False,False, 'best base']], columns=cols, index=[0]))
    # Get Stacking Fmax scores
    z_scoring_params = [True, False]
    # z_scoring_params = [True]
    stackers_dict = {
                    # "RF_L1.S": RandomForestRegressor(criterion='mae'),
                    # "RF_L1.CV.S": GridSearchCV(RandomForestRegressor(criterion='mae'),
                    #                          param_grid={
                    #                                      'max_features':['auto','sqrt','log2']}),
                    # # "RF_L2.S": RandomForestRegressor(),
                    #  "RF_L2.CV.S": GridSearchCV(RandomForestRegressor(),
                    #                          param_grid={'max_features':['auto','sqrt','log2']}),
                    # # "SVM_L1.S": LinearSVR(),
                    # "SVM_L1.CV.S": GridSearchCV(LinearSVR(),
                    #                           param_grid={'C': [0.25, 0.5, 1, 2, 5]}),
                    # # "SVM_L2.S": LinearSVR(loss='squared_epsilon_insensitive'),
                    # "SVM_L2.CV.S": GridSearchCV(LinearSVR(loss='squared_epsilon_insensitive'),
                    #                          param_grid={'C': [0.25, 0.5, 1, 2, 5]}),
                    # "OLS.S": LinearRegression(),
                    # # "AdaBoost.S": AdaBoostRegressor(),
                    # "AdaBoost.CV.S": GridSearchCV(AdaBoostRegressor(),
                    #                             param_grid={'learning_rate':[0.1, 0.25, 0.5, 0.75, 1],
                    #                                         'loss': ['linear', 'square', 'exponential'],
                    #                                         }),
                    # # "DT.S": DecisionTreeRegressor(),
                    #  "DT.CV.S": GridSearchCV(DecisionTreeRegressor(),
                    #                       param_grid={'criterion':['friedman_mse', 'mse', 'mae']}),
                    # # "GradientBoosting.S": GradientBoostingRegressor(),
                    #  "GradientBoosting.CV.S": GridSearchCV(GradientBoostingRegressor(),
                    #          param_grid={'loss': ['ls', 'lad', 'huber', 'quantile'],
                    #                      'criterion':['friedman_mse', 'mse', 'mae'],
                    #                      },
                    #          # scoring=wmse_sklearn
                    #          ),
                    # # "KNN.S": KNeighborsRegressor(),
                    #  "KNN.CV.S": GridSearchCV(KNeighborsRegressor(),
                    #         param_grid={'weights':['uniform', 'distance'],
                    #                     'p':[1, 2],
                    #                     'n_neighbors': [2,3,5,10,15],
                    #                     },
                    #          # scoring=wmse_sklearn
                    #          ),
                    # #  "XGB.S": XGBRegressor(),
                    # "XGB.CV.S": GridSearchCV(XGBRegressor(),
                    #                       param_grid={"objective": ['reg:squarederror',
                    #                                                 'reg:squaredlogerror',
                    #                                                 'reg:pseudohubererror',
                    #                                                 # 'reg:gamma',
                    #                                                 'reg:tweedie'
                    #                                                 ]}),
                    # #  "Huber.S": HuberRegressor(),
                    #  "Huber.CV.S": GridSearchCV(HuberRegressor(),
                    #         param_grid={"epsilon":[1.0, 1.2,  1.35, 1.5, 2],
                    #             'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
                    #         # scoring=wmse_sklearn
                    #          ),
                    # #  "ElasticNet.S": ElasticNet(),
                    #  "ElasticNet.CV.S": ElasticNetCV(),
                    # #  "Lasso.S": Lasso(),
                    #  "Lasso.CV.S": LassoCV(),
                    # #  "BayesianRidge.S": BayesianRidge(),
                    #  "BayesianRidge.CV.S": GridSearchCV(BayesianRidge(),
                    #                                     param_grid={'tol': [1e-2, 1e-3, 1e-4],
                    #                                                 'alpha_1':[1e-5,1e-6,1e-7],
                    #                                                 'alpha_2':[1e-5,1e-6,1e-7],
                    #                                                 'lambda_1':[1e-5,1e-6,1e-7],
                    #                                                 'lambda_2':[1e-5,1e-6,1e-7]
                    #                                                 }),
                    # #  "Lars.S": Lars(),
                    #  "Lars.CV.S": LarsCV(),
                     "Ridge.S": Ridge()
                     # "Ridge.CV.S": RidgeCV(),
                    #  # "LassoLarsCV.S": LassoLarsCV(),
                    # #  # "RANSAC.S",
                    # #  "TheilSen.S": TheilSenRegressor(),
                    #  "TheilSen.CV.S": GridSearchCV(TheilSenRegressor(),
                    #                             param_grid={'tol':[1e-2, 1e-3, 1e-4, 1e-5],
                    #                                         }),
                    # #  "MLP.S": MLPRegressor(),
                    #  "MLP.CV.S": GridSearchCV(MLPRegressor(),
                    #                        param_grid={
                    #                                    'alpha': [1e-5, 0.0001, 1e-3,],
                    #                                    'learning_rate':['constant','invscaling','adaptive'],
                    #                                    'learning_rate_init':[0.01, 0.001, 0.005],
                    #                                    'early_stopping': [True, False]}),
                    #  # "ARD.S": ARDRegression(),
                    #  "ARD.CV.S": GridSearchCV(ARDRegression(),
                    #                           param_grid={'tol': [1e-2, 1e-3, 1e-4],
                    #                                         'alpha_1':[1e-5,1e-6,1e-7],
                    #                                         'alpha_2':[1e-5,1e-6,1e-7],
                    #                                         'lambda_1':[1e-5,1e-6,1e-7],
                    #                                         'lambda_2':[1e-5,1e-6,1e-7],
                    #                                       'threshold_lambda': [1e3, 1e4, 1e5]
                    #                                       }),
                    # # "KernelSVM_RBF.S": sklearn.svm.SVR(),
                    # "KernelSVM_RBF.CV.S": GridSearchCV(sklearn.svm.SVR(),
                    #                                    param_grid={'C':[0.1, 1.0, 2.0, 5.0, 10.0],
                    #                                         'epsilon':[0.1, 0.5, 1.0]},
                    #          # scoring=wmse_sklearn
                    # )
                    }
    stackers = [v for k, v in stackers_dict.items()]
    stacker_names_ref = [k for k, v in stackers_dict.items()]

    for z_scoring_param in z_scoring_params:
        # print(z_scoring_params)
        # print(1)
        if z_scoring_param:
            stacker_names = [s+'.Z_scoring' for s in stacker_names_ref]
        else:
            stacker_names = stacker_names_ref

        # stacker_names_feat_imp = ['{}_stacked_feat_imp'.format(s) for s in stacker_names]
        df_cols = ['wmse_train_base','wmse_test_base', 'fold', 'stacker',
                   'feat_imp', 'base_data', 'base_cls', 'base_bag']
        stacked_df = pd.DataFrame(columns= df_cols)
        for i, (stacker_name, stacker) in enumerate(zip(stacker_names, stackers)):
            print('[%s] Start building model ################################' % (stacker_name))
            if (not testing_bool):
                stacking_output = []
                for fold in f_list:
                    stack = stacked_generalization(path, stacker_name, stacker, fold, agg, stacked_df,
                                                   regression=True, z_scoring=z_scoring_param)
                    stacked_df = stack.pop('stacked_df')
                    stacking_output.append(stack)
            else:
                stacking_output = [stacked_generalization(path, stacker_name, stacker, '67890', agg, stacked_df,
                                                          regression=True, z_scoring=z_scoring_param)]
                stacked_df = stacking_output[0].pop('stacked_df')
            predictions_dfs = [s['testing_df'] for s in stacking_output]
            _training = stacking_output[0]['training']

            predictions_df = pd.concat(predictions_dfs)
            predictions_df.set_index('id', inplace=True)
            # print(predictions_df)
            if inference_only is True:
                output_df = predictions_df.prediction.to_frame()
                output_df.reset_index(inplace=True)
                output_df.rename(columns={'id': 'measurement_id'}, inplace=True)
                output_df = output_df[['measurement_id', 'prediction']]
                # output_df.rename({'id':'measurement_id'}, inplace=True)
                output_df.to_csv("{}/{}/{}{}".format(path, "analysis", stacker_name.replace('.', '_'), '_test_predictions.csv'),
                                 index=False)
            else:
                weighted_mse_score = weighted_mse(predictions_df.label, predictions_df.prediction)

                print('[%s] Finished evaluating model ###########################' % (stacker_name))
                print('[%s] Weighted MSE score is %s.' % (stacker_name, weighted_mse_score))
                params_optimized = '.CV.S' in stacker_names_ref[i]
                df = pd.DataFrame(data=[[dn, weighted_mse_score, stacker_name, False, z_scoring_param, params_optimized, stacker_names_ref[i]]], columns=cols, index=[0])
                dfs.append(df)


    # """
    # 1 Model per subject
    # """
    # print('[CES.by_subject] Start building model #################################')
    # ces_by_subject = CES(path, fold_values, agg, subject_model=True, regression=True)
    # print('[CES.by_subject] Finished evaluating model ############################')
    # print('[CES.by_subject] Weighted MSE score is %s.' % ces_by_subject['weighted_mse'])
    # # # print('[Mean_by_subject] Start building model ################################')
    # # # mean_by_subject = mean_weighted_mse(path, fold_values, agg, subject_model=True)
    # # # print('[Mean_by_subject] Finished evaluating model ###########################')
    # # # print('[Mean_by_subject] Weighted MSE score is %s.' % mean['weighted_mse'])
    # # # print('[Median_by_subject] Start building model ################################')
    # # # median_by_subject = mean_weighted_mse(path, fold_values, agg, median=True, subject_model=True)
    # # # print('[Median_by_subject] Finished evaluating model ###########################')
    # # # print('[Median_by_subject] Weighted MSE score is %s.' % median['weighted_mse'])
    # dfs.append(pd.DataFrame(data=[[dn, ces_by_subject['weighted_mse'], 'CES.by_subject', True, False, False, 'CES']], columns=cols, index=[0]))
    # # # dfs.append(pd.DataFrame(data=[[dn, mean_by_subject['weighted_mse'], 'Mean_by_subject']], columns=cols, index=[0]))
    # # # dfs.append(pd.DataFrame(data=[[dn, median_by_subject['weighted_mse'], 'Median_by_subject']], columns=cols, index=[0]))
    # # # dfs.append(pd.DataFrame(data=[[dn, bestbase['weighted_mse'], 'best base']], columns=cols, index=[0]))
    # # # Get Stacking Fmax scores
    # #
    # for z_scoring_param in z_scoring_params:
    #     stacker_names = stacker_names_ref
    #     if z_scoring_param:
    #         stacker_names = [s+'.Z_scoring' for s in stacker_names_ref]
    #     stacker_names = [s+'.by_subject' for s in stacker_names]
    #
    #     # stacker_names_feat_imp = ['{}_stacked_feat_imp'.format(s) for s in stacker_names]
    #     df_cols = ['wmse_train_base', 'wmse_test_base', 'fold', 'stacker',
    #                'feat_imp', 'base_data', 'base_cls', 'base_bag']
    #     stacked_df = pd.DataFrame(columns=df_cols)
    #
    #     for i, (stacker_name, stacker) in enumerate(zip(stacker_names, stackers)):
    #         print('[%s] Start building model ################################' % (stacker_name))
    #
    #         if (not testing_bool):
    #             stacking_output = []
    #             for fold in f_list:
    #                 stack = stacked_generalization(path, stacker_name, stacker, fold, agg, stacked_df,
    #                                                subject_model=True, regression=True, z_scoring=z_scoring_param)
    #                 stacked_df = stack.pop('stacked_df')
    #                 stacking_output.append(stack)
    #         else:
    #             stacking_output = [
    #                 stacked_generalization(path, stacker_name, stacker, '67890', agg, stacked_df,
    #                                        subject_model=True,regression=True, z_scoring=z_scoring_param)]
    #             stacked_df = stacking_output[0].pop('stacked_df')
    #         predictions_dfs = [s['testing_df'] for s in stacking_output]
    #         _training = stacking_output[0]['training']
    #
    #         predictions_df = pd.concat(predictions_dfs)
    #         predictions_df.set_index('id', inplace=True)
    #         # print(predictions_df)
    #         weighted_mse_score = weighted_mse(predictions_df.label, predictions_df.prediction)
    #
    #         print('[%s] Finished evaluating model ###########################' % (stacker_name))
    #         print('[%s] Weighted MSE score is %s.' % (stacker_name, weighted_mse_score))
    #         params_optimized = '.CV.S' in stacker_names_ref[i]
    #         df = pd.DataFrame(data=[[dn, weighted_mse_score, stacker_name, True, z_scoring_param, params_optimized, stacker_names_ref[i]]], columns=cols, index=[0])
    #         dfs.append(df)
    #
    # dfs = pd.concat(dfs)
    #
    # # hue_list = ['stacker','base_data', 'base_cls']
    # # y_list = ['f_train_base','f_test_base']
    # # x_list = ['feat_imp']
    # # plot_path = './plot/feat_imp_'+path.split('/')[-1]
    # # common.check_dir_n_mkdir(plot_path)
    # # params_list = list(product(x_list, y_list, hue_list))
    # # # print(stacked_df)
    # # for params in params_list:
    # #     x, y, hue = params
    # #     fn = 'scatter_{}_by_{}'
    # #     title = 'F measure of {} base classifier VS Feature Importance of stackers (by {})'
    # #     if 'train' in y:
    # #         fn = fn.format('train', hue)
    # #         title = title.format('train', hue)
    # #     else:
    # #         fn = fn.format('test', hue)
    # #         title = title.format('test', hue)
    # #     plot_scatter(df=stacked_df, x_col=x, y_col=y, hue_col=hue, fn=fn, path=plot_path, title=title)
    #
    # # Save results
    # print('Saving results #############################################')
    # if not exists('%s/analysis' % path):
    #     mkdir('%s/analysis' % path)
    # path_suffix = path.split('/')[-1]
    # dfs.sort_values(by='weighted_mse').to_csv("%s/analysis/performance-%s.csv" % (path,path_suffix), index=False)


### parse arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--fold', '-F', type=int, default=5, help='cross-validation fold')
parser.add_argument('--aggregate', '-A', type=int, default=1, help='if aggregate is needed, feed bagcount, else 1')
parser.add_argument('--regression', '-reg', type=str2bool, default='False', help='Regression or Classification')
parser.add_argument('--inference_only', '-infer', type=str2bool, default='False', help='Regression or Classification')
args = parser.parse_args()
data_path = abspath(args.path)
# fns = listdir(data_path)
# excluding_folder = ['analysis']
# fns = [fn for fn in fns if not fn in excluding_folder]
# fns = [fn for fn in fns if not 'tcca' in fn]
# fns = [fn for fn in fns if fn != 'analysis']
# fns = [data_path + '/' + fn for fn in fns]
# feature_folders = [fn for fn in fns if isdir(fn)]

feature_folders = common.data_dir_list(data_path)
if len(feature_folders) == 0:
    feature_folders = common.data_dir_list(os.path.join(data_path, '../'))
assert len(feature_folders) > 0
### get weka properties from weka.properties
p = load_properties(data_path)
# # fold_values = range(int(p['foldCount']))
assert ('foldAttribute' in p) or ('foldCount' in p)
if 'foldAttribute' in p:
    # input_fn = '%s/%s' % (feature_folders[0], 'data.arff')
    # assert exists(input_fn)
    # headers = load_arff_headers(input_fn)
    # fold_values = headers[p['foldAttribute']]
    df = common.read_arff_to_pandas_df(os.path.join(feature_folders[0],'data.arff'))
    fold_values = df[p['foldAttribute']].unique()
else:
    fold_values = range(int(p['foldCount']))
# print(fold_values)
# pca_fold_values = ['pca_{}'.format(fv) for fv in fold_values]
# fold_values = ['validation']
# fold_values = ['test']
testing_bool = ('67890' in fold_values and 'foldAttribute' in p)


if args.regression:
    main_regression(args.path, fold_values, args.aggregate, args.inference_only)
else:
    main_classification(args.path, fold_values, args.aggregate)
    # main(os.path.join(args.path, 'pca_EI'), pca_fold_values, args.aggregate)
