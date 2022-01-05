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

from sklearn.metrics import fbeta_score, make_scorer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVR
# from beat_pd_scoring_function import weighted_mse, \
#     measurement_subject_df, measurement_id_col, subject_id_col
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

# wmse_sklearn = make_scorer(weighted_mse, greater_is_better=False)
fmax_sklearn = make_scorer(common.f_max, greater_is_better=True, needs_proba=True)
auprc_sklearn = make_scorer(common.auprc, greater_is_better=True, needs_proba=True)

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
    # if regression:
    #     score = weighted_mse(labels, predictions)
    # else:
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
        # best_candidate = candidates[common.argbest(candidate_scores)]
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
#     return get_pred
#     ictions(test_df, best_ensemble, fold, seedval), \
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
           get_predictions(train_df, best_ensemble, fold, seedval), \
           best_ensemble, train_df

def thres_fmax(train_label_df, train_pred_df, testing_bool=False):
    if testing_bool:
        fmax_training = common.fmeasure_score(train_label_df, train_pred_df)
        thres = fmax_training['thres']
    else:
        thres = None

    return thres

def CES_classifier(path, fold_count=range(5), agg=1, attr_imp=False):
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
    best_ensembles = []

    for seedval in seeds:
        # for fold in range(fold_count):
        for fold in fold_count:
            # if '67890' in fold:
            # if testing_bool or (not 'foldAttribute' in p):
            pred_df, perf_df, train_pred_df, best_ensemble, train_df = method_function(fold, seedval, path, agg)
            predictions_dfs.append(pred_df)
            train_predictions_dfs.append(train_pred_df)
            performance_dfs.append(perf_df)
            thres = thres_fmax(train_pred_df.label, train_pred_df.prediction)
            if attr_imp:
                if fold == 1:
                    best_ensembles.append(best_ensemble)
    performance_df = pd.concat(performance_dfs)
    performance_df.to_csv('%s/analysis/selection-%s-%s-iterations.csv' % (path, method, 'fmax'), index=False)
    predictions_df = pd.concat(predictions_dfs)
    predictions_df['method'] = method
    predictions_df['metric'] = 'fmax'
    predictions_df.to_csv('%s/analysis/selection-%s-%s.csv' % (path, method, 'fmax'), index=False)
    auc = sklearn.metrics.roc_auc_score(predictions_df.label, predictions_df.prediction)
    auprc = common.auprc(predictions_df.label, predictions_df.prediction)
    print(thres)
    fmax = (common.fmeasure_score(predictions_df.label, predictions_df.prediction, thres=thres))
    if attr_imp:
        print(best_ensembles[0])
        frequency_bp_selected = best_ensembles[0].value_counts()
        local_model_weight_df = pd.DataFrame(data=np.zeros(len(train_df.columns)), columns=train_df.columns, index=[0])
        # for bp, freq in frequency_bp_selected.items():
        for bp in list(train_df.columns):
            if bp.split('.')[0] in frequency_bp_selected.index:
                local_model_weight_df[bp] = frequency_bp_selected[bp.split('.')[0]]
        local_model_weight_df['ensemble_method'] = 'CES'
    else:
        local_model_weight_df = None
    return {'f-measure':fmax, 'auc':float(auc), 'auprc':auprc, 'model_weight': local_model_weight_df}

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
        # else:
        #     weighted_mse_score = weighted_mse(predictions_df['label'].values, predictions_scores)
        #     return {'weighted_mse': weighted_mse_score}
    else:
        auc = '%.3f' % (sklearn.metrics.roc_auc_score(predictions_df.label, predictions_df.prediction))
        fmax = (common.fmeasure_score(predictions_df.label, predictions_df.prediction, thres=thres))
        return {'f-measure': fmax, 'auc': float(auc)}


# m

def aggregating_ensemble(path, fold_count=range(5), agg=1, attr_imp=False, median=False,
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
        train_df, train_label, test_df, test_label = common.read_fold(path, fold)
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
        # if regression:
        #     weighted_mse_score = weighted_mse(labels, predictions)
        #     return {'weighted_mse': weighted_mse_score}
        # else:
        # thres = thres_fmax(train_label, _unbag_mean(train_df))
        thres = thres_fmax(train_labels, train_dfs)

        fmax = common.fmeasure_score(labels, predictions, thres)
        auc = sklearn.metrics.roc_auc_score(labels, predictions)
        auprc = common.auprc(labels, predictions)

        if attr_imp:
            # frequency_bp_selected = best_ensembles[0]['ensemble'].value_counts()
            local_model_weight_df = pd.DataFrame(data=np.ones(len(train_df.columns)), columns=train_df.columns,
                                                 index=[0])
            local_model_weight_df['ensemble_method'] = 'mean'
        else:
            local_model_weight_df = None

        return {'f-measure': fmax, 'auc': auc, 'auprc': auprc, 'model_weight': local_model_weight_df}



def bestbase_classifier(path, fold_count=range(5), agg=1, attr_imp=False):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    predictions = []
    labels = []

    for fold in fold_count:
        train_df, train_label, test_df, label = common.read_fold(path, fold)
        test_df = common.unbag(test_df, agg)
        predictions.append(test_df)
        labels = append(labels, label)
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
    # if regression:
    #     weighted_mse_score_list = [weighted_mse(labels, predictions.iloc[:, i]) for i in
    #                                range(len(predictions.columns))]
    #     min_score = min(weighted_mse_score_list)
    #     name_cls = predictions.columns.tolist()[np.argmin(weighted_mse_score_list)]
    #     return {'weighted_mse': min_score, 'name': name_cls}
    # else:
        # thres = thres_fmax(train_label, common.unbag(train_df, agg))
    thres = None
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
    # test_df = common.unbag(test_df, agg)
    stacker = stacker.fit(train_df, train_labels)

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
        # print(stacker.coef_)
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


def main_classification(path, f_list, agg=1, attr_imp=False):
    #
    dn = abspath(path).split('/')[-1]
    # cols = ['data_name', 'fmax', 'method']
    cols = ['data_name', 'fmax', 'method', 'auc', 'auprc']

    dfs = []

    local_model_weight_dfs = []
    aggregated_dict = {'CES': CES_classifier,
                       'Mean': aggregating_ensemble,
                       'best base': bestbase_classifier}

    for key, val in aggregated_dict.items():
        print('[{}] Start building model #################################'.format(key))
        perf = val(path, fold_values, agg, attr_imp)
        if key != 'best base':
            fmax_perf = perf['f-measure']['F']
        else:
            fmax_perf = perf['f-measure']
            if attr_imp:
                local_model_weight_dfs.append(perf['model_weight'])
        auc_perf = perf['auc']
        auprc_perf = perf['auprc']
        print('[{}] Finished evaluating model ############################'.format(key))
        print('[{}] F-max score is {}.'.format(key, fmax_perf))
        print('[{}] AUC score is {}.'.format(key, auc_perf) )
        print('[{}] AUPRC score is {}.'.format(key, auprc_perf))
        dfs.append(pd.DataFrame(data=[[dn, fmax_perf, key, auc_perf, auprc_perf]], columns=cols, index=[0]))

    print('Saving results #############################################')
    analysis_path = '%s/analysis' % path
    if not exists(analysis_path):
        mkdir(analysis_path)
    """ Stacking Ensemble """
    stackers_dict = {
                     "RF.S": RandomForestClassifier(),
                     "SVM.S": SVC(kernel='linear', probability=True),
                     "NB.S": GaussianNB(),
                     "LR.S": LogisticRegression(),
                     "AdaBoost.S": AdaBoostClassifier(),
                     "DT.S": DecisionTreeClassifier(),
                     "GradientBoosting.S": GradientBoostingClassifier(),
                     "KNN.S": KNeighborsClassifier(),
                     "XGB.S": XGBClassifier()
                    }
    df_cols = ['f_train_base','f_test_base', 'fold', 'stacker',
               'feat_imp', 'base_data', 'base_cls', 'base_bag']
    stacked_df = pd.DataFrame(columns= df_cols)

    # for i, (stacker_name, stacker) in enumerate(zip(stacker_names, stackers)):
    for i, (stacker_name, stacker) in enumerate(stackers_dict.items()):
        print('[%s] Start building model ################################' % (stacker_name))

        # if (not testing_bool):
        stacking_output = []
        for fold in f_list:
            stack = stacked_generalization(path, stacker_name, stacker, fold, agg, stacked_df)
            stacked_df = stack.pop('stacked_df')
            if attr_imp:
                if fold == 1:
            #     print(fold)
                    stacking_output.append(stack)
            else:
                stacking_output.append(stack)
        # else:
            # stacking_output = [stacked_generalization(path, stacker_name, stacker, '67890', agg, stacked_df)]
            # stacked_df = stacking_output[0].pop('stacked_df')
        predictions_dfs = [s['testing_df'] for s in stacking_output]
        if attr_imp is True:
            training_dfs = stacking_output[0]['train_dfs'][0]
            training_labels = pd.DataFrame({'label': stacking_output[0]['train_dfs'][1]})
            print(training_dfs.shape)
            stacker.fit(training_dfs, training_labels)
            # predict_scores = stacker.predict_proba(training_dfs)[:,1]
            n_repeats = 100
            stacker_pi = permutation_importance(estimator=stacker,
                                               X=training_dfs,
                                               y=training_labels,
                                           n_repeats=n_repeats,
                                            random_state=0,
                                               scoring = auprc_sklearn
                                                )
            print(stacker_pi.importances_mean)
            print(stacker_pi.importances_mean.shape)
            pi_df = pd.DataFrame(data=[stacker_pi.importances_mean], columns=training_dfs.columns, index=[0])
            pi_df['ensemble_method'] = stacker_name
            local_model_weight_dfs.append(pi_df)



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
    if attr_imp is True:
        local_mr_df = pd.concat(local_model_weight_dfs)
        local_mr_df.to_csv(os.path.join(analysis_path, 'pi_stackers.csv'))


    """ Save results """

    dfs.to_csv(os.path.join(analysis_path, "performance.csv"), index=False)

### parse arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--fold', '-F', type=int, default=5, help='cross-validation fold')
parser.add_argument('--aggregate', '-A', type=int, default=1, help='if aggregate is needed, feed bagcount, else 1')
parser.add_argument('--regression', '-reg', type=str2bool, default='False', help='Regression or Classification')
# parser.add_argument('--inference_only', '-infer', type=str2bool, default='False', help='Inference to test set')
parser.add_argument('--attr_imp', type=str2bool, default='False', help='get the attribute importance from stacker')
args = parser.parse_args()
data_path = abspath(args.path)

feature_folders = common.data_dir_list(data_path)
if len(feature_folders) == 0:
    feature_folders = common.data_dir_list(os.path.join(data_path, '../'))
assert len(feature_folders) > 0
### get weka properties from weka.properties
p = load_properties(data_path)
# # fold_values = range(int(p['foldCount']))
assert ('foldAttribute' in p) or ('foldCount' in p)
if 'foldAttribute' in p:
    df = common.read_arff_to_pandas_df(os.path.join(feature_folders[0],'data.arff'))
    fold_values = df[p['foldAttribute']].unique()
else:
    fold_values = range(int(p['foldCount']))




main_classification(args.path, fold_values, args.aggregate, args.attr_imp)
