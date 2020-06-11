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
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.linear_model import SGDClassifier  # SGD
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.linear_model import LogisticRegression  # Logistic regression
from sklearn.ensemble import AdaBoostClassifier  # Adaboost
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.ensemble import GradientBoostingClassifier  # Logit Boost with parameter(loss='deviance')
from sklearn.neighbors import KNeighborsClassifier  # K nearest neighbors (IBk in weka)
from sklearn.svm import SVC
import sklearn
import warnings
from common import load_arff_headers, load_properties
from os.path import abspath, isdir
from os import remove, system, listdir
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


def checkFolder(path, fold_count=5):
    for fold in range(fold_count):
        if not exists('%s/predictions-%d.csv.gz' % (path, fold)):
            return False
        if not exists('%s/validation-%d.csv.gz' % (path, fold)):
            return False
    return True


def get_performance(df, ensemble, fold, seedval):
    labels = df.index.get_level_values('label').values
    predictions = df[ensemble].mean(axis=1)
    return {'fold': fold, 'seed': seedval, 'score': common.fmeasure_score(labels, predictions)['F'], 'ensemble': ensemble[-1],
            'ensemble_size': len(ensemble)}


def get_predictions(df, ensemble, fold, seedval):
    ids = df.index.get_level_values('id')
    labels = df.index.get_level_values('label')
    predictions = df[ensemble].mean(axis=1)
    diversity = common.diversity_score(df[ensemble].values)
    return pd.DataFrame(
        {'fold': fold, 'seed': seedval, 'id': ids, 'label': labels, 'prediction': predictions, 'diversity': diversity,
         'ensemble_size': len(ensemble)})


def select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i):
    initial_ensemble_size = 2
    max_candidates = 50
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace=False)
        candidate_scores = [common.score(train_labels, train_df[ensemble + [candidate]].mean(axis=1)) for candidate in
                            candidates]
        best_candidate = candidates[common.argbest(candidate_scores)]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def selection(fold, seedval, path, agg):
    seed(seedval)
    initial_ensemble_size = 2
    max_ensemble_size = 50
    max_candidates = 50
    max_diversity_candidates = 5
    accuracy_weight = 0.5
    max_clusters = 20
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    train_df = common.unbag(train_df, agg)
    test_df = common.unbag(test_df, agg)
    best_classifiers = train_df.apply(lambda x: common.fmeasure_score(train_labels, x)['F']).sort_values(
        ascending=not common.greater_is_better)
    train_performance = []
    test_performance = []
    ensemble = []
    for i in range(min(max_ensemble_size, len(best_classifiers))):
        best_candidate = select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i)
        ensemble.append(best_candidate)
        train_performance.append(get_performance(train_df, ensemble, fold, seedval))
        test_performance.append(get_performance(test_df, ensemble, fold, seedval))
    train_performance_df = pd.DataFrame.from_records(train_performance)
    best_ensemble_size = common.get_best_performer(train_performance_df).ensemble_size.values
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
            if testing_bool or (not 'foldAttribute' in p):
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
    auc = '%.3f' % (sklearn.metrics.roc_auc_score(predictions_df.label, predictions_df.prediction))

    # if ('67890' in fold_count and 'foldAttribute' in p):
    #     train_predictions_df = pd.concat(train_predictions_dfs)
    #
    #
    # else:
    print(thres)
    fmax = (common.fmeasure_score(predictions_df.label, predictions_df.prediction, thres=thres))
    return {'f-measure':fmax, 'auc':float(auc)}


def mean_fmax(path, fold_count=range(5), agg=1):
    def _unbag_mean(df, agg=agg):
        df = common.unbag(df, agg)
        return df.mean(axis=1).values
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    predictions = []
    labels = []
    # for fold in range(fold_count):
    for fold in fold_count:

        if testing_bool or (not 'foldAttribute' in p):
            train_df, train_label, test_df, test_label = common.read_fold(path, fold)
            predict = _unbag_mean(test_df, agg)
            predictions = append(predictions, predict)
            labels = append(labels, test_label)
            thres = thres_fmax(train_label, _unbag_mean(train_df))

    fmax = common.fmeasure_score(labels, predictions, thres)
    auc = '%.3f' % (sklearn.metrics.roc_auc_score(labels, predictions))

    return {'f-measure':fmax, 'auc':float(auc)}


def bestbase_fmax(path, fold_count=range(5), agg=1):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    predictions = []
    labels = []

    # for fold in range(fold_count):
    for fold in fold_count:
        # if '67890' in fold:
        if testing_bool or (not 'foldAttribute' in p):
            train_df, train_label, test_df, label = common.read_fold(path, fold)
            test_df = common.unbag(test_df, agg)
            predictions.append(test_df)
            labels = append(labels, label)
            # thres = thres_fmax(train_label, common.unbag(train_df, agg))
    predictions = pd.concat(predictions)

    # need to be changed
    fmax_list = [common.fmeasure_score(labels, predictions.iloc[:, i])['F'] for i in range(len(predictions.columns))]
    auc_list = [sklearn.metrics.roc_auc_score(labels, predictions.iloc[:, i]) for i in range(len(predictions.columns))]

    return {'f-measure':max(fmax_list), 'auc':max(auc_list)}
    # return max(fmax_list), max(auc_list)


def stacked_generalization(path, stacker_name, stacker, fold, agg):
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    train_df_cols = train_df.columns
    f_train_base = [common.fmeasure_score(train_labels, train_df[c].values) for c in train_df_cols]
    thres_train_base = [f['thres'] for f in f_train_base]
    # fscore_train_base = [f['F'] for f in f_train_base]
    # fscore_test_base = [common.fmeasure_score(test_labels, test_df[c].values, thres_train_base[idx]) for idx, c in enumerate(train_df_cols)]

    train_df = train_df - np.array(thres_train_base)
    test_df = test_df - np.array(thres_train_base)
    stacker = stacker.fit(train_df, train_labels)

    try:
        test_predictions = stacker.predict_proba(test_df)[:, 1]
        train_predictions = stacker.predict_proba(train_df)[:, 1]
    except:
        test_predictions = stacker.predict(test_df)[:, 1]
        train_predictions = stacker.predict(train_df)[:, 1]

    df = pd.DataFrame(
        {'fold': fold, 'id': test_df.index.get_level_values('id'), 'label': test_labels, 'prediction': test_predictions,
         'diversity': common.diversity_score(test_df.values)})
    return {'testing_df':df, "training": [train_labels, train_predictions]}


def main(path, fold_count=5, agg=1):

    dn = abspath(path).split('/')[-1]
    # cols = ['data_name', 'fmax', 'method']
    cols = ['data_name', 'fmax', 'method', 'auc']

    dfs = []
    print('[CES] Start building model #################################')
    ces = CES_fmax(path, fold_values, agg)
    print('[CES] Finished evaluating model ############################')
    print('[CES] F-max score is %s.' % ces['f-measure']['F'])
    print('[CES] AUC score is %s.' % ces['auc'])
    print('[Mean] Start building model ################################')
    mean = mean_fmax(path, fold_values, agg)
    print('[Mean] Finished evaluating model ###########################')
    print('[Mean] F-max score is %s.' % mean['f-measure']['F'])
    print('[Mean] AUC score is %s.' % mean['auc'])
    print('[Best Base] Start building model ###########################')
    bestbase = bestbase_fmax(path, fold_values, agg)
    print('[Best Base] Finished evaluating model ######################')
    print('[Best Base] F-max score is %s.' % bestbase['f-measure'])
    print('[Best Base] AUC score is %s.' % bestbase['auc'])
    dfs.append(pd.DataFrame(data=[[dn, ces['f-measure']['F'], 'CES', ces['auc']]], columns=cols, index=[0]))
    dfs.append(pd.DataFrame(data=[[dn, mean['f-measure']['F'], 'Mean', mean['auc']]], columns=cols, index=[0]))
    dfs.append(pd.DataFrame(data=[[dn, bestbase['f-measure'], 'best base', bestbase['auc']]], columns=cols, index=[0]))
    # Get Stacking Fmax scores
    stackers = [RandomForestClassifier(n_estimators=200, max_depth=2, bootstrap=False, random_state=0),
                SVC(C=1.0, cache_size=10000, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', probability=True,
                    max_iter=1e8, random_state=None, shrinking=True,
                    tol=0.001, verbose=False), GaussianNB(), LogisticRegression(), AdaBoostClassifier(),
                DecisionTreeClassifier(), GradientBoostingClassifier(loss='deviance'), KNeighborsClassifier()]
    stacker_names = ["RF.S", "SVM.S", "NB.S", "LR.S", "AB.S", "DT.S", "LB.S", "KNN.S"]
    for i, (stacker_name, stacker) in enumerate(zip(stacker_names, stackers)):
        print('[%s] Start building model ################################' % (stacker_name))
        if (not 'foldAttribute' in p):
            stacking_output = [stacked_generalization(path, stacker_name, stacker, fold, agg) for fold in fold_values]
        else:
            stacking_output = [stacked_generalization(path, stacker_name, stacker, '67890', agg)]
        predictions_dfs = [s['testing_df'] for s in stacking_output]
        _training = stacking_output[0]['training']
        thres = thres_fmax(_training[0], _training[1])

        predictions_df = pd.concat(predictions_dfs)
        print(thres)
        fmax = common.fmeasure_score(predictions_df.label, predictions_df.prediction, thres)
        print(fmax)
        auc = sklearn.metrics.roc_auc_score(predictions_df.label, predictions_df.prediction)
        print('[%s] Finished evaluating model ###########################' % (stacker_name))
        print('[%s] F-measure score is %s.' % (stacker_name, fmax['F']))
        print('[%s] Precision score is %s.' % (stacker_name, fmax['P']))
        print('[%s] Recall score is %s.' % (stacker_name, fmax['R']))
        print('[%s] AUC score is %s.' % (stacker_name, auc))
        df = pd.DataFrame(data=[[dn, fmax, stacker_name, auc]], columns=cols, index=[0])
        dfs.append(df)
    dfs = pd.concat(dfs)
    # Save results
    print('Saving results #############################################')
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    dfs.to_csv("%s/analysis/performance.csv" % path, index=False)


### parse arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--fold', '-F', type=int, default=5, help='cross-validation fold')
parser.add_argument('--aggregate', '-A', type=int, default=1, help='if aggregate is needed, feed bagcount, else 1')
args = parser.parse_args()
data_path = abspath(args.path)
fns = listdir(data_path)
fns = [fn for fn in fns if fn != 'analysis']
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
main(args.path, args.fold, args.aggregate)
