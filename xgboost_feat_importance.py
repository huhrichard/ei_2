import xgboost
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
import numpy as np
from pandas.api.types import is_string_dtype
import common
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from fancyimpute import IterativeSVD

f_path = './not_on_github/covid19/msdw_covid19_27May2020/csv'
# f_path = '.'
# fn = 'vitals'
fn = sys.argv[-1]
concat_df = pd.read_csv('{}/{}.csv'.format(f_path, fn), index_col=0)
outcome_df = pd.read_csv('{}/{}'.format(f_path, 'outcome.csv'), index_col=0)
# concat_svdImputed = pd.pd.read_csv('{}/{}.csv'.format(f_path, fn), index_col=0)
concat_dtype = concat_df.dtypes

# print(concat_dtype)

for c, dtype in concat_dtype.items():
    # print(c, dtype)
    if is_string_dtype(dtype):
        concat_df = pd.concat([concat_df, pd.get_dummies(concat_df[c], prefix=c, dummy_na=True)],axis=1)
        # print(concat_df.columns)
        concat_df.drop(columns=[c], inplace=True)
        # print(concat_df.columns)

concat_df = concat_df.apply(pd.to_numeric)
concat_df=(concat_df-concat_df.mean())/concat_df.std()
concat_df[:] = IterativeSVD().fit_transform(concat_df.values)

data_outcome_df = pd.concat([concat_df, outcome_df], axis=1, join='inner')





kf = KFold(n_splits=5, shuffle=True, random_state=1)
outcome = 'DECEASED_INDICATOR'
y = data_outcome_df[outcome].values.astype(float)
X = data_outcome_df.drop(columns=[outcome]).values.astype(float)
X_cols = data_outcome_df.drop(columns=[outcome]).columns.to_list()
feat_imp_list = []
test_pred_list = []
test_label_list = []


print(X.shape)

data = ['Comorbidities', 'Demographics', 'Medications', 'Labs', 'Vitals']

def classifications(clf_name, clf):
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # clf = xgboost.XGBClassifier(missing=np.nan)
        clf.fit(X_train, y_train)
        feat_imp_list.append(clf.feature_importances_)
        test_pred_list.append(clf.predict_proba(X_test)[:,1])
        test_label_list.append(y_test)

    feat_imp = np.array(feat_imp_list)
    test_pred = np.concatenate(test_pred_list)
    test_label = np.concatenate(test_label_list)
    f_max = common.fmeasure_score(test_label, test_pred)
    auc = sklearn.metrics.roc_auc_score(test_label, test_pred)
    print(clf_name)
    print(f_max)
    print('AUC:', auc)


    feat_imp_df = pd.DataFrame(data=feat_imp, columns=X_cols)
    # feat_imp_df = feat_imp_df.stack()
    feat_imp_median = feat_imp_df.median(axis=0)
    # feat_imp_median
    feat_imp_median.sort_values(inplace=True, ascending=False)
    #

    feat_imp_df['idx'] = feat_imp_df.index
    print(feat_imp_median)
    feat_imp_df = pd.melt(feat_imp_df, id_vars=['idx'], value_vars=X_cols,
                          var_name='feature', value_name='feature_importance')
    feat_imp_df['data'] = ''
    feat_imp_median_by_data = []
    for datum in data:
        feat_imp_df.loc[feat_imp_df['feature'].str.contains(datum), 'data'] = datum
        feat_imp_median_of_datum = feat_imp_df.loc[feat_imp_df['data']==datum, 'feature_importance'].median()
        feat_imp_median_by_data.append((feat_imp_median_of_datum, datum))

    sorted_tuple = sorted(feat_imp_median_by_data, reverse=True, key=lambda x: x[0])
    sorted_data = [s[1] for s in sorted_tuple]


    fig, ax = plt.subplots(1,1, figsize=(6,24))
    ax = sns.boxplot(ax=ax, data=feat_imp_df, x='feature_importance',
                     y='feature', orient='h', order=feat_imp_median.index.to_list())
    ax.set_title('Feat. Imp. of {}'.format(clf_name))
    fig.savefig('plot/feat_imp_{}_{}.pdf'.format(clf_name, fn), bbox_inches="tight")

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    ax1 = sns.boxplot(ax=ax1, data=feat_imp_df, x='feature_importance',
                     y='data', orient='h', order=sorted_data)
    ax1.set_title('Feat. Imp. of {}'.format(clf_name))
    fig1.savefig('plot/feat_imp_{}_{}_by_data.pdf'.format(clf_name, fn), bbox_inches="tight")
    # feat_imp_by
    #
    # print(feat_imp.shape)
    # print(test_pred.shape)
    # print(test_label.shape)

clf_dict = {'RF': RandomForestClassifier(),
            'Adaboost': AdaBoostClassifier(),
            'xgboost': xgboost.XGBClassifier()
            }

for k, v in clf_dict.items():
    classifications(k, v)
