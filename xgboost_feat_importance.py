import xgboost
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from pandas.api.types import is_string_dtype
import common
import seaborn as sns
import matplotlib.pyplot as plt
# from fancyimpute import IterativeSVD

# f_path = 'msdw_covid19_27May2020'
f_path = '.'
concat_df = pd.read_csv('{}/{}'.format(f_path, 'concatenated.csv'), index_col=0)
outcome_df = pd.read_csv('{}/{}'.format(f_path, 'outcome.csv'), index_col=0)

concat_dtype = concat_df.dtypes

# print(concat_dtype)

for c, dtype in concat_dtype.items():
    # print(c, dtype)
    if is_string_dtype(dtype):
        concat_df = pd.concat([concat_df, pd.get_dummies(concat_df[c], prefix=c, dummy_na=True)],axis=1)
        # print(concat_df.columns)
        concat_df.drop(columns=[c], inplace=True)
        # print(concat_df.columns)

# concat_df = concat_df.apply(pd.to_numeric)
# concat_df=(concat_df-concat_df.mean())/concat_df.std()
# concat_df[:] = IterativeSVD().fit_transform(concat_df.values)

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
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = xgboost.XGBClassifier(missing=np.nan)
    clf.fit(X_train, y_train)
    feat_imp_list.append(clf.feature_importances_)
    test_pred_list.append(clf.predict_proba(X_test)[:,1])
    test_label_list.append(y_test)

feat_imp = np.array(feat_imp_list)
test_pred = np.concatenate(test_pred_list)
test_label = np.concatenate(test_label_list)
f_max = common.fmeasure_score(test_label, test_pred)
print(f_max)

feat_imp_df = pd.DataFrame(data=feat_imp, columns=X_cols)
# feat_imp_df = feat_imp_df.stack()
feat_imp_median = feat_imp_df.median(axis=1)
feat_imp_median.sort_values(by=0, axis=1, inplace=True)


feat_imp_df['idx'] = feat_imp_df.index
print(feat_imp_df)
feat_imp_df = pd.melt(feat_imp_df, id_vars=['idx'], value_vars=X_cols,
                      var_name='feature', value_name='feature_importance')

fig, ax = plt.subplots(1,1, figsize=(6,18))
ax = sns.boxplot(ax=ax, data=feat_imp_df, x='feature_importance',
                 y='feature', orient='h', order=feat_imp_median.columns.to_list())
fig.savefig('plot/feat_imp_xgboost.pdf', bbox_inches="tight")

print(feat_imp.shape)
print(test_pred.shape)
print(test_label.shape)
