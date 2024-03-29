import pandas as pd
import matplotlib

# import Orange
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

from os.path import abspath, isdir, exists
from os import remove, system, listdir

import os, fnmatch
import sys
from os import system

def find(pattern, path):
    result = []
    print(path)
    for root, dirs, files in os.walk(path):
        print(root, dirs, files)
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

dict_suffix = ['demographics',
               'labs',
               'comorbidities',
               'vitals']

plot_dir = './plot/'

deceased_outcome_since_prefix = 'DECEASED_AT_{}DAYS'
deceased_outcome_since_prefix_plot = 'DECEASED\nAT_{}DAYS'
outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
# outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
# deceased_days_timeframe = [3, 5, 7, 10]
deceased_days_timeframe = [10, 7, 5, 3]
for dday in deceased_days_timeframe:
    outcomes[deceased_outcome_since_prefix.format(dday)] = deceased_outcome_since_prefix_plot.format(dday)

cp = sns.color_palette(n_colors=len(dict_suffix))
for outcome_key, outcome_val in outcomes.items():
    # csv_path = os.path.join(*[sys.argv[-1], outcome_key, 'analysis', 'coefs_lr_pi.csv'])
    csv_path = os.path.join(*[sys.argv[-1], outcome_key, 'analysis', 'coefs_lr.csv'])
    lr_coefs_df = pd.read_csv(csv_path, index_col=0)
    lr_coefs_cols = lr_coefs_df.columns.tolist()
    print(lr_coefs_cols)
    print(lr_coefs_df)
    # lr_coefs_cols['fold'] = 0
    # lr_coefs_cols.remove('fold')
    melted_lr_coefs_df = pd.melt(lr_coefs_df,
                                 # id_vars=[],
                                 value_vars=lr_coefs_cols)
    print(melted_lr_coefs_df)
    coefs_cat = melted_lr_coefs_df['variable'].str.split('.', expand=True)
    coefs_cat.columns = ['Modality', 'Base Predictor', 'Bag']
    melted_df = pd.concat([melted_lr_coefs_df, coefs_cat], axis=1)
    # print(melted_df)
    melted_df['Absolute Coefficient'] = abs(melted_df['value'])
    melted_df.rename(columns={'value': 'Coefficient'},
                     inplace=True)
    plots = ['Absolute Coefficient',
             # 'Coefficient'
             ]
    for p in plots:
        median_coef_dict_by_modal = [np.median(melted_df.loc[coefs_cat['Modality'] == k,
                                                             p]) for k in dict_suffix]
        print(median_coef_dict_by_modal)
        sorted_list = sorted(zip(median_coef_dict_by_modal, dict_suffix, cp), reverse=True, key=lambda x: x[0])
        modal_list = [m[1] for m in sorted_list]
        sorted_cp = [m[2] for m in sorted_list]

        fig1, ax1 = plt.subplots(1, 1, figsize=(11, 6))
        ax1 = sns.boxplot(ax=ax1, y=p, x='Modality',
                          data=melted_df, order=modal_list, palette=sorted_cp,
                          linewidth=2, width=0.5)

        ax1.set_title(outcome_key)
        # fig1.savefig('{}{}{}_LR_{}_pi.pdf'.format(plot_dir, 'covid19/', outcome_key, p), bbox_inches="tight")
        fig1.savefig('{}{}{}_LR_{}.pdf'.format(plot_dir, 'covid19/', outcome_key, p), bbox_inches="tight")


for outcome_key, outcome_val in outcomes.items():
    csv_path = os.path.join(*[sys.argv[-1], outcome_key, 'analysis', 'coefs_lr_pi.csv'])
    lr_coefs_df = pd.read_csv(csv_path, index_col=0)
    lr_coefs_cols = lr_coefs_df.columns.tolist()
    print(lr_coefs_cols)
    print(lr_coefs_df)
    # lr_coefs_cols['fold'] = 0
    # lr_coefs_cols.remove('fold')
    melted_lr_coefs_df = pd.melt(lr_coefs_df,
                                 # id_vars=[],
                                 value_vars=lr_coefs_cols)
    print(melted_lr_coefs_df)
    coefs_cat = melted_lr_coefs_df['variable'].str.split('.', expand=True)
    coefs_cat.columns = ['Modality', 'Base Predictor', 'Bag']
    melted_df = pd.concat([melted_lr_coefs_df, coefs_cat], axis=1)
    # print(melted_df)
    # melted_df['Absolute Coefficient'] = abs(melted_df['value'])
    melted_df.rename(columns={'value': 'Permutation Importances'},
                     inplace=True)
    plots = ['Permutation Importances']
    for p in plots:
        median_coef_dict_by_modal = [np.median(melted_df.loc[coefs_cat['Modality'] == k,
                                                             p]) for k in dict_suffix]
        print(median_coef_dict_by_modal)
        sorted_list = sorted(zip(median_coef_dict_by_modal, dict_suffix, cp), reverse=True, key=lambda x: x[0])
        modal_list = [m[1] for m in sorted_list]
        sorted_cp = [m[2] for m in sorted_list]

        fig1, ax1 = plt.subplots(1, 1, figsize=(11, 6))
        ax1 = sns.boxplot(ax=ax1, y=p, x='Modality',
                          data=melted_df, order=modal_list, palette=sorted_cp,
                          linewidth=2, width=0.5)

        ax1.set_title(outcome_key)
        fig1.savefig('{}{}{}_LR_{}.pdf'.format(plot_dir, 'covid19/', outcome_key, p), bbox_inches="tight")
