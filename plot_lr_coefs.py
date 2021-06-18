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

deceased_outcome_since_prefix = 'DECEASED_AT_{}DAYS'
deceased_outcome_since_prefix_plot = 'DECEASED\nAT_{}DAYS'
outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
# outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
# deceased_days_timeframe = [3, 5, 7, 10]
deceased_days_timeframe = [10, 7, 5, 3]
for dday in deceased_days_timeframe:
    outcomes[deceased_outcome_since_prefix.format(dday)] = deceased_outcome_since_prefix_plot.format(dday)

for outcome_key, outcome_val in outcomes.items():
    csv_path = os.path.join(*[sys.argv[-1], outcome_key, 'analysis', 'coefs_lr.csv'])
    lr_coefs_df = pd.read_csv(csv_path)
    lr_coefs_cols = lr_coefs_df.columns.tolist()
    lr_coefs_cols.remove('fold')
    melted_lr_coefs_df = pd.melt(lr_coefs_df,
                                 id_vars=['fold'],
                                 value_vars=lr_coefs_cols)
    print(melted_lr_coefs_df)
    coefs_cat = melted_lr_coefs_df['value'].str.rsplit('.')
    coefs_cat.rename(columns={0:'Modality', 1:'Base Predictor'})

    melted_df = pd.concat(melted_lr_coefs_df, coefs_cat, axis=1, ignore_index=True)
    print(melted_df)
    fig2, ax2 = plt.subplots(1, 1, figsize=(11, 6))