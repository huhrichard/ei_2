import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from goatools.base import get_godag
from goatools.semantic import get_info_content
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.anno.factory import get_objanno
from goatools.semantic import TermCounts
from goatools.semantic import get_info_content

from goatools.base import download_go_basic_obo

obo_fname = download_go_basic_obo()
from os.path import abspath

import os, fnmatch
import sys

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def extract_df_by_method(df, method='', drop_columns=['method']):
    return_df = df[df['method']==method]
    return_df.rename(columns={'fmax':'fmax_{}'.format(method)}, inplace=True)
    return_df.drop(drop_columns, axis=1, inplace=True)
    return return_df

def add_colon(str):
    return str[:2]+':'+str[2:]

if __name__ == "__main__":

    # Load all performance csv
    performance_file_list = find('performance.csv', sys.argv[-1])
    performance_df_list = []
    for performance_file in performance_file_list:
        df = pd.read_csv(performance_file)
        performance_df_list.append(df)

    performance_df = pd.concat(performance_df_list)

    godag = get_godag("go-basic.obo")
    performance_df['data_name'] = performance_df['data_name'].apply(add_colon)
    go_terms_set = set(list(performance_df['data_name']))
    gosubdag = GoSubDag(go_terms_set, godag)
    performance_df['go_level'] = 0
    for go_term in go_terms_set:
        try:
            depth = gosubdag.go2obj[go_term].level
        except:
            depth = 0
        performance_df.loc[performance_df['data_name']==go_term, 'go_level'] = depth

    best_base_df = extract_df_by_method(performance_df, method='best base')
    lrs_df = extract_df_by_method(performance_df, method='LR.S', drop_columns=['method', 'go_level'])

    delta_fmax_df = pd.merge(best_base_df, lrs_df, on='data_name', how='inner')
    delta_fmax_df['delta_fmax_LR.S'] = delta_fmax_df['fmax_LR.S']-delta_fmax_df['fmax_best base']


    go_level = sorted(set(delta_fmax_df['go_level']))
    delta_fmax_by_level = [delta_fmax_df.loc[delta_fmax_df['go_level']==l, 'delta_fmax_LR.S'] for l in go_level]
    fmax_lrs_by_level = [delta_fmax_df.loc[delta_fmax_df['go_level']==l, 'fmax_LR.S'] for l in go_level]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    bp1 = ax1.boxplot(delta_fmax_by_level)
    ax1.set_xticklabels(go_level)
    ax1.set_xlabel('Level in GO hierarchy')
    ax1.set_ylabel(r'$\Delta F_{max} $')
    fig1.savefig('delta_fmax.png', bbox_inches="tight")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    bp2 = ax2.boxplot(fmax_lrs_by_level)
    ax2.set_xticklabels(go_level)
    ax2.set_xlabel('Level in GO hierarchy')
    ax2.set_ylabel(r'$ F_{max}$ of LR.Stacking')
    fig2.savefig('fmax_lrs.png', bbox_inches="tight")


    print(delta_fmax_df)









