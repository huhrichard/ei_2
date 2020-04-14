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
import seaborn as sns

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
    group = sys.argv[-2]
    if '-' not in group:
        group = '>' + group
    title_name = "#annotated proteins: {}".format(group)
    file_prefix = sys.argv[-1]
    dict_suffix = {'': 'EI',
                   'coexpression': 'Coexpression',
                   'cooccurence': 'Coocuurence',
                   # 'database': 'Database',
                   'database': 'Curated database',
                   'deepNF': 'DeepNF',
                   # 'experimental': 'Experimental',
                   'experimental': 'PPI',
                   'mashup': 'Mashup',
                   'fusion': 'Fusion',
                   'neighborhood': 'Neighborhood'}

    godag = get_godag("go-basic.obo")
    # performance_df_dict = dict()
    # fmax_df = pd.DataFrame()
    fmax_list = []
    mean_fmax_list = []
    data_list = []
    for key, val in dict_suffix.items():
        if len(key) > 0:
            go_dir = sys.argv[-1] + '_' + key
        else:
            go_dir = sys.argv[-1]
        performance_file_list = find('performance.csv', go_dir)

        # dir = sys.argv[-1].split('/')[-2]
        performance_df_list = []
        for performance_file in performance_file_list:
            df = pd.read_csv(performance_file)
            performance_df_list.append(df)

        performance_df = pd.concat(performance_df_list)
        # print(performance_df.columns)
        performance_df['data_name'] = performance_df['data_name'].apply(add_colon)
        go_terms_set = set(list(performance_df['data_name']))
        gosubdag = GoSubDag(go_terms_set, godag)
        performance_df['go_depth'] = 0
        for go_term in go_terms_set:
            try:
                depth = gosubdag.go2obj[go_term].depth
                # depth = gosubdag.go2obj[go_term].level
            except:
                depth = 0
            performance_df.loc[performance_df['data_name']==go_term, 'go_depth'] = depth
        lrs_df = extract_df_by_method(performance_df, method='LR.S', drop_columns=['method', 'go_depth'])

        # performance_df['delta_fmax_LR.S'] = performance_df['fmax_LR.S'] - performance_df['fmax_best base']
        # best_base_df = extract_df_by_method(performance_df, method='best base')
        # performance_df_dict[val] = performance_df
        print(val, lrs_df.shape)
        fmax_list.append(lrs_df['fmax_LR.S'].values)
        mean_fmax_list.append(np.mean(lrs_df['fmax_LR.S'].values))
        data_list.append(val)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    sorted_fmax_list = [f for m, f in sorted(zip(mean_fmax_list, fmax_list), reverse=True)]
    sorted_dataname_list = [f for m, f in sorted(zip(mean_fmax_list, data_list), reverse=True)]
    print(sorted_dataname_list)

    ax1.boxplot(sorted_fmax_list)
    ax1.set_ylabel(r'$F_{max}$')
    ax1.set_xticklabels(sorted_dataname_list)
    ax1.set_title(title_name)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
    fig1.savefig('f_max_comparison_{}.png'.format(sys.argv[-2]))

    # for key, df in performance_df_dict.items():

















