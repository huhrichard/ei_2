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

def best_stacking_score(df, stacking_suffix='.S'):

    # return_df = pd.DataFrame([])
    # col_wo_method = df.columns.values.tolist()
    # col_wo_method.remove('method')
    # col_wo_method.remove('')
    # print(col_wo_method, df.columns)

    pivoted_df = df.pivot_table('fmax', ['data_name'], 'method')
    # pivoted_df = pivoted_df.reindex(['data_name']+df['method'].unique())
    # print(pivoted_df.columns)
    # print(pivoted_df)
    cols = pivoted_df.columns.values
    stacking_cols = []
    for col in cols:
        if stacking_suffix in col:
            stacking_cols.append(col)

    # pivoted_df['best_stacking_fmax'] = 0
    pivoted_df['best_stacking_fmax'] = (pivoted_df[stacking_cols]).max(axis=1).values
    # pivoted_df.loc['best_stacking_method'] = ''
    # print(pivoted_df[stacking_cols])
    pivoted_df['best_stacking_method'] = (pivoted_df[stacking_cols]).idxmax(axis=1).values
    return pivoted_df

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
                   'deepNF': 'DeepNF',
                   'mashup': 'Mashup',
                   'coexpression': 'Coexpression',
                   'cooccurence': 'Coocuurence',
                   # 'database': 'Database',
                   'database': 'Curated database',

                   # 'experimental': 'Experimental',
                   'experimental': 'PPI',
                   'fusion': 'Fusion',
                   'neighborhood': 'Neighborhood'}
    cp = sns.color_palette(n_colors=len(dict_suffix))

    godag = get_godag("go-basic.obo")
    objanno = get_objanno('goa_human.gaf', 'gaf', godag=godag)
    termcounts = TermCounts(godag, objanno.get_id2gos_nss())
    # performance_df_dict = dict()
    # fmax_df = pd.DataFrame()
    fmax_list = []
    mean_fmax_list = []
    data_list = []
    stacking_df_list = []
    is_go = False
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
        # print(performance_df['data_name'].values[0])
        if 'GO' in performance_df['data_name'].values[0]:
            is_go = True
            gosubdag = GoSubDag(go_terms_set, godag)
            performance_df['go_depth'] = 0
            performance_df['go_ic'] = 0
            for go_term in go_terms_set:
                try:
                    depth = gosubdag.go2obj[go_term].depth
                    ic = get_info_content(go_term, termcounts)
                    depth = gosubdag.go2obj[go_term].level
                except:
                    depth = 0
                    ic = 0
                performance_df.loc[performance_df['data_name']==go_term, 'go_depth'] = depth
                performance_df.loc[performance_df['data_name']==go_term, 'go_ic'] = ic
        # stacking_df = extract_df_by_method(performance_df, method='LR.S', drop_columns=['method'])
        stacking_df = best_stacking_score(performance_df)

        stacking_df['input'] = val

        # performance_df['delta_fmax_LR.S'] = performance_df['fmax_LR.S'] - performance_df['fmax_best base']
        # best_base_df = extract_df_by_method(performance_df, method='best base')
        # performance_df_dict[val] = performance_df
        print(val, stacking_df.shape)
        fmax_list.append(stacking_df['best_stacking_fmax'].values)
        mean_fmax_list.append(np.median(stacking_df['best_stacking_fmax'].values))
        data_list.append(val)
        stacking_df_list.append(stacking_df)

    sorted_fmax_list = [f for m, f in sorted(zip(mean_fmax_list, fmax_list), reverse=True)]
    sorted_dataname_list = [f for m, f in sorted(zip(mean_fmax_list, data_list), reverse=True)]
    sorted_cp = [f for m, f in sorted(zip(mean_fmax_list, cp), reverse=True)]

    img_str = 'hpo'
    if is_go:
        img_str = 'go'
    ylabel = r'$F_{max}$ of best stacking'
    print(sorted_dataname_list)
    stacking_df_cat = pd.concat(stacking_df_list)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1 = sns.boxplot(ax=ax1, y='best_stacking_fmax', x='input',
                      data=stacking_df_cat, palette=sorted_cp, order=sorted_dataname_list)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('')
    ax1.set_title(title_name)
    fig1.savefig('f_max_{}_comparison_{}.png'.format(img_str, sys.argv[-2]), bbox_inches="tight")

    if is_go:
        fig2_plot_only = ['Mashup', 'DeepNF', 'EI']
        # idx_sorted_dataname = [sorted_dataname_list.index(p) for p in fig2_plot_only]
        # cp_plot_only = [sorted_cp[idx] for idx in idx_sorted_dataname]
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2 = sns.boxplot(ax=ax2, y='best_stacking_fmax', x='go_depth',
                          data=stacking_df_cat[stacking_df_cat['input'].isin(fig2_plot_only)],
                          # palette=c,
                          hue='input', hue_order=fig2_plot_only,
                          order=sorted(set(stacking_df_cat['go_depth'].values)))
        ax2.get_legend().remove()
        ax2.legend(loc='upper right')
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel('Depth in GO Hierarchy')
        ax2.set_title(title_name)
        fig2.savefig('f_max_{}_by_depth_{}.png'.format(img_str, sys.argv[-2]), bbox_inches="tight")

        # fig2_plot_only = ['Mashup', 'DeepNF', 'EI']
        # idx_sorted_dataname = [sorted_dataname_list.index(p) for p in fig2_plot_only]
        # cp_plot_only = [sorted_cp[idx] for idx in idx_sorted_dataname]

        ic_of_terms = stacking_df_cat['go_ic'].values
        # _, bin_edges = np.histogram(ic_of_terms, bins=5)
        bin_edges = np.percentile(ic_of_terms, np.linspace(0, 100, 6))
        ic_group_list = []
        stacking_df_cat['ic_group'] = 'temp'
        for idx, edge in enumerate(bin_edges[:-1]):
            next_edge = bin_edges[(idx+1)]
            group_name = '{:.2f}-{:.2f}'.format(edge, next_edge)
            stacking_df_cat.loc[(stacking_df_cat['go_ic'] <= next_edge) & (stacking_df_cat['go_ic'] >= edge), 'ic_group'] = group_name
            ic_group_list.append(group_name)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3 = sns.boxplot(ax=ax3, y='best_stacking_fmax', x='ic_group',
                          data=stacking_df_cat[stacking_df_cat['input'].isin(fig2_plot_only)],
                          # palette=c,
                          hue='input', hue_order=fig2_plot_only,
                          order=ic_group_list)
        ax3.get_legend().remove()
        ax3.legend(loc='upper right')
        ax3.set_ylabel(ylabel)
        ax3.set_xlabel('Information Content')
        ax3.set_title(title_name)
        fig3.savefig('f_max_{}_by_ic_{}.png'.format(img_str, sys.argv[-2]), bbox_inches="tight")

    #
    # ax1.boxplot(sorted_fmax_list)


    # ax1.set_title(title_name)


    # for key, df in performance_df_dict.items():

















