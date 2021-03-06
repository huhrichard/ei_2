import pandas as pd
import matplotlib
# import Orange
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
# from os.path import abspath
from os.path import abspath, isdir, exists
from os import remove, system, listdir

import os, fnmatch
import sys
from os import system

system('module load R')
plot_dir = './plot/'

def find(pattern, path):
    result = []
    print(path)
    for root, dirs, files in os.walk(path):
        print(root, dirs, files)
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def extract_df_by_method(df, method='', drop_columns=['method']):
    return_df = df[df['method']==method]
    return_df.rename(columns={'fmax':'fmax_{}'.format(method)}, inplace=True)
    return_df.drop(drop_columns, axis=1, inplace=True)
    return return_df

def best_ensemble_score(df, input, ensemble_suffix='.S'):
    list_best_base = ['deepNF', 'mashup']
    # return_df = pd.DataFrame([])
    col_wo_method = df.columns.tolist()
    # col_wo_method.remove('method')
    # col_wo_method.remove('fmax')
    # # col_wo_method.remove('')
    # print(col_wo_method, df.columns)

    pivoted_df = df.pivot_table('fmax', ['data_name'], 'method')
    # pivoted_df = df.pivot_table('fmax', col_wo_method, 'method')
    # pivoted_df = pivoted_df.reindex(['data_name']+df['method'].unique())
    # print(pivoted_df.columns)
    # print(pivoted_df)
    # cols = pivoted_df.columns.values
    # ensemble_cols = []
    # for col in cols:
    #     if ensemble_suffix in col:
    #         ensemble_cols.append(col)
    ensemble_cols = df['method'].unique().tolist()
    ensemble_cols.remove('best base')
    # ensemble_cols.remove('XGB.S')

    # pivoted_df['best_fmax'] = 0
    if input in list_best_base:
        pivoted_df['best_fmax'] = pivoted_df['best base'].values
    else:
        pivoted_df['best_fmax'] = (pivoted_df[ensemble_cols]).max(axis=1).values
    # pivoted_df.loc['best_ensemble_method'] = ''
    # print(pivoted_df[ensemble_cols])
    pivoted_df['best_ensemble_method'] = (pivoted_df[ensemble_cols]).idxmax(axis=1).values
    return pivoted_df.reset_index()

def add_colon(str):
    return str[:2]+':'+str[2:]

if __name__ == "__main__":

    # Load all performance csv
    group = sys.argv[-2]
    if '-' not in group:
        group = '>' + group
    title_name = "#annotated proteins: {}".format(group)
    file_prefix = sys.argv[-3]
    dict_suffix = {'EI': 'Ensemble\nIntegration',
                   'deepNF': 'DeepNF',
                   'mashup': 'Mashup',
                   '/coexpression': 'Coexpression',
                   '/cooccurence': 'Co-occurrence',
                   # 'database': 'Database',
                   '/database': 'Curated\nDatabases',

                   # 'experimental': 'Experimental',
                   '/experimental': 'PPI',
                   '/fusion': 'Fusion',
                   '/neighborhood': 'Neighborhood'}
    # rwr_dict_suffix = {'rwrImpute_'+k : v+('\n(RWR Impute)') for k, v in dict_suffix.items() if k != ''}
    # rwr_dict_suffix['rwrImpute'] = 'Ensemble\nIntegration\n(RWR Impute)'
    if 'Impute' in file_prefix:
        dict_suffix = {k : v+('\n(RWR Impute)') for k, v in dict_suffix.items()}
    # rwr_dict_suffix['rwrImpute'] = 'Ensemble\nIntegration\n(RWR Impute)'
    cp = sns.color_palette(n_colors=len(dict_suffix))

    godag = get_godag("go-basic.obo")
    objanno = get_objanno('goa_human.gaf', 'gaf', godag=godag)
    termcounts = TermCounts(godag, objanno.get_id2gos_nss())
    # performance_df_dict = dict()
    # fmax_df = pd.DataFrame()
    fmax_list = []
    median_fmax_list = []
    data_list = []
    ensemble_df_list = []
    is_go = 'go' in sys.argv[-1]

    # ensemble_df
    for key, val in dict_suffix.items():
        # if len(key) > 0:
        #     go_dir = sys.argv[-1] + '_' + key
        # else:
        #     go_dir = sys.argv[-1]
        if not '/' in key:
            go_dir = sys.argv[-1] + '_' + key
            sub_data_folder = ''
        else:
            go_dir = sys.argv[-1] + '_EI'
            sub_data_folder = key+'/'

        fns = listdir(go_dir)
        # fns = [fn for fn in fns if fn != 'analysis']
        fns = [go_dir + '/' + fn for fn in fns]
        term_dirs = [fn for fn in fns if isdir(fn)]
        # if len(feature_folders) == 0:
        #     feature_folders.append('./')
        # assert len(feature_folders) > 0
        performance_file_list = {}
        for term_dir in term_dirs:

            file_name = term_dir + '/' +sub_data_folder + 'analysis/' + 'performance.csv'
            # print(file_name)
            term_name = term_dir.split('/')[-1]

            if exists(file_name):
                performance_file_list[term_name] = file_name
            # if not '/' in key:
            #     # performance_file_list += find('performance.csv', term_dir + 'analysis/')
            #     # temp = find('performance.csv', term_dir + 'analysis/')
            #     # print(temp)
            #     performance_file_list += find('performance.csv', term_dir + 'analysis/')
            # else:
            #     performance_file_list += find('performance.csv', term_dir + key + '/')
        # print(key, term_dirs)
        # print(performance_file_list)
        # dir = sys.argv[-1].split('/')[-2]
        performance_df_list = []
        for term_name, performance_file in performance_file_list.items():
            df = pd.read_csv(performance_file)
            df['data_name'] = term_name
            # print(df)
            performance_df_list.append(df)

        performance_df = pd.concat(performance_df_list)
        # print(performance_df.columns)
        performance_df['data_name'] = performance_df['data_name'].apply(add_colon)
        go_terms_set = set(list(performance_df['data_name']))
        # print(performance_df['data_name'].values[0])

        # ensemble_df = extract_df_by_method(performance_df, method='LR.S', drop_columns=['method'])
        ensemble_df = best_ensemble_score(performance_df, input=key)
        if 'GO' in ensemble_df['data_name'].values[0]:
            is_go = True
            gosubdag = GoSubDag(go_terms_set, godag)
            ensemble_df['go_depth'] = 0
            ensemble_df['go_ic'] = 0
            for go_term in go_terms_set:
                try:
                    depth = gosubdag.go2obj[go_term].depth
                    ic = get_info_content(go_term, termcounts)
                except:
                    depth = 0
                    ic = 0
                ensemble_df.loc[ensemble_df['data_name']==go_term, 'go_depth'] = depth
                ensemble_df.loc[ensemble_df['data_name']==go_term, 'go_ic'] = ic

        ensemble_df['input'] = val

        # performance_df['delta_fmax_LR.S'] = performance_df['fmax_LR.S'] - performance_df['fmax_best base']
        # best_base_df = extract_df_by_method(performance_df, method='best base')
        # performance_df_dict[val] = performance_df
        # print(val, group, ensemble_df.shape)
        # fmax_list.append(ensemble_df['best_fmax'].values)
        # median_fmax_list.append(np.nanmedian(ensemble_df['best_fmax'].values))
        # data_list.append(val)
        ensemble_df_list.append(ensemble_df)

    # print(median_fmax_list)
    # print(len(fmax_list), len(median_fmax_list))

    # sorted_fmax_list = [f for m, f in sorted(zip(median_fmax_list, fmax_list), reverse=True, key=lambda x: x[0])]
    # sorted_dataname_list = [f for m, f in sorted(zip(median_fmax_list, data_list), reverse=True, key=lambda x: x[0])]
    # sorted_cp = [f for m, f in sorted(zip(median_fmax_list, cp), reverse=True, key=lambda x: x[0])]

    img_str = 'hpo'
    if is_go:
        img_str = 'go'
    ylabel = r'$F_{max}$'
    # print(sorted_dataname_list)
    # print(sorted_fmax_list)


    ensemble_df_cat = pd.concat(ensemble_df_list)
    # print('shape before drop', ensemble_df_cat.shape)
    # ensemble_df_cat.dropna(inplace=True)
    # print('shape after drop', ensemble_df_cat.shape)

    cd_input = ensemble_df_cat[['data_name', 'best_fmax', 'input']]


    cd_input_df = cd_input.pivot_table('best_fmax', ['data_name'], 'input').reset_index()
    cd_input_df.set_index('data_name', inplace=True)

    cd_csv_fn = '{}cd_input_{}_{}.csv'.format(plot_dir + 'cd_csv/', file_prefix, sys.argv[-2])
    cd_input_df.to_csv(cd_csv_fn, index_label=False)
    cmd = "R CMD BATCH --no-save --no-restore '--args cd_fn=\"{}\"' R/plotCDdiagram.R".format(cd_csv_fn)
    os.system(cmd)

    cd_input_df.dropna(inplace=True)
    median_fmax_list = np.median(cd_input_df.values, axis=0)
    fmax_list = cd_input_df.values
    # data_list = [k for v, k in dict_suffix.items()]
    data_list = cd_input_df.columns.tolist()
    # dict_value_list = [k for v, k in dict_suffix.items()]
    index_data_list = [data_list.index(k) for v, k in dict_suffix.items()]
    cp_new = [cp[idx] for idx in index_data_list]

    sorted_list = sorted(zip(median_fmax_list, fmax_list, data_list, cp_new), reverse=True, key=lambda x: x[0])
    sorted_dataname_list = [s[2] for s in sorted_list]
    print(sorted_dataname_list)
    sorted_cp = [s[3] for s in sorted_list]

    # make input for cd plot



    # print(cd_input_df)
    # average_rank_df = cd_input_df.rank(axis=1)
    # print(average_rank_df)
    # for row in cd_input_df.index:
    #     average_rank_df.loc[row] = df.loc[row].rank
    # print(ensemble_df_cat, ensemble_df_cat.columns)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1 = sns.boxplot(ax=ax1, y='best_fmax', x='input',
                      data=ensemble_df_cat, palette=sorted_cp, order=sorted_dataname_list)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('')
    ax1.set_title(title_name)
    fig1.savefig('{}f_max_{}_comparison_{}.pdf'.format(plot_dir, file_prefix, sys.argv[-2]), bbox_inches="tight")

    if is_go:

        # fig2_plot_only = ['Mashup', 'DeepNF', 'EI']
        fig2_plot_only = ['Mashup', 'DeepNF', 'Ensemble\nIntegration']
        # idx_sorted_dataname = [sorted_dataname_list.index(p) for p in fig2_plot_only]
        # cp_plot_only = [sorted_cp[idx] for idx in idx_sorted_dataname]
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2 = sns.boxplot(ax=ax2, y='best_fmax', x='go_depth',
                          data=ensemble_df_cat[ensemble_df_cat['input'].isin(fig2_plot_only)],
                          # palette=c,
                          hue='input', hue_order=fig2_plot_only,
                          order=sorted(set(ensemble_df_cat['go_depth'].values)))
        ax2.get_legend().remove()
        ax2.legend(loc='upper right')
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel('Depth in GO Hierarchy')
        ax2.set_title(title_name)
        fig2.savefig('{}f_max_{}_by_depth_{}.pdf'.format(plot_dir, img_str, sys.argv[-2]), bbox_inches="tight")

        # fig2_plot_only = ['Mashup', 'DeepNF', 'EI']
        # idx_sorted_dataname = [sorted_dataname_list.index(p) for p in fig2_plot_only]
        # cp_plot_only = [sorted_cp[idx] for idx in idx_sorted_dataname]

        ic_of_terms = ensemble_df_cat['go_ic'].values
        # _, bin_edges = np.histogram(ic_of_terms, bins=5)
        bin_edges = np.percentile(ic_of_terms, np.linspace(0, 100, 6))
        ic_group_list = []
        ensemble_df_cat['ic_group'] = 'temp'
        for idx, edge in enumerate(bin_edges[:-1]):
            next_edge = bin_edges[(idx+1)]
            group_name = '{:.2f}-{:.2f}'.format(edge, next_edge)
            ensemble_df_cat.loc[(ensemble_df_cat['go_ic'] <= next_edge) & (ensemble_df_cat['go_ic'] >= edge), 'ic_group'] = group_name
            ic_group_list.append(group_name)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3 = sns.boxplot(ax=ax3, y='best_fmax', x='ic_group',
                          data=ensemble_df_cat[ensemble_df_cat['input'].isin(fig2_plot_only)],
                          # palette=c,
                          hue='input', hue_order=fig2_plot_only,
                          order=ic_group_list)
        ax3.get_legend().remove()
        ax3.legend(loc='upper right')
        ax3.set_ylabel(ylabel)
        ax3.set_xlabel('Information Content')
        ax3.set_title(title_name)
        fig3.savefig('{}f_max_{}_by_ic_{}.pdf'.format(plot_dir, img_str, sys.argv[-2]), bbox_inches="tight")

    #
    # ax1.boxplot(sorted_fmax_list)


    # ax1.set_title(title_name)


    # for key, df in performance_df_dict.items():

















