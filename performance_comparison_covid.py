import pandas as pd
import matplotlib

# import Orange
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# from goatools.base import get_godag
# from goatools.semantic import get_info_content
# from goatools.gosubdag.gosubdag import GoSubDag
# from goatools.anno.factory import get_objanno
# from goatools.semantic import TermCounts
#
# from goatools.base import download_go_basic_obo
import seaborn as sns

# obo_fname = download_go_basic_obo()
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
    return_df = df[df['method'] == method]
    return_df.rename(columns={'fmax': 'fmax_{}'.format(method)}, inplace=True)
    return_df.drop(drop_columns, axis=1, inplace=True)
    return return_df


def best_ensemble_score(df, input, ensemble_suffix='.S', metric='fmax'):
    list_best_base = ['deepNF', 'mashup']
    # return_df = pd.DataFrame([])
    col_wo_method = df.columns.tolist()
    # col_wo_method.remove('method')
    # col_wo_method.remove('fmax')
    # # col_wo_method.remove('')
    # print(col_wo_method, df.columns)

    pivoted_df = df.pivot_table(metric, ['data_name'], 'method')
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
    if 'best base' in ensemble_cols:
        ensemble_cols.remove('best base')
    # ensemble_cols.remove('XGB.S')
    best_metric_str = 'best_' + metric
    # pivoted_df['best_fmax'] = 0
    if input in list_best_base:
        pivoted_df[best_metric_str] = pivoted_df['best base'].values
    else:
        pivoted_df[best_metric_str] = (pivoted_df[ensemble_cols]).max(axis=1).values
    # pivoted_df.loc['best_ensemble_method'] = ''
    # print(pivoted_df[ensemble_cols])
    pivoted_df['best_ensemble_method'] = (pivoted_df[ensemble_cols]).idxmax(axis=1).values
    return pivoted_df.reset_index()


def add_colon(str):
    return str[:2] + ':' + str[2:]


if __name__ == "__main__":

    # Load all performance csv
    metrics = {'fmax': r'$F_{max}$',
               'auc': 'AUC'}
    # group = sys.argv[-2]
    # if '-' not in group:
    #     group = '>' + group
    # title_name = "#annotated proteins: {}".format(group)
    title_name = "COVID-19 death predictors"
    file_prefix = 'covid-19'

    dict_suffix = {'EI': 'Ensemble\nIntegration',
                   # 'EI_PowerSet':'Ensemble Integration\nPower Set',
                   'demographics': 'Demo-\ngraphics\n(11)',
                   # 'labs':'Laboratory\ntests',
                   'labs': 'Lab\ntests\n(49)',
                   # 'medications': 'Medica-\ntions\n(26)',
                   'comorbidities': 'Co-morbi-\ndities\n(19)',
                   'vitals': 'Vital\nsigns\n(6)',
                   'concatenated': 'Concat-\nenated\nAll',
                   'xgboost': 'XGBoost'}

    cp = sns.color_palette(n_colors=len(dict_suffix))
    for mk, mv in metrics.items():
        fmax_list = []
        median_fmax_list = []
        data_list = []
        ensemble_df_list = []
        # is_go = 'go' in sys.argv[-1]

        # ensemble_df
        for key, val in dict_suffix.items():
            # if len(key) > 0:
            #     go_dir = sys.argv[-1] + '_' + key
            # else:
            #     go_dir = sys.argv[-1]
            # if not '/' in key:
            #     go_dir = sys.argv[-1] + '_' + key
            #     sub_data_folder = ''
            # else:
            #     go_dir = sys.argv[-1] + '_EI'
            #     sub_data_folder = key+'/'
            go_dir = os.path.join(sys.argv[-1], key)
            sub_data_folder = ''
            fns = listdir(go_dir)
            # fns = [fn for fn in fns if fn != 'analysis']
            fns = [go_dir + '/' + fn for fn in fns]
            term_dirs = [fn for fn in fns if isdir(fn)]
            # if len(feature_folders) == 0:
            #     feature_folders.append('./')
            # assert len(feature_folders) > 0
            performance_file_list = {}
            for term_dir in term_dirs:

                file_name = term_dir + '/' + sub_data_folder + 'analysis/' + 'performance.csv'
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
            # performance_df['data_name'] = performance_df['data_name'].apply(add_colon)
            go_terms_set = performance_df['data_name'].unique()
            # print(go_terms_set)
            # print(performance_df['data_name'].values[0])

            # ensemble_df = extract_df_by_method(performance_df, method='LR.S', drop_columns=['method'])
            ensemble_df = best_ensemble_score(performance_df, input=key, metric=mk)

            ensemble_df['Method'] = val
            ensemble_df['key'] = key

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

        # img_str = 'hpo'
        # if is_go:
        #     img_str = 'go'
        img_str = 'covid'
        ylabel = mv
        # print(sorted_dataname_list)
        # print(sorted_fmax_list)

        ensemble_df_cat = pd.concat(ensemble_df_list)

        print(ensemble_df_cat['Method'].unique())
        # print('shape before drop', ensemble_df_cat.shape)
        # ensemble_df_cat.dropna(inplace=True)
        # print('shape after drop', ensemble_df_cat.shape)
        best_metric_str = 'best_' + mk
        cd_input = ensemble_df_cat[['data_name', best_metric_str, 'key']]

        cd_input_df = cd_input.pivot_table(best_metric_str, ['data_name'], 'key').reset_index()
        cd_input_df.set_index('data_name', inplace=True)

        median_fmax_list = np.median(cd_input_df.values, axis=0)
        fmax_list = cd_input_df.values
        # data_list = [k for v, k in dict_suffix.items()]
        cd_list = cd_input_df.columns.tolist()
        data_list = [dict_suffix[c] for c in cd_list]
        # dict_value_list = [k for v, k in dict_suffix.items()]
        index_data_list = [data_list.index(k) for v, k in dict_suffix.items()]
        cp_new = [cp[idx] for idx in index_data_list]

        cd_indices = cd_input_df.index.tolist()
        cd_input_df.index = [i.capitalize() for i in cd_indices]

        cd_csv_fn = '{}cd_input_{}_{}_{}.csv'.format(plot_dir + 'cd_csv/', mk, file_prefix, 'covid19')
        cd_input_df.to_csv(cd_csv_fn, index_label=False)
        cmd = "R CMD BATCH --no-save --no-restore '--args cd_fn=\"{}\"' R/plotCDdiagram.R".format(cd_csv_fn)
        os.system(cmd)

        cd_input_df.dropna(inplace=True)

        # print(len(median_fmax_list))
        # print(fmax_list.shape)
        # print(len(data_list))
        # print(len(cp_new))
        ensemble_df_cat.rename(columns={'data_name': 'Outcome'}, inplace=True)

        sorted_list = sorted(zip(median_fmax_list, data_list, cp_new), reverse=True, key=lambda x: x[0])
        sorted_dataname_list = [s[1] for s in sorted_list]
        print(sorted_dataname_list)
        sorted_cp = [s[2] for s in sorted_list]

        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax1 = sns.boxplot(ax=ax1, y=best_metric_str, x='Method',
                          data=ensemble_df_cat, palette=sorted_cp, order=sorted_dataname_list,
                          linewidth=2, width=0.5)
        # for tick in ax1.get_xticklabels():
        #     tick.set_rotation(45)
        #     tick.set_horizontalalignment("right")
        ax1.set_ylabel(ylabel, fontsize=22, fontweight='semibold')
        ax1.set_xlabel('')
        ax1.set_title(title_name, fontweight='semibold')
        for tick in ax1.get_xticklabels():
            tick.set_fontsize(14)
            # tick.set_rotation(45)
            tick.set_fontweight('semibold')
            # tick.set_horizontalalignment("right")

        for tick in ax1.get_yticklabels():
            tick.set_fontsize(16)
            tick.set_fontweight('semibold')
        fig1.savefig('{}{}{}_{}_comparison.pdf'.format(plot_dir, 'covid19/', mk, file_prefix), bbox_inches="tight")

        deceased_outcome_since_prefix = 'DECEASED_AT_{}DAYS'
        deceased_outcome_since_prefix_plot = 'DECEASED\nAT_{}DAYS'
        outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
        # outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
        # deceased_days_timeframe = [3, 5, 7, 10]
        deceased_days_timeframe = [10, 7, 5, 3]
        for dday in deceased_days_timeframe:
            # outcomes.append(deceased_outcome_since_prefix.format(dday))
            outcomes[deceased_outcome_since_prefix.format(dday)] = deceased_outcome_since_prefix_plot.format(dday)
            #

        ensemble_df_cat.replace(outcomes)
        outcomes_order = [k for k, v in outcomes.items()]

        sorted_dataname_list = [s.replace('-\n', '').replace('\n',' ') for s in sorted_dataname_list]

        fig2, ax2 = plt.subplots(1, 1, figsize=(11, 6))
        ax2 = sns.barplot(ax=ax2, y=best_metric_str, x='Outcome', hue='Method',
                          hue_order=sorted_dataname_list,
                          data=ensemble_df_cat, palette=sorted_cp,
                          order=outcomes_order,
                          )
        # for tick in ax1.get_xticklabels():
        #     tick.set_rotation(45)
        #     tick.set_horizontalalignment("right")
        ax2.set_ylabel(ylabel, fontsize=22, fontweight='semibold')
        ax2.set_xlabel('')
        ax2.set_title(title_name, fontweight='semibold')
        for tick in ax2.get_xticklabels():
            tick.set_fontsize(14)
            # tick.set_rotation(45)
            tick.set_fontweight('semibold')
            # tick.set_horizontalalignment("right")

        for tick in ax2.get_yticklabels():
            tick.set_fontsize(16)
            tick.set_fontweight('semibold')
        fig2.savefig('{}{}{}_{}_bar_comparison.pdf'.format(plot_dir, 'covid19/', mk, file_prefix), bbox_inches="tight")