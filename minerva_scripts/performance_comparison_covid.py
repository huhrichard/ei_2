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
sys.path.append('./')
import common

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
    metrics = {'fmax': r'$\bf F_{max}$',
               'auc': 'AUC',
               'auprc': 'AUPRC'}
    # group = sys.argv[-2]
    # if '-' not in group:
    #     group = '>' + group
    # title_name = "#annotated proteins: {}".format(group)
    title_name = "COVID-19 death predictors"
    file_prefix = 'covid-19'

    dict_suffix = {'EI': 'Ensemble\nIntegration',
                   # 'EI_PowerSet':'Ensemble Integration\nPower Set',
                   'admission': 'Admission\n(23)',
                   # 'labs':'Laboratory\ntests',
                   'labs': 'Lab\ntests\n(44)',
                   # 'medications': 'Medica-\ntions\n(26)',
                   'comorbidities': 'Co-morbi-\ndities\n(23)',
                   'vitals': 'Vital\nsigns\n(9)',
                   # 'concatenated': 'Concat-\nenated\nAll',
                   'xgboost': 'XGBoost'}

    cp = sns.color_palette(n_colors=len(dict_suffix))
    for mk, mv in metrics.items():
        fmax_list = []
        median_fmax_list = []
        data_list = []
        ensemble_df_list = []
        performance_plot_df = []
        # is_go = 'go' in sys.argv[-1]

        # ensemble_df
        prediction_plot_df = []
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
            print(term_dirs)
            # if len(feature_folders) == 0:
            #     feature_folders.append('./')
            # assert len(feature_folders) > 0
            performance_file_list = {}
            for term_dir in term_dirs:

                performance_file_name = term_dir + '/' + sub_data_folder + 'analysis/' + 'performance.csv'
                prediction_file_name = term_dir + '/' + sub_data_folder + 'analysis/' + 'predictions.csv'
                # print(file_name)
                term_name = term_dir.split('/')[-1]

                if exists(performance_file_name):
                    performance_file_list[term_name] = (performance_file_name, prediction_file_name)
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
            prediction_df_list = []
            for term_name, (performance_file, prediction_file) in performance_file_list.items():
                df = pd.read_csv(performance_file)
                df['data_name'] = term_name

                pred_df = pd.read_csv(prediction_file)
                pred_df['data_name'] = term_name
                prediction_df_list.append(pred_df)
                # print(df)
                performance_df_list.append(df)

            performance_df = pd.concat(performance_df_list)
            prediction_df = pd.concat(prediction_df_list)
            # print(performance_df.columns)
            # performance_df['data_name'] = performance_df['data_name'].apply(add_colon)
            go_terms_set = performance_df['data_name'].unique()
            # print(go_terms_set)
            # print(performance_df['data_name'].values[0])

            # ensemble_df = extract_df_by_method(performance_df, method='LR.S', drop_columns=['method'])
            ensemble_df = best_ensemble_score(performance_df, input=key, metric=mk)

            best_performing_dfs = []
            for term_name, (_perf, _pred) in performance_file_list.items():
                best_performer = ensemble_df.loc[ensemble_df['data_name'] == term_name,'best_ensemble_method'].values
                # print(prediction_df)
                best_performer_pred = prediction_df.loc[prediction_df['data_name'] == term_name,[best_performer[0], 'label', 'data_name']]
                best_performer_pred.rename(columns={best_performer[0]:'prediction'}, inplace=True)
                best_performing_dfs.append(best_performer_pred)

            best_performing_df = pd.concat(best_performing_dfs)
            best_performing_df['Method'] = val
            best_performing_df['key'] = key

            ensemble_df['Method'] = val
            ensemble_df['key'] = key
            performance_df['Method'] = val
            performance_df['key'] = key

            # performance_df['delta_fmax_LR.S'] = performance_df['fmax_LR.S'] - performance_df['fmax_best base']
            # best_base_df = extract_df_by_method(performance_df, method='best base')
            # performance_df_dict[val] = performance_df
            # print(val, group, ensemble_df.shape)
            # fmax_list.append(ensemble_df['best_fmax'].values)
            # median_fmax_list.append(np.nanmedian(ensemble_df['best_fmax'].values))
            # data_list.append(val)
            ensemble_df_list.append(ensemble_df)
            performance_plot_df.append(performance_df)
            prediction_plot_df.append(best_performing_df)



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
        performance_df_cat = pd.concat(performance_plot_df)
        best_performer_pred_cat = pd.concat(prediction_plot_df)

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
        ax1.set_ylabel(ylabel, fontsize=22, fontweight='bold')
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

        """
        boxplot with only deceased indicator
        """
        deceased_outcome_since_prefix = 'DECEASED_AT_{}DAYS'
        # deceased_outcome_since_prefix_plot = 'Deceased in {}days'
        outcomes_newline_dict = {'DECEASED_INDICATOR': 'Mortality during hospitalization',
                                 'DECEASED_in_0-3_DAYS': 'Mortality within 0-3 days',
                                 'DECEASED_in_3-5_DAYS': 'Mortality within 3-5 days',
                                 # 'DECEASED_in_5-7_DAYS': 'Deceased in 5 to 7 days',
                                 # 'DECEASED_in_7-10_DAYS': 'Deceased in 7 to 10 days',
                                 # 'DECEASED_in_0-5_DAYS': 'Deceased in 0 to 5 days',
                                 # 'DECEASED_in_0-7_DAYS': 'Deceased in 0 to 7 days',
                                 # 'DECEASED_in_0-10_DAYS': 'Deceased in 0 to 10 days',
                                 # 'DECEASED_after_10_DAYS': 'Deceased after 10 days',
                                 'DECEASED_after_5_DAYS': 'Mortality after 5 days',
                                 }
        # outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
        # deceased_days_timeframe = [3, 5, 7, 10]
        # deceased_days_timeframe = [10, 7, 5, 3]
        # for dday in deceased_days_timeframe:
            # outcomes.append(deceased_outcome_since_prefix.format(dday))
            # outcomes_newline_dict[deceased_outcome_since_prefix.format(dday)] = deceased_outcome_since_prefix_plot.format(dday)
            #

        for out_k, out_v in outcomes_newline_dict.items():
            performance_df_cat_di_only = performance_df_cat.loc[performance_df_cat['data_name'] == out_k]

            xgb_series = performance_df_cat_di_only.loc[performance_df_cat_di_only['Method'] == 'XGBoost']
            performance_df_cat_di_only = performance_df_cat_di_only.loc[performance_df_cat_di_only['key'] != 'XGBoost']

            sorted_list = sorted(zip(median_fmax_list, data_list, cp_new), reverse=True, key=lambda x: x[0])
            sorted_dataname_list = [s[1] for s in sorted_list]
            print(sorted_dataname_list)
            sorted_cp = [s[2] for s in sorted_list]

            print(performance_df_cat_di_only.columns)
            print(xgb_series)

            xgb_idx = sorted_dataname_list.index('XGBoost')
            sorted_cp_no_xgb = sorted_cp.copy()
            sorted_cp_no_xgb.pop(xgb_idx)

            sorted_dataname_list_no_xgb = sorted_dataname_list.copy()
            sorted_dataname_list_no_xgb.pop(xgb_idx)

            fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
            ax3 = sns.boxplot(ax=ax3, y=mk, x='Method',
                              data=performance_df_cat_di_only, palette=sorted_cp_no_xgb, order=sorted_dataname_list_no_xgb,
                              linewidth=2, width=0.5)
            ax3 = sns.stripplot(ax=ax3, y=mk, x='Method',
                              data=performance_df_cat_di_only, order=sorted_dataname_list_no_xgb,
                                size=6, color=".3", linewidth=0)

            ax3.axhline(y=xgb_series[mk].values, color='r', ls='--', label='XGBoost')
            # for tick in ax3.get_xticklabels():
            #     tick.set_rotation(45)
            #     tick.set_horizontalalignment("right")
            ax3.set_ylabel(ylabel, fontsize=24, fontweight='bold')
            ax3.set_xlabel('')
            # ax3.set_title(out_v, fontweight='semibold', fontsize=22)
            ax3.legend(loc='upper right', prop={'weight':'bold', 'size':20})
            # ax3.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
            #            mode="expand", borderaxespad=0, ncol=3, prop={'weight':'bold', 'size':20})
            for tick in ax3.get_xticklabels():
                tick.set_fontsize(22)
                # tick.set_rotation(45)
                tick.set_fontweight('semibold')
                # tick.set_horizontalalignment("right")

            for tick in ax3.get_yticklabels():
                tick.set_fontsize(22)
                tick.set_fontweight('semibold')
            fig3.savefig('{}{}{}_{}_comparison_{}.pdf'.format(plot_dir, 'covid19/', mk,
                                                              file_prefix, out_k), bbox_inches="tight")

            # TODO: make AUPRC plot
            if (mk == 'fmax') and (out_k == 'DECEASED_INDICATOR'):
                best_performer_out_k = best_performer_pred_cat.loc[best_performer_pred_cat['data_name'] == out_k]
                best_performer_prc = []
                best_performer_prmax = {}
                for pred_method in sorted_dataname_list:
                    best_performer_outk_mk = best_performer_out_k.loc[best_performer_out_k['Method'] == pred_method]
                    print(best_performer_outk_mk)
                    fs = common.fmeasure_score(best_performer_outk_mk.label, best_performer_outk_mk.prediction, None)
                    p_curve, r_curve = fs['PR-curve']
                    best_performer_prc_df = pd.DataFrame({'precision':p_curve, 'recall':r_curve})
                    best_performer_name = ensemble_df_cat.loc[(ensemble_df_cat['Outcome']==out_k) & (ensemble_df_cat['Method']==pred_method), 'best_ensemble_method'].values
                    print(best_performer_name)
                    if pred_method != 'XGBoost':
                        best_performer_prc_df['method'] = '{}\n({})'.format(pred_method.split('\n(')[0], best_performer_name[0])
                    else:
                        best_performer_prc_df['method'] = 'XGBoost'
                    pmax = fs['P']
                    rmax = fs['R']
                    best_performer_prmax[pred_method] = [pmax, rmax]
                    best_performer_prc.append(best_performer_prc_df)

                best_performer_prc_cat = pd.concat(best_performer_prc)
                fig_prc, ax_prc = plt.subplots(1, 1, figsize=(10, 8))
                ax_prc = sns.lineplot(ax=ax_prc, data=best_performer_prc_cat,
                                      x="recall", y="precision", hue="method",
                                      palette=sorted_cp, sizes=(2.5,2.5), ci=None)
                ax_prc.legend(loc='upper right', prop={'weight': 'bold', 'size': 14}).set_title(None)
                ax_prc.set_xticks(np.arange(0,1.2,0.2))
                ax_prc.set_yticks(np.arange(0,1.2,0.2))
                ax_prc.set_ylabel('Precision', fontsize=24, fontweight='bold')
                ax_prc.set_ylabel('Recall', fontsize=24, fontweight='bold')
                for tick in ax_prc.get_xticklabels():
                    tick.set_fontsize(22)
                    # tick.set_rotation(45)
                    tick.set_fontweight('semibold')
                    # tick.set_horizontalalignment("right")

                for tick in ax_prc.get_yticklabels():
                    tick.set_fontsize(22)
                    tick.set_fontweight('semibold')

                for pr_idx, (pred_method, pr_list) in enumerate(best_performer_prmax.items()):
                    ax_prc.plot(pr_list[0], pr_list[1], c=sorted_cp[pr_idx])
                    # best_performer_prmax[pred_method] = [pmax, rmax]

                fig_prc.savefig('{}{}{}_{}_comparison.pdf'.format(plot_dir, 'covid19/', 'PRcurve', file_prefix), bbox_inches="tight")






            # Without xgb

            performance_df_cat_di_only = performance_df_cat.loc[performance_df_cat['data_name'] == out_k]

            xgb_series = performance_df_cat_di_only.loc[performance_df_cat_di_only['Method'] == 'XGBoost']
            performance_df_cat_di_only = performance_df_cat_di_only.loc[performance_df_cat_di_only['key'] != 'XGBoost']

            sorted_list = sorted(zip(median_fmax_list, data_list, cp_new), reverse=True, key=lambda x: x[0])
            sorted_dataname_list = [s[1] for s in sorted_list]
            print(sorted_dataname_list)
            sorted_cp = [s[2] for s in sorted_list]

            print(performance_df_cat_di_only.columns)
            print(xgb_series)

            xgb_idx = sorted_dataname_list.index('XGBoost')
            sorted_cp_no_xgb = sorted_cp.copy()
            sorted_cp_no_xgb.pop(xgb_idx)

            sorted_dataname_list_no_xgb = sorted_dataname_list.copy()
            sorted_dataname_list_no_xgb.pop(xgb_idx)

            fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
            ax3 = sns.boxplot(ax=ax3, y=mk, x='Method',
                              data=performance_df_cat_di_only, palette=sorted_cp_no_xgb,
                              order=sorted_dataname_list_no_xgb,
                              linewidth=2, width=0.5)

            # ax3.axhline(y=xgb_series[mk].values, color='r', ls='--', label='XGBoost')
            # for tick in ax3.get_xticklabels():
            #     tick.set_rotation(45)
            #     tick.set_horizontalalignment("right")
            ax3.set_ylabel(ylabel, fontsize=24, fontweight='bold')
            ax3.set_xlabel('')
            # ax3.set_title(out_v, fontweight='semibold', fontsize=22)
            ax3.legend(loc='upper right', prop={'weight':'bold', 'size':20})
            for tick in ax3.get_xticklabels():
                tick.set_fontsize(22)
                # tick.set_rotation(45)
                tick.set_fontweight('semibold')
                # tick.set_horizontalalignment("right")

            for tick in ax3.get_yticklabels():
                tick.set_fontsize(22)
                tick.set_fontweight('semibold')
            fig3.savefig('{}{}{}_{}_comparison_{}_withoutXGB.pdf'.format(plot_dir, 'covid19/', mk,
                                                              file_prefix, out_k), bbox_inches="tight")





        deceased_outcome_since_prefix = 'DECEASED_AT_{}DAYS'
        deceased_outcome_since_prefix_plot = 'Deceased\nin {}days'
        # outcomes_newline_dict = {'DECEASED_INDICATOR': 'Deceased\nIndicator',}
        # outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
        # deceased_days_timeframe = [3, 5, 7, 10]
        # deceased_days_timeframe = [10, 7, 5, 3]
        # for dday in deceased_days_timeframe:
        #     # outcomes.append(deceased_outcome_since_prefix.format(dday))
        #     outcomes_newline_dict[deceased_outcome_since_prefix.format(dday)] = deceased_outcome_since_prefix_plot.format(dday)

        outcomes_newline_dict = {'DECEASED_INDICATOR': 'Mortality\nduring\nhospitalization',
                                 'DECEASED_in_0-3_DAYS': 'Mortality\nwithin\n0-3 days',
                                 'DECEASED_in_3-5_DAYS': 'Mortality\nwithin\n3-5 days',
                                 # 'DECEASED_in_5-7_DAYS': 'Deceased in 5 to 7 days',
                                 # 'DECEASED_in_7-10_DAYS': 'Deceased in 7 to 10 days',
                                 # 'DECEASED_in_0-5_DAYS': 'Deceased in 0 to 5 days',
                                 # 'DECEASED_in_0-7_DAYS': 'Deceased in 0 to 7 days',
                                 # 'DECEASED_in_0-10_DAYS': 'Deceased in 0 to 10 days',
                                 # 'DECEASED_after_10_DAYS': 'Deceased after 10 days',
                                 'DECEASED_after_5_DAYS': 'Mortality\nafter\n5 days',
                                 }

        ensemble_df_cat.replace(outcomes_newline_dict, inplace=True)
        outcomes_order = [v for k, v in outcomes_newline_dict.items()]

        sorted_dataname_dict = {s:s.replace('-\n', '').replace('\n',' ').split('(')[0] for s in sorted_dataname_list}
        sorted_dataname_list = [v for k,v in sorted_dataname_dict.items()]
        ensemble_df_cat.replace(sorted_dataname_dict, inplace=True)
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
        ax2 = sns.barplot(ax=ax2, y=best_metric_str, x='Outcome', hue='Method',
                          hue_order=sorted_dataname_list,
                          data=ensemble_df_cat, palette=sorted_cp,
                          order=outcomes_order,
                          )
        ax2.legend(ncol=int(np.ceil(len(sorted_dataname_list)/2)),
                   prop={'weight':'bold', 'size':19})
        ax2.set_ylim([0, 1.05])
        # for tick in ax3.get_xticklabels():
        #     tick.set_rotation(45)
        #     tick.set_horizontalalignment("right")
        ax2.set_ylabel(ylabel, fontsize=24, fontweight='bold')
        ax2.set_xlabel('')
        # ax2.set_title(title_name, fontweight='semibold', fontsize=22)
        for tick in ax2.get_xticklabels():
            tick.set_fontsize(24)
            # tick.set_rotation(45)
            tick.set_fontweight('semibold')
            # tick.set_horizontalalignment("right")

        for tick in ax2.get_yticklabels():
            tick.set_fontsize(22)
            tick.set_fontweight('semibold')
        # ax2.legend(loc='upper right')
        fig2.savefig('{}{}{}_{}_bar_comparison.pdf'.format(plot_dir, 'covid19/', mk, file_prefix), bbox_inches="tight")
        ensemble_df_cat.to_csv(os.path.join(plot_dir, 'performance_cat_covid_{}.csv'.format(mk)))

        deceased_outcome_since_prefix = 'DECEASED_AT_{}DAYS'
        deceased_outcome_since_prefix_plot = 'Deceased\nin {}days'
        # outcomes_newline_dict = {'DECEASED_INDICATOR': 'Deceased\nIndicator',}
        # outcomes = {'DECEASED_INDICATOR': 'DECEASED\nINDICATOR'}
        # deceased_days_timeframe = [3, 5, 7, 10]
        # deceased_days_timeframe = [10, 7, 5, 3]
        # for dday in deceased_days_timeframe:
        #     # outcomes.append(deceased_outcome_since_prefix.format(dday))
        #     outcomes_newline_dict[deceased_outcome_since_prefix.format(dday)] = deceased_outcome_since_prefix_plot.format(dday)

        outcomes_newline_dict = {
                # 'DECEASED_INDICATOR': 'Mortality\nduring\nhospitalization',
                                 'DECEASED_in_0-3_DAYS': 'Mortality\nwithin\n0-3 days',
                                 'DECEASED_in_3-5_DAYS': 'Mortality\nwithin\n3-5 days',
                                 # 'DECEASED_in_5-7_DAYS': 'Deceased in 5 to 7 days',
                                 # 'DECEASED_in_7-10_DAYS': 'Deceased in 7 to 10 days',
                                 # 'DECEASED_in_0-5_DAYS': 'Deceased in 0 to 5 days',
                                 # 'DECEASED_in_0-7_DAYS': 'Deceased in 0 to 7 days',
                                 # 'DECEASED_in_0-10_DAYS': 'Deceased in 0 to 10 days',
                                 # 'DECEASED_after_10_DAYS': 'Deceased after 10 days',
                                 'DECEASED_after_5_DAYS': 'Mortality\nafter\n5 days',
                                 }

        # performance_df_cat_di_only = performance_df_cat.loc[performance_df_cat['data_name'] == out_k]

        xgb_series = ensemble_df_cat.loc[ensemble_df_cat['Method'] == 'XGBoost']
        performance_df_cat_noxgb = ensemble_df_cat.loc[ensemble_df_cat['key'] != 'XGBoost']

        sorted_list = sorted(zip(median_fmax_list, data_list, cp_new), reverse=True, key=lambda x: x[0])
        sorted_dataname_list = [s[1] for s in sorted_list]
        print(sorted_dataname_list)
        sorted_cp = [s[2] for s in sorted_list]

        # print(performance_df_cat_di_only.columns)
        print(xgb_series)

        xgb_idx = sorted_dataname_list.index('XGBoost')
        sorted_cp_no_xgb = sorted_cp.copy()
        sorted_cp_no_xgb.pop(xgb_idx)

        sorted_dataname_list_no_xgb = sorted_dataname_list.copy()
        sorted_dataname_list_no_xgb.pop(xgb_idx)

        performance_df_cat_noxgb.replace(outcomes_newline_dict, inplace=True)
        outcomes_order = [v for k, v in outcomes_newline_dict.items()]

        sorted_dataname_dict = {s: s.replace('-\n', '').replace('\n', ' ').split('(')[0] for s in sorted_dataname_list_no_xgb}
        sorted_dataname_list = [v for k, v in sorted_dataname_dict.items()]
        performance_df_cat_noxgb.replace(sorted_dataname_dict, inplace=True)
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        ax2 = sns.barplot(ax=ax2, y=best_metric_str, x='Outcome', hue='Method',
                          hue_order=sorted_dataname_list,
                          data=performance_df_cat_noxgb, palette=sorted_cp_no_xgb,
                          order=outcomes_order,
                          )
        ax2.legend(ncol=int(np.ceil(len(sorted_dataname_list)/2)),
                   prop={'weight':'bold','size':16})
        ax2.set_ylim([0, 1.05])
        # for tick in ax3.get_xticklabels():
        #     tick.set_rotation(45)
        #     tick.set_horizontalalignment("right")
        ax2.set_ylabel(ylabel, fontsize=24, fontweight='bold')
        ax2.set_xlabel('')
        # ax2.set_title(title_name, fontweight='semibold', fontsize=22)
        for tick in ax2.get_xticklabels():
            tick.set_fontsize(22)
            # tick.set_rotation(45)
            tick.set_fontweight('semibold')
            # tick.set_horizontalalignment("right")

        for tick in ax2.get_yticklabels():
            tick.set_fontsize(20)
            tick.set_fontweight('semibold')
        # ax2.legend(loc='upper right')
        fig2.savefig('{}{}{}_{}_bar_comparison_withoutXGB.pdf'.format(plot_dir, 'covid19/', mk, file_prefix), bbox_inches="tight")
        ensemble_df_cat.to_csv(os.path.join(plot_dir, 'performance_cat_covid_{}.csv'.format(mk)))
