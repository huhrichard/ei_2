import os
import sys
import pandas as pd
from matplotlib import rc
import numpy as np
# import scikit_posthocs as sp


# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

base_path = '/sc/arion/scratch/liy42/covid19_DECEASED_INDICATOR/'


import matplotlib.pyplot as plt
import seaborn as sns



# list_of_method = ['EI', 'demographics',
#                   'labs', 'medications',
#                   'vitals', 'concatenated',
#                   'EI_svdImpute', 'EI_svdImpute_rank_5', 'EI_svdImpute_rank_20',
#                   'concatenated_svdImpute', 'concatenated_svdImpute_rank_5', 'concatenated_svdImpute_rank_20',
#                   'labs_svdImpute', 'labs_svdImpute_rank_5', 'labs_svdImpute_rank_20'
#                   ]

outcome_list = ['DECEASED_INDICATOR']

calling_script = str(sys.argv[-1])
plot_dir = './plot/covid19/'

base_command = 'python {} --path {}'

path_of_performance = '/analysis/performance.csv'

dict_of_method = {
                    # 'EI': 'EI',
                  'EI':'Ensemble\nIntegration',
                  'EI_PowerSet':'Ensemble Integration\nPower Set',
                  'demographics':'Demo-\ngraphics',
                  # 'labs':'Laboratory\ntests',
                  'labs':'Lab\ntests',
                  'medications': 'Medica-\ntions',
                  'comorbidities': 'Co-morbi-\ndities',
                  'vitals': 'Vital\nsigns',
                  'concatenated': 'Concantenated\nAll',
                  'EI_svdImpute': 'Ensemble Integration\nSVDImpute',
                  'EI_svdImpute_rank_5':'Ensemble Integration\nSVDImpute(rank=5)',
                  'EI_svdImpute_rank_20': 'Ensemble Integration\nSVDImpute(rank=20)',
                  'concatenated_svdImpute': 'Concantenated All\nSVDImpute',
                  'concatenated_svdImpute_rank_5': 'Concantenated All\nSVDImpute(rank=5)',
                  'concatenated_svdImpute_rank_20': 'Concantenated All\nSVDImpute(rank=20)',
                  'labs_svdImpute': 'Labs\nSVDImpute',
                  'labs_svdImpute_rank_5':'Labs\nSVDImpute(rank=5)',
                  'labs_svdImpute_rank_20': 'Labs\nSVDImpute(rank=20)'
                  }

def plot_boxplot_fmax_auc(list_of_method, fig_fn_suffix):
    cp = sns.color_palette(n_colors=len(list_of_method))
    dict_suffix = [dict_of_method[k] for k in list_of_method]
    fmax_median_list = []
    auc_median_list = []
    performance_df_list = []
    for m in list_of_method:
        for outcome in outcome_list:
            dir_name = base_path+outcome+'_'+m
            df = pd.read_csv(dir_name+path_of_performance)
            df['data_name'] = df['data_name'].str.replace("DECEASED_INDICATOR_", "")
            df['data_name'] = df['data_name'].str.replace(m, dict_of_method[m])
            # df.rename(columns='')
            performance_df_list.append(df)
            fmax_median_list.append(df['fmax'].median())
            auc_median_list.append(df['auc'].median())

    performance_cat_df = pd.concat(performance_df_list)

    fmax_label = r'$F_{max}$'
    def custom_boxplot(boxplot_y_metric, boxplot_ylabel, metric_median_list):
        sorted_tuple = sorted(zip(metric_median_list, dict_suffix, cp), reverse=True, key=lambda x: x[0])
        sorted_algo_names = [s[1] for s in sorted_tuple]
        sorted_cp = [s[2] for s in sorted_tuple]
        print(sorted_tuple)

        sep_space = 1.5
        fig1, ax1 = plt.subplots(1,1, figsize=(7.5,6))
        ax1 = sns.boxplot(ax=ax1, y=boxplot_y_metric, x='data_name',
                          data=performance_cat_df,
                          palette=sorted_cp, order=sorted_algo_names,
                          linewidth=2, width=0.5
                          )


        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        ax1.set_ylabel(boxplot_ylabel, fontsize=22)
        ax1.set_xticks([1, 2, 2.8, 3.8, 4.6, 5.6, 6.6], sorted_algo_names)
        for tick in ax1.get_xticklabels():
            tick.set_fontsize(14)
            # tick.set_rotation(45)
            # tick.set_fontweight('bold')
            # tick.set_horizontalalignment("right")

        for tick in ax1.get_yticklabels():
            tick.set_fontsize(16)
            # tick.set_rotation(45)
            # tick.set_fontweight('bold')
            # tick.set_horizontalalignment("right")
            # tick.set_verticalalignment("center")
        ax1.set_xlabel('')
        # ax1.set_title('COVID-19 Deceased Prediction')
        fig1.savefig('{}covid19_{}_{}_comparison.pdf'.format(plot_dir, boxplot_y_metric, fig_fn_suffix), bbox_inches="tight")

        cd_input = performance_cat_df[['data_name', boxplot_y_metric, 'method']]

        cd_input_df = cd_input.pivot_table(boxplot_y_metric, ['method'], 'data_name').reset_index()
        cd_input_df.set_index('method', inplace=True)
        cd_csv_fn = '{}covid19_cd_input_{}_{}.csv'.format(plot_dir + 'cd_csv/', boxplot_y_metric, fig_fn_suffix)
        cd_input_df.to_csv(cd_csv_fn, index_label=False)
        cmd = "R CMD BATCH --no-save --no-restore '--args cd_fn=\"{}\"' R/plotCDdiagram.R".format(cd_csv_fn)
        os.system(cmd)
        # pairwise_df = sp.posthoc_nemenyi_friedman(cd_input_df)
        # print(pairwise_df)
        # pairwise_df.to_csv('{}covid19_pairwise_difference_{}_{}.csv'.format(plot_dir + 'cd_csv/', boxplot_y_metric, fig_fn_suffix))

    custom_boxplot('fmax', fmax_label, fmax_median_list)
    custom_boxplot('auc', 'AUC', auc_median_list)


list_of_method_dict = {'weka_impute':['EI', 'demographics',
                                  'labs', 'medications',
                                  'vitals','comorbidities',
                                      # 'concatenated',
                                      # 'EI_PowerSet'
                                      ],
                    # 'svd_impute': ['demographics', 'medications',
                    #                       'vitals', 'EI_svdImpute',
                    #                       'concatenated_svdImpute',
                    #                'labs_svdImpute','comorbidities',],
                    #    'svd_impute_rank5':  ['demographics', 'medications',
                    #                           'vitals', 'EI_svdImpute_rank_5',
                    #                           'concatenated_svdImpute_rank_5',
                    #                          'labs_svdImpute_rank_5',
                    #                          'comorbidities',],
                    #     'svd_impute_rank20':  ['demographics', 'medications',
                    #                           'vitals', 'EI_svdImpute_rank_20',
                    #                           'concatenated_svdImpute_rank_20',
                    #                            'labs_svdImpute_rank_20',
                    #                            'comorbidities']
                       }

for k, v in list_of_method_dict.items():
    plot_boxplot_fmax_auc(v, k)

# list_of_method_weka_impute = ['demographics', 'medications',
#                   'vitals', 'concatenated', 'EI_svdImpute',
#                   'concatenated_svdImpute', 'labs_svdImpute']