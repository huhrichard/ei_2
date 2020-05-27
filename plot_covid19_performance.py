import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_path = '/sc/hydra/scratch/liy42/covid19_DECEASED_INDICATOR/'

list_of_method = ['EI', 'demographics',
                  'labs', 'medications',
                  'vitals', 'concatenated',
                  'EI_svdImpute', 'EI_svdImpute_rank_5', 'EI_svdImpute_rank_20',
                  'concatenated_svdImpute', 'concatenated_svdImpute_rank_5', 'concatenated_svdImpute_rank_20',
                  'labs_svdImpute', 'labs_svdImpute_rank_5', 'labs_svdImpute_rank_20'
                  ]

outcome_list = ['DECEASED_INDICATOR']

calling_script = str(sys.argv[-1])
plot_dir = './plot/covid19/'

base_command = 'python {} --path {}'

path_of_performance = '/analysis/performance.csv'

dict_of_method = {'EI':'Ensemble\nIntegration',
                  'demographics':'Demographics',
                  'labs':'Labs',
                  'medications': 'Medications',
                  'vitals': 'Vitals',
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
cp = sns.color_palette(n_colors=len(dict_of_method))
dict_suffix = [v for k, v in dict_of_method.items()]
fmax_median_list = []
auc_median_list = []
performance_df_list = []
for m, show in dict_of_method.items():
    for outcome in outcome_list:
        dir_name = base_path+outcome+'_'+m
        df = pd.read_csv(dir_name+path_of_performance)
        df['data_name'] = df['data_name'].str.replace("DECEASED_INDICATOR_", "")
        df['data_name'] = df['data_name'].str.replace(m, show)
        # df.rename(columns='')
        performance_df_list.append(df)
        fmax_median_list.append(df['fmax'].median())
        auc_median_list.append(df['auc'].median())

performance_cat_df = pd.concat(performance_df_list)

sorted_tuple_by_fmax = sorted(zip(fmax_median_list, dict_suffix, cp), reverse=True, key=lambda x: x[0])
sorted_algo_names_by_fmax = [s[1] for s in sorted_tuple_by_fmax]
sorted_cp_by_fmax = [s[2] for s in sorted_tuple_by_fmax]
print(sorted_tuple_by_fmax)

sorted_tuple_by_auc = sorted(zip(fmax_median_list, dict_suffix, cp), reverse=True, key=lambda x: x[0])
sorted_algo_names_by_auc = [s[1] for s in sorted_tuple_by_auc]
sorted_cp_by_auc = [s[2] for s in sorted_tuple_by_auc]
print(sorted_tuple_by_auc)
fmax_label = r'$F_{max}$'
def custom_boxplot(boxplot_y_metric, boxplot_ylabel, sorted_algo_names, sorted_cp):
    fig1, ax1 = plt.subplots(1,1, figsize=(6,12))
    ax1 = sns.boxplot(ax=ax1, y=boxplot_y_metric, x='data_name', data=performance_cat_df,
                      palette=sorted_cp, order=sorted_algo_names)
    ax1.set_ylabel(boxplot_ylabel, fontsize=22)
    for tick in ax1.get_xticklabels():
        tick.set_fontsize(16)
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
        tick.set_verticalalignment("center")

    ax1.set_title('COVID-19 Deceased Prediction')
    fig1.savefig('{}covid19_{}_comparison.png'.format(plot_dir, boxplot_y_metric), bbox_inches="tight")

custom_boxplot('fmax', fmax_label, sorted_algo_names_by_fmax, sorted_cp_by_fmax)
custom_boxplot('auc', 'AUC', sorted_algo_names_by_auc, sorted_cp_by_auc)
