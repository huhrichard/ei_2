import os
import sys
import pandas as pd
from matplotlib import rc
import numpy as np
# import scikit_posthocs as sp
from itertools import chain, combinations

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('axes', linewidth=2)



import matplotlib.pyplot as plt
import seaborn as sns



# list_of_method = ['EI', 'demographics',
#                   'labs', 'medications',
#                   'vitals', 'concatenated',
#                   'EI_svdImpute', 'EI_svdImpute_rank_5', 'EI_svdImpute_rank_20',
#                   'concatenated_svdImpute', 'concatenated_svdImpute_rank_5', 'concatenated_svdImpute_rank_20',
#                   'labs_svdImpute', 'labs_svdImpute_rank_5', 'labs_svdImpute_rank_20'
#                   ]

# outcome_list = ['DECEASED_INDICATOR']
outcome_list = ['HP0012638', 'HP0000707']

calling_script = str(sys.argv[-1])
plot_dir = './plot/tcca_test/'

base_command = 'python {} --path {}'

path_of_performance = '/analysis/performance.csv'



dict_of_method = {
                    # 'EI': 'EI',
                  'EI':'Ensemble\nIntegration',
                  # 'EI_PowerSet':'Ensemble Integration\nPower Set',
                  'demographics':'Demo-\ngraphics\n(11)',
                  # 'labs':'Laboratory\ntests',
                  'labs':'Lab\ntests\n(49)',
                  'medications': 'Medica-\ntions\n(26)',
                  'comorbidities': 'Co-morbi-\ndities\n(19)',
                  'vitals': 'Vital\nsigns\n(6)',
                  'concatenated': 'Concat-\nenated\nAll',
                  # 'tcca': 'EI_TensorCCA()',
                  # 'medications_binary': 'Medica-\ntions\n(binary)\n(26)',
                  # 'EI_med_binary':'Ensemble\nIntegration\n(binary\nmed)',
                  # 'concatenated_med_binary': 'Concat-\nenated\nAll\n(binary\nmed)',
                  # 'EI_svdImpute': 'Ensemble Integration\nSVDImpute',
                  # 'EI_svdImpute_rank_5':'Ensemble Integration\nSVDImpute(rank=5)',
                  # 'EI_svdImpute_rank_20': 'Ensemble Integration\nSVDImpute(rank=20)',
                  # 'concatenated_svdImpute': 'Concantenated All\nSVDImpute',
                  # 'concatenated_svdImpute_rank_5': 'Concantenated All\nSVDImpute(rank=5)',
                  # 'concatenated_svdImpute_rank_20': 'Concantenated All\nSVDImpute(rank=20)',
                  # 'labs_svdImpute': 'Labs\nSVDImpute',
                  # 'labs_svdImpute_rank_5':'Labs\nSVDImpute(rank=5)',
                  # 'labs_svdImpute_rank_20': 'Labs\nSVDImpute(rank=20)'

                  }

rdim = np.array(range(10))+1
tcca_list = []
for r in rdim:
    k = 'tcca{}'.format(r)
    tcca_list.append(k)
    dict_of_method['tcca{}'.format(r)] = 'EI_TCCA\n({})'.format(r)

lm = []
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

list_of_data = ['demographics',
                  'labs', 'medications',
                  'vitals','comorbidities']

feature_power_set = powerset(list_of_data)

for s in feature_power_set:
    # print(s, len(s))
    if len(s) > 1 and len(s) < len(list_of_data):
        feat = ''
        dict_name = ''
        for sub in s:
            feat = feat + '+' + sub
            dict_name = dict_name + '+\n' + dict_of_method[sub]
        lm.append(feat[1:])
        dict_of_method[feat[1:]] = dict_name[2:]
print(dict_of_method)


def custom_boxplot(boxplot_y_metric, boxplot_ylabel, metric_median_list, dict_suffix, cp,
                   performance_cat_df, exp_name, outcome, fig_fn_suffix):
    # sorted_tuple = sorted(zip(metric_median_list, dict_suffix, cp), reverse=True, key=lambda x: x[0])
    # sorted_algo_names = [s[1] for s in sorted_tuple]
    # sorted_cp = [s[2] for s in sorted_tuple]

    sorted_tuple = list(zip(metric_median_list, dict_suffix, cp))
    sorted_algo_names = [s[1] for s in sorted_tuple]
    sorted_cp = [s[2] for s in sorted_tuple]
    print(sorted_tuple)

    sep_space = 1.5
    fig1, ax1 = plt.subplots(1, 1, figsize=(18, 6))
    ax1 = sns.boxplot(ax=ax1, y=boxplot_y_metric, x='data_name',
                      data=performance_cat_df,
                      palette=sorted_cp, order=sorted_algo_names,
                      linewidth=2, width=0.5
                      )

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    ax1.set_ylabel(boxplot_ylabel, fontsize=22, fontweight='semibold')
    # ax1.set_xticks([1, 2, 2.8, 3.8, 4.6, 5.6], sorted_algo_names)
    for tick in ax1.get_xticklabels():
        tick.set_fontsize(14)
        # tick.set_rotation(45)
        tick.set_fontweight('semibold')
        # tick.set_horizontalalignment("right")

    for tick in ax1.get_yticklabels():
        tick.set_fontsize(16)
        tick.set_fontweight('semibold')
        # tick.set_rotation(45)
        # tick.set_fontweight('bold')
        # tick.set_horizontalalignment("right")
        # tick.set_verticalalignment("center")
    ax1.set_xlabel('')
    # ax1.set_title('COVID-19 Deceased Prediction')
    fig1.savefig('{}covid19_{}_{}_comparison_{}.tif'.format(plot_dir, boxplot_y_metric, fig_fn_suffix, exp_name),
                 bbox_inches="tight",
                 pil_kwargs={"compression": "tiff_lzw"}
                 )

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 12))
    pivoted = performance_cat_df.pivot("data_name", "method", boxplot_y_metric)
    pivoted = pivoted.reindex(sorted_algo_names)
    pivoted = pivoted.reindex(pivoted.median().sort_values().index, axis=1)
    # pivoted.assign(m=pivoted.median(axis=1)).sort_values('m').drop('m', axis=1)

    print(pivoted)
    ax2 = sns.heatmap(ax=ax2,
                      data=pivoted,
                      annot=True, fmt='.3f')

    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    # fig2.tight_layout()
    fig2.savefig('{}covid19_{}_{}_heatmap_{}.tif'.format(plot_dir, boxplot_y_metric, fig_fn_suffix, exp_name),
                 bbox_inches="tight",
                 pil_kwargs={"compression": "tiff_lzw"}
                 )

    cd_input = performance_cat_df[['data_name', boxplot_y_metric, 'method']]

    cd_input_df = cd_input.pivot_table(boxplot_y_metric, ['method'], 'data_name').reset_index()
    cd_input_df.set_index('method', inplace=True)
    cd_csv_fn = '{}covid19_cd_input_{}_{}_{}_{}.csv'.format(plot_dir + 'cd_csv/', boxplot_y_metric, fig_fn_suffix,
                                                         exp_name, outcome)
    cd_input_df.to_csv(cd_csv_fn, index_label=False)
    cmd = "R CMD BATCH --no-save --no-restore '--args cd_fn=\"{}\"' R/plotCDdiagram.R".format(cd_csv_fn)
    os.system(cmd)
    # pairwise_df = sp.posthoc_nemenyi_friedman(cd_input_df)
    # print(pairwise_df)
    # pairwise_df.to_csv('{}covid19_pairwise_difference_{}_{}.csv'.format(plot_dir + 'cd_csv/', boxplot_y_metric, fig_fn_suffix))


def plot_boxplot_fmax_auc(list_of_method, fig_fn_suffix, base_path_tuple):
    exp_name, base_path = base_path_tuple

    # dict_suffix = [dict_of_method[k] for k in list_of_method]
    fmax_median_list = []
    auc_median_list = []
    performance_df_list = []
    print(list_of_method)
    for outcome in outcome_list:
        for m in list_of_method:
            # if m == 'concatenated' or m == 'EI':
            #     dir_name = base_path+outcome+'_'+m
            # else:
            #     dir_name = base_path+outcome+'_EI/'+m
            if m == 'concatenated' or m == 'EI':
                dir_name = base_path+outcome
            else:
                dir_name = base_path+outcome+'/'+m
            df = pd.read_csv(dir_name+path_of_performance)
            df = df[~(df['method']=='CES')]
            df = df[~(df['method']=='Mean')]
            df = df[~(df['method']=='best base')]
            df = df[~(df['method']=='DT.S')]
            df['data_name'] = df['data_name'].str.replace("DECEASED_INDICATOR_", "")
            # print(dict_of_method[m])
            if m == 'EI':
                # df['data_name'] = df['data_name'].str.replace(m, dict_of_method[m], regex=False)
                df['data_name'] = df['data_name'].str.replace(outcome, dict_of_method[m], regex=False)
            rows_with_plus =  df['data_name'].str.contains('\+')
            count =  df.loc[rows_with_plus, 'data_name'].str.count('\+',).add(1).astype(str)
            print(count)
            df.loc[rows_with_plus, 'data_name'] = '#dataset\nincluded=' + count
            print(df['data_name'])
            # df.rename(columns='')
            performance_df_list.append(df)
            # fmax_median_list.append(df['fmax'].median())
            # auc_median_list.append(df['auc'].median())


        performance_cat_df = pd.concat(performance_df_list)
        dict_suffix = performance_cat_df['data_name'].unique().tolist()
        fmax_median_list = [performance_cat_df.loc[performance_cat_df['data_name']==k, 'fmax'].median() for k in dict_suffix]
        auc_median_list = [performance_cat_df.loc[performance_cat_df['data_name']==k, 'auc'].median() for k in dict_suffix]
        cp = sns.color_palette(n_colors=len(dict_suffix))
        print(performance_cat_df['data_name'])
        fmax_label = r'$F_{max}$'
        custom_boxplot('fmax', fmax_label, fmax_median_list, dict_suffix, cp,
                   performance_cat_df, exp_name, outcome, fig_fn_suffix)
        custom_boxplot('auc', 'AUC', auc_median_list, dict_suffix, cp,
                   performance_cat_df, exp_name, outcome, fig_fn_suffix)





list_of_method_dict = {'weka_impute':[
                                      # 'demographics',
                                  # 'labs', 'medications',
                                  # 'vitals','comorbidities',
                                  #     'concatenated',
                                        'EI',
                                    # 'medications_binary', 'EI_med_binary', 'concatenated_med_binary'
                                      # 'EI_PowerSet'
                                      ]+tcca_list,
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
# base_path_27may = ('before_27May','/sc/arion/scratch/liy42/covid19_DECEASED_INDICATOR/')
base_path_27may = ('hpo_test','/sc/arion/scratch/liy42/EIdata_top2_hpo_EI/')
# base_path_1Jun = ('27MayToJun1','/sc/arion/scratch/liy42/covid19_DECEASED_INDICATOR_test/')

for k, v in list_of_method_dict.items():
    plot_boxplot_fmax_auc(v, k, base_path_27may)
    # plot_boxplot_fmax_auc(v, k, base_path_1Jun)

# list_of_method_weka_impute = ['demographics', 'medications',
#                   'vitals', 'concatenated', 'EI_svdImpute',
#                   'concatenated_svdImpute', 'labs_svdImpute']