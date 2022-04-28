import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dict_to_compare = {
                    '1000': '1000_',
                    '500-1000': '500_1000_',
                    '200-500': '200_500_',
                    '100-200': '100_200_',
                    '50-100': '50_100_',
                    # '10-50': '10_50_'
                    }

fpath = './plot/performances'
perf_fmt = 'performances_cat_fmax_EI_go_{}.csv.gz'

stacker_list = {
             "RF.S": "S.RF",
             "SVM.S": "S.SVM",
             "NB.S": "S.NB",
             "LR.S": "S.LR",
             "AdaBoost.S": "S.AB",
             "DT.S": "S.DT",
             "GradientBoosting.S": "S.GB",
             "KNN.S": "S.KNN",
             "XGB.S": "S.XGB"
            }

highest_fmax_df_list = []

for k, v in dict_to_compare.items():
    perf_path = os.path.join(fpath, perf_fmt.format(k))
    perf_df = pd.read_csv(perf_path, compression='gzip')
    perf_df = perf_df.loc[perf_df['method'] != 'best base']
    highest_fmax_df = perf_df.sort_values('fmax', ascending=False).drop_duplicates('data_name').reset_index()
    method_name = highest_fmax_df['method'].unique()
    highest_fmax_df.replace(stacker_list, inplace=True)
    fig2 = plt.figure(figsize=(13, 6))
    ax2 = fig2.add_subplot(111)
    # ax2 = sns.countplot(ax=ax2, x='method', data=highest_fmax_df)
    ax2 = sns.countplot(ax=ax2, y='method', data=highest_fmax_df,
                        order=highest_fmax_df['method'].value_counts().index)
    ax2.set_xlabel('Ensemble Methods of EI')

    fig2.savefig('{}/ens_histogram_go_{}.pdf'.format(fpath, k), bbox_inches="tight")
    highest_fmax_df_list.append(highest_fmax_df)

highest_fmax_df_cat = pd.concat(highest_fmax_df_list)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111)
# ax = sns.countplot(ax=ax, x='method', data=highest_fmax_df_cat)
ax = sns.countplot(ax=ax, y='method', data=highest_fmax_df_cat,
                   log=True, order=highest_fmax_df_cat['method'].value_counts().index)
for tick in ax.get_xticklabels():
    tick.set_fontsize(22)
    # tick.set_rotation(45)
    tick.set_fontweight('semibold')
    # tick.set_horizontalalignment("right")

for tick in ax.get_yticklabels():
    tick.set_fontsize(22)
    tick.set_fontweight('semibold')

ax.set_ylabel('')
ax.set_xlabel('Number of GO terms', fontsize=24, fontweight='bold')



fig.savefig('{}/ens_histogram_go.pdf'.format(fpath), bbox_inches="tight")


