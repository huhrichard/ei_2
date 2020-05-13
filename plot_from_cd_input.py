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

plt.rcParams.update({'font.size': 20, 'figure.figsize':(10, 6)})

obo_fname = download_go_basic_obo()
from os.path import abspath

import os, fnmatch
import sys
from os import system

# system('module load R')
plot_dir = './plot/'

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(path, name))
    return result

dict_suffix = ['Ensemble\nIntegration',
                'DeepNF',
                'Mashup',
                'Coexpression',
                'Co-occurrence',
                   # 'database': 'Database',
                'Curated\nDatabases',
                 'PPI',
                 'Fusion',
                 'Neighborhood']
cp = sns.color_palette(n_colors=len(dict_suffix))

godag = get_godag("go-basic.obo")
objanno = get_objanno('goa_human.gaf', 'gaf', godag=godag)
termcounts = TermCounts(godag, objanno.get_id2gos_nss())

list_ontology = ['go', 'hpo']
# list_of_groups = ['1000', '500-1000', '200-500', '100-200', '50-100', '10-50']
ylabel = r'$F_{max}$'
for ontology in list_ontology:
    # for group in list_of_groups:
    fn = 'cd_input_{}_*.csv'.format(ontology)

    file_list = find(fn, './')
    print(file_list)
    for file in file_list:

        group_fn_suffix = file.split('.csv')[0].split('cd_input_{}_'.format(ontology))[-1]

        if '-' not in group_fn_suffix:
            group = '>' + group_fn_suffix
        else:
            group = group_fn_suffix

        cd_df = pd.read_csv(file)
        # print(cd_df)

        algo_names = cd_df.columns
        if 'Coocuurence' in algo_names:
            cd_df.rename(columns={'Coocuurence': 'Co-occurrence'}, inplace=True)
        if 'Cooccurence' in algo_names:
            cd_df.rename(columns={'Cooccurence': 'Co-occurrence'}, inplace=True)
        if 'EI' in algo_names:
            cd_df.rename(columns={'EI': 'Ensemble\nIntegration'}, inplace=True)
        if 'Curated database' in algo_names:
            cd_df.rename(columns={'Curated database': 'Curated\nDatabases'}, inplace=True)

        cd_df.to_csv(file, index_label=False)
        cmd = "R CMD BATCH --no-save --no-restore '--args cd_fn=\"{}\"' R/plotCDdiagram.R".format(file.split('/')[-1])
        os.system(cmd)
        # cd_df.rename(columns={'EI':'Ensemble\nIntegration', })
        algo_names = cd_df.columns.values.tolist()
        cd_df_median = cd_df[dict_suffix].median()
        # algo_names =
        print(cd_df_median)
        sorted_tuple = sorted(zip(cd_df_median, dict_suffix, cp), reverse=True, key=lambda x: x[0])
        sorted_algo_names = [s[1] for s in sorted_tuple]
        sorted_cp = [s[2] for s in sorted_tuple]
        print(sorted_tuple)

        cd_df.reset_index(level=0, inplace=True)
        cd_df.rename(columns={'index': ontology}, inplace=True)
        if ontology == 'go':
            go_terms_set = set(list(cd_df[ontology]))
            gosubdag = GoSubDag(go_terms_set, godag)
            cd_df['go_depth'] = 0
            cd_df['go_ic'] = 0
            for go_term in go_terms_set:
                try:
                    depth = gosubdag.go2obj[go_term].depth
                    ic = get_info_content(go_term, termcounts)

                except:
                    depth = 0
                    ic = 0
                cd_df.loc[cd_df[ontology]==go_term, 'go_depth'] = depth
                cd_df.loc[cd_df[ontology]==go_term, 'go_ic'] = ic
            list_id_vars = [ontology, 'go_depth', 'go_ic']
        else:
            list_id_vars = [ontology]

        cd_df_melted = pd.melt(cd_df, id_vars=list_id_vars, value_vars=algo_names, value_name='fmax')
        cd_df_melted.rename(columns={'variable': 'algo'}, inplace=True)
        print(cd_df_melted)
        title_name = "#annotated proteins: {}".format(group)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1 = sns.boxplot(ax=ax1, y='fmax', x='algo',
                          data=cd_df_melted, palette=sorted_cp, order=sorted_algo_names)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment("right")
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel('')
        ax1.set_title(title_name)
        fig1.savefig('{}f_max_{}_comparison_{}.png'.format(plot_dir, ontology, group_fn_suffix), bbox_inches="tight")
        #
        if ontology == 'go':
            cd_df_melted.replace({'Ensemble\nIntegration':'EI'}, inplace=True)
            fig2_plot_only = ['Mashup', 'DeepNF', 'EI']
            # idx_sorted_dataname = [sorted_dataname_list.index(p) for p in fig2_plot_only]
            # cp_plot_only = [sorted_cp[idx] for idx in idx_sorted_dataname]
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2 = sns.boxplot(ax=ax2, y='fmax', x='go_depth',
                              data=cd_df_melted[cd_df_melted['algo'].isin(fig2_plot_only)],
                              # palette=c,
                              hue='algo', hue_order=fig2_plot_only,
                              order=sorted(set(cd_df_melted['go_depth'].values)))
            ax2.get_legend().remove()
            ax2.legend(loc='upper right')
            ax2.set_ylabel(ylabel)
            ax2.set_xlabel('Depth in GO Hierarchy')
            ax2.set_title(title_name)
            fig2.savefig('{}f_max_{}_by_depth_{}.png'.format(plot_dir, ontology, group_fn_suffix), bbox_inches="tight")

            # fig2_plot_only = ['Mashup', 'DeepNF', 'EI']
            # idx_sorted_dataname = [sorted_dataname_list.index(p) for p in fig2_plot_only]
            # cp_plot_only = [sorted_cp[idx] for idx in idx_sorted_dataname]

            ic_of_terms = cd_df_melted['go_ic'].values
            # _, bin_edges = np.histogram(ic_of_terms, bins=5)
            bin_edges = np.percentile(ic_of_terms, np.linspace(0, 100, 6))
            ic_group_list = []
            cd_df_melted['ic_group'] = 'temp'
            for idx, edge in enumerate(bin_edges[:-1]):
                next_edge = bin_edges[(idx + 1)]
                group_name = '{:.2f}-{:.2f}'.format(edge, next_edge)
                cd_df_melted.loc[(cd_df_melted['go_ic'] <= next_edge) & (
                            cd_df_melted['go_ic'] >= edge), 'ic_group'] = group_name
                ic_group_list.append(group_name)
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            ax3 = sns.boxplot(ax=ax3, y='fmax', x='ic_group',
                              data=cd_df_melted[cd_df_melted['algo'].isin(fig2_plot_only)],
                              # palette=c,
                              hue='algo', hue_order=fig2_plot_only,
                              order=ic_group_list)
            ax3.get_legend().remove()
            ax3.legend(loc='upper right')
            ax3.set_ylabel(ylabel)
            ax3.set_xlabel('Information Content')
            ax3.set_title(title_name)
            fig3.savefig('{}f_max_{}_by_ic_{}.png'.format(plot_dir, ontology, group_fn_suffix), bbox_inches="tight")
