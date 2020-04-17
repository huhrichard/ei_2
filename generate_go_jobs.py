import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from goatools.base import get_godag
from goatools.semantic import get_info_content
# from goatools.gosubdag.gosubdag import GoSubDag
from goatools.anno.factory import get_objanno
from goatools.semantic import TermCounts
from goatools.semantic import get_info_content
import math
import sys

from goatools.base import download_go_basic_obo
obo_fname = download_go_basic_obo()
from goatools.associations import dnld_assc

from goatools.obo_parser import GODag

# godag = GODag('go-basic_2018.obo')
godag = get_godag('go-basic.obo')
ontology = sys.argv[-1]
if ontology == 'go':
    is_go = True
else:
    is_go = False

if is_go:
    path = './not_on_github/tsv/'+'GO2HPO_binary.tsv'
else:
    path = './not_on_github/tsv/'+'pos-neg-O-10.tsv'
# path = 'GO_annotations_Sept2017_EI_experiments.tsv'
df = pd.read_csv(path, sep='\t',index_col=0)
number_protein = df.shape[0]
# print(df.shape)
# print(df.df.column
# print(df.shape)
# np_array = df.values
pos_entry = (df == 1).values
go_terms_from_tsv = df.columns
go_pos_count = sum(pos_entry)
# plt.figure()
# plt.hist(go_pos_count, bins=100)
# plt.show()

# suffix = '_experimental'
# suffix = ''
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
for suffix, val in dict_suffix.items():
    if suffix != '':
        ontology_suffix = ontology + '_' + suffix
    else:
        ontology_suffix = ontology
    go_by_count_dict = {'EIdata_500_1000_{}.jobs'.format(ontology_suffix):np.logical_and((go_pos_count>500), (go_pos_count<=1000)),
                        'EIdata_1000_{}.jobs'.format(ontology_suffix): go_pos_count > 1000,
                        'EIdata_200_500_{}.jobs'.format(ontology_suffix): np.logical_and((go_pos_count>200), (go_pos_count<500))}

    IC_list = []

    for fn, bool_array in go_by_count_dict.items():
        jobs_fn = './jobs/'+fn
        f = open(jobs_fn, 'w')
        go_stats = 0
        # plt.figure()
        go_by_groups = go_terms_from_tsv[bool_array]
        for go in go_by_groups:
            try:
                if is_go:
                    depth_go = godag[go].depth
                    if depth_go >= 2:
                        if suffix != '':
                            f.write('python generate_data.py {} {}/ {}\n'.format(go, fn.split('.')[0], suffix))
                        else:
                            f.write('python generate_data.py {} {}/ \n'.format(go, fn.split('.')[0]))
                else:
                    if suffix != '':
                        f.write('python generate_data.py {} {}/ {}\n'.format(go, fn.split('.')[0], suffix))
                    else:
                        f.write('python generate_data.py {} {}/ \n'.format(go, fn.split('.')[0]))

            except KeyError:
                pass

        # plt.hist(IC_list, weights=np.ones(len(IC_list))/len(IC_list))
        #
        # plt.title(fn.split('.')[0])
        # plt.xlabel('Information Content')
        # plt.ylabel('Fraction of GO')
        # IC_list = []
        # plt.savefig(fn.split('.')[0]+'.png')
        # plt.clf()
        # print(fn, go_stats)
        f.close()