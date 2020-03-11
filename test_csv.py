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

from goatools.base import download_go_basic_obo
# obo_fname = download_go_basic_obo()
from goatools.associations import dnld_assc

from goatools.obo_parser import GODag

# godag = GODag('go-basic_2018.obo')
godag = get_godag('go-basic.obo')

path = './not_on_github/tsv/GO2HPO_binary.tsv'
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

go_by_count_dict = {'EIdata_500_1000_go.jobs':np.logical_and((go_pos_count>500), (go_pos_count<=1000)),
                    'EIdata_1000_go.jobs': go_pos_count > 1000,
                    'EIdata_200_500_go.jobs': np.logical_and((go_pos_count>200), (go_pos_count<500))}


# print(pos_count)

# go_annotated_500_to_1000_jobs_fn = 'EIdata_500_1000_go.jobs'
# go_500_to_1000_count = np.logical_and((pos_count>500), (pos_count<=1000))
#
# go_annotated_more_1000_jobs_fn = 'EIdata_1000_go.jobs'
# go_1000_count = pos_count > 1000
#
# go_annotated_200_to_500_jobs_fn = 'EIdata_200_500_go.jobs'
# go_200_to_500_count = np.logical_and((pos_count>200), (pos_count<500))


# fmt_str = ('{I:2}) {NS} {GO:10} {dcnt:4}  D{depth:02}   '
#            '{hsa:6.3f} {mmu:6.3f} {dme:6.3f} '
#            '{GO_name}')
#
# # Print selected GO information
# print('                              |<----- tinfo --->|')
# print('IDX NS GO ID      dcnt Depth   hsa    mmu    dme  Name')
# print('--- -- ---------- ---- ------ ------ ------ ----- --------------------')

# # Choose a deep leaf-level GO ID associated with "bacteria"
# DESC = 'bacteria'            # GO Term name contains this
# NSPC = 'cellular_component'  # Desired namespace
#
# # Create a chooser function which returns True or False
# def chooser(goterm):
#     """Choose a leaf-level GO term based on its name"""
#     b_match = DESC in goterm.name
#     # True if GO term is leaf-level (has no children)
#     b_leaf = not goterm.children
#     # True if GO term is in 'cellular_component' namespace (nspc)
#     b_nspc = goterm.namespace == NSPC
#     return b_match and b_leaf and b_nspc
#
# # Get GO terms with desired name in desired GO DAG branch
# go_ids_selected = set(o.item_id for o in godag.values() if chooser(o))

# go_ids_selected = set(list(go_terms_from_tsv))

# gosubdag = GoSubDag(go_ids_selected, godag)

# go_id, go_term = max(gosubdag.go2obj.items(), key=lambda t: t[1].depth)

# Print GO ID, using print format in gosubdag
# print(go_id, go_term.name)


# go_ids_chosen = go_term.get_all_parents()

# hsa_objanno = get_objanno('goa_human.gpad', 'gpad', godag=godag)
# hsa_objanno = get_objanno('goa_human.gaf', 'gaf', godag=godag)

# term_count = TermCounts(godag, hsa_objanno)

associations = dnld_assc('goa_human.gaf', godag)
term_count = TermCounts(godag, associations)


go_ids_selected = set(go_terms_from_tsv)
# gosubdag = GoSubDag(go_ids_selected, godag, tcntobj=term_count)

IC_list = []

for fn, bool_array in go_by_count_dict.items():

    f = open(fn, 'w')
    go_stats = 0
    plt.figure()
    go_by_groups = go_terms_from_tsv[bool_array]
    for go in go_by_groups:

        try:
            depth_go = godag[go].depth
            # depth_go = gosubdag.go2nt[go].depth
            # ic = -np.log2()
            # tinfo_hsa_go = gosubdag.go2nt[go].tinfo
            # print(tinfo_go)
            tinfo_hsa_go = get_info_content(go, term_count)
            IC_list.append(tinfo_hsa_go)



            print(go, depth_go, tinfo_hsa_go)
            # if depth_go >= 2 and
            if depth_go >= 2 and tinfo_hsa_go > 5:
            # if tinfo_hsa_go > 5:
            # if depth_go >= 2:
            # f.write('python generate_data.py {} {}/\n'.format(go, fn.split('.')[0]))
                f.write(go+'\n')
                go_stats += 1
        except KeyError:
            pass

    plt.hist(IC_list, weights=np.ones(len(IC_list))/len(IC_list))

    plt.title(fn.split('.')[0])
    plt.xlabel('Information Content')
    plt.ylabel('Fraction of GO')
    IC_list = []
    plt.savefig(fn.split('.')[0]+'.png')
    plt.clf()
    print(fn, go_stats)
    f.close()