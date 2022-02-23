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
import os
from os.path import exists, abspath, isdir
from os import mkdir

from goatools.base import download_go_basic_obo
obo_fname = download_go_basic_obo()
from goatools.associations import dnld_assc

from goatools.obo_parser import GODag
import argparse

# godag = GODag('go-basic_2018.obo')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prepro_path', type=str, required=True, help='data path')
    parser.add_argument('--go_tsv', type=str, required=True, help='')
    parser.add_argument('--ontology', type=str, default='go', help='')
    # parser.add_argument('--rdim', '-R', type=int, default=10, help='desired reduced dimension')
    # parser.add_argument('--clf_as_view', '-cav', type=str2bool, default='false', help='desired reduced dimension')
    args = parser.parse_args()
    godag = get_godag(os.path.join(args.prepro_path, 'go.obo'))
    ontology = args.ontology
    if 'go' in ontology:
        is_go = True
    else:
        is_go = False

    # if is_go:
    #     path = './not_on_github/tsv/'+'GO2HPO_binary.tsv'
    # else:
    #     path = './not_on_github/tsv/'+'pos-neg-O-10.tsv'
    path = os.path.join(args.prepro_path, args.go_tsv)
    # path = 'GO_annotations_Sept2017_EI_experiments.tsv'
    df = pd.read_csv(path, sep='\t',index_col=0)
    print(df.shape)
    number_protein = df.shape[0]
    pos_entry = (df == 1).values
    go_terms_from_tsv = df.columns
    go_pos_count = sum(pos_entry)
    dict_suffix = {
                    # 'EI': 'EI',
                   # 'deepNF': 'DeepNF',
                   'mashup': 'mashup',
                   }

    sorted = np.argsort(-1*go_pos_count)[:2]
    print(sorted)
    top2_bool = np.zeros(len(go_pos_count)).astype(bool)
    print(go_pos_count[sorted])
    top2_bool[sorted] = True
    for suffix, val in dict_suffix.items():
        ontology_suffix = ontology + '_' + suffix
        go_by_count_dict = {
                            # 'EIdata_top2_{}.jobs'.format(ontology_suffix): top2_bool
                            'EIdata_500_1000_{}.jobs'.format(ontology_suffix):np.logical_and((go_pos_count>500), (go_pos_count<=1000)),
                            'EIdata_1000_{}.jobs'.format(ontology_suffix): go_pos_count > 1000,
                            'EIdata_200_500_{}.jobs'.format(ontology_suffix): np.logical_and((go_pos_count>200), (go_pos_count<=500)),
                            # 'EIdata_10_50_{}.jobs'.format(ontology_suffix): np.logical_and((go_pos_count>10), (go_pos_count<=50)),
                            # 'EIdata_10_{}.jobs'.format(ontology_suffix): go_pos_count<=10,
                            # 'EIdata_50_100_{}.jobs'.format(ontology_suffix): np.logical_and((go_pos_count>50), (go_pos_count<=100)),
                            # 'EIdata_100_200_{}.jobs'.format(ontology_suffix): np.logical_and((go_pos_count>100), (go_pos_count<=200)),
                            }

        IC_list = []

        for fn, bool_array in go_by_count_dict.items():
            jobs_fn = './jobs/'+fn
            f = open(jobs_fn, 'w')
            go_stats = 0
            go_group_dir = os.path.join('/sc/arion/scratch/liy42/', fn.split('.')[0])
            if not exists(go_group_dir):
                mkdir(go_group_dir)
            # plt.figure()
            go_by_groups = go_terms_from_tsv[bool_array]
            print(len(go_by_groups))
            for go in go_by_groups:
                try:
                    if is_go:
                        depth_go = godag[go].depth
                        if depth_go >= 2:
                            # f.write('python generate_data.py {} {}/ {}\n'.format(go, fn.split('.')[0], suffix))
                            f.write('python generate_data.py --outcome {} --output_dir {} --method {} --feature_csv_path {} --outcome_tsv_path {}\n'.format(go, go_group_dir, suffix, args.prepro_path, path))
                    else:
                        f.write('python generate_data.py --outcome {} --output_dir {} --method {} --feature_csv_path {} --outcome_tsv_path {}\n'.format(go, go_group_dir, suffix, args.prepro_path, path))
                except KeyError:
                    pass
            f.close()