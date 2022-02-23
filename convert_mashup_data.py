from sys import argv
import pickle
import argparse
from os.path import exists, abspath, isdir
import networkx as nx
import pandas as pd
import os
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mashup_path', type=str, required=True, help='data path')
    parser.add_argument('--output_path', type=str, required=True, help='')
    # parser.add_argument('--pkl_path', type=str, required=True, help='')
    # parser.add_argument('--rdim', '-R', type=int, default=10, help='desired reduced dimension')
    # parser.add_argument('--clf_as_view', '-cav', type=str2bool, default='false', help='desired reduced dimension')
    args = parser.parse_args()
    # data_path = abspath(args.path)

    # G = nx.read_edgelist(os.path.join(args.deepnf_path, 'coexpression_deepnf.txt'), edgetype=float, data=(('weight', float),))
    # node_list = list(G.nodes())
    # print(node_list)
    # print(len(node_list))
    # node_list_int = [int(i) for i in node_list]
    protein_df = pd.read_csv('../oracle_data/go_string_2022/GO_STRING_2022/edge_list/gene_list.txt')
    # print(protein_df)
    # print(protein_list.shape)
    protein_list = list(protein_df.columns.tolist())
    for p in list(protein_df.values):
        protein_list.append(str(p[0]))
    print(protein_list)

    mashup_csv = pd.read_csv(args.mashup_path, header=None)
    mashup_csv = mashup_csv.T

    mashup_cols = ['mashup_{}'.format(i) for i in range(mashup_csv.shape[1])]
    print(mashup_csv)
    mashup_df = pd.DataFrame(mashup_csv.values,
                             index=protein_list,
                             columns=mashup_cols).sort_index()
    print(mashup_df)
    mashup_df.index = protein_list
    print(mashup_df)
    mashup_df.to_csv(os.path.join(args.output_path, 'mashup.csv'))


