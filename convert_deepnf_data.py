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
    parser.add_argument('--deepnf_path', type=str, required=True, help='data path')
    parser.add_argument('--output_path', type=str, required=True, help='')
    parser.add_argument('--pkl', type=str, required=True, help='')
    # parser.add_argument('--rdim', '-R', type=int, default=10, help='desired reduced dimension')
    # parser.add_argument('--clf_as_view', '-cav', type=str2bool, default='false', help='desired reduced dimension')
    args = parser.parse_args()
    data_path = abspath(args.path)

    G = nx.read_edgelist(os.path.join(args.deepnf_path, 'coexpression_deepnf.txt'), edgetype=float, data=(('weight', float),))
    node_list = list(G.nodes())
    protein_list = pd.read_csv(os.path.join(args.deepnf_path, 'gene_list.txt'))

    with open(os.path.join(args.deepnf_path, args.pkl), 'rb') as handle:
        b = pickle.load(handle)
    print(b)
    print(type(b))

