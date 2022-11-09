import os
import sys
from itertools import chain, combinations
import numpy as np
from sys import argv
import argparse
sys.path.append('./')
from common import str2bool
# base_path = '/sc/arion/scratch/liy42/covid19_DECEASED_INDICATOR/'


list_of_method = ['EI',
                  'fam_hx', 'vital', 'labs', 'comorbidities', 'encounter'
                  # 'admission',
                  # 'labs',
                  # # 'medications',
                  # 'vitals', 'concatenated',
                  # 'comorbidities',
                  # 'xgboost'
                    # 'medications_binary', 'EI_med_binary', 'concatenated_med_binary'
                  # 'EI_svdImpute', 'EI_svdImpute_rank_5', 'EI_svdImpute_rank_20',
                  # 'concatenated_svdImpute', 'concatenated_svdImpute_rank_5', 'concatenated_svdImpute_rank_20',
                  # 'labs_svdImpute', 'labs_svdImpute_rank_5', 'labs_svdImpute_rank_20'
                  ]

parser = argparse.ArgumentParser(description='Feed some bsub parameters')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--rank', type=str2bool, default='false', help='attribute importance')
parser.add_argument('--script', type=str, default='none', help='attribute importance')
args = parser.parse_args()

base_path = args.path
script_name = args.script
rank = args.rank

# cmd_str = 'python {} --path {} --by_jobf false --rank {}'
cmd_str = 'python {} --path {} --rank {}'
for m in list_of_method:
    # if 'ensemble' in script_name:
    #     cmd_str = cmd_str +
    m_path = os.path.join(base_path, m)
    os.system(cmd_str.format(script_name, m_path, rank))
