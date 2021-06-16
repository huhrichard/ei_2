import os
import sys
from itertools import chain, combinations
import numpy as np
from sys import argv
# base_path = '/sc/arion/scratch/liy42/covid19_DECEASED_INDICATOR/'


list_of_method = ['EI', 'demographics',
                  'labs',
                  # 'medications',
                  'vitals', 'concatenated',
                  'comorbidities',
                  'xgboost'
                    # 'medications_binary', 'EI_med_binary', 'concatenated_med_binary'
                  # 'EI_svdImpute', 'EI_svdImpute_rank_5', 'EI_svdImpute_rank_20',
                  # 'concatenated_svdImpute', 'concatenated_svdImpute_rank_5', 'concatenated_svdImpute_rank_20',
                  # 'labs_svdImpute', 'labs_svdImpute_rank_5', 'labs_svdImpute_rank_20'
                  ]

base_path = argv[-1]
script_name = argv[-2]
cmd_str = 'python {} --path {}'
for m in list_of_method:
    m_path = os.path.join(base_path, m)
    os.system(cmd_str.format(script_name, m_path))
