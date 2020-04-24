
import os
from os import system

dir_prefix = '/sc/hydra/scratch/liy42/EIdata_'
dict_to_compare = {
                    # '1000': '1000_',
                    # '500-1000': '500_1000_',
                    # '200-500': '200_500_',
                    '100-200': '100_200_',
                    '50-100': '50_100_',
                    }
#
list_ontology = [
                 # 'go',
                 'hpo'
                ]
python_cmd = 'python performance_comparison.py {} {}'
system('module load R')

for o in list_ontology:
    for key, val in dict_to_compare.items():
        cmd = python_cmd.format(key, dir_prefix+val+o)
        print(cmd)
        system(cmd)
