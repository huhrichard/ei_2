
import os
from os import system

dir_prefix = '/sc/arion/scratch/liy42/EIdata_'
dict_to_compare = {
                    # '1000': '1000_',
                    # '500-1000': '500_1000_',
                    # '200-500': '200_500_',
                    # '100-200': '100_200_',
                    '50-100': '50_100_',
                    # '10-50': '10_50_'
                    }
#
list_ontology = [
                 'go',
                 # 'go_rwrImpute',
                 # 'hpo'
                ]
python_cmd = 'python performance_comparison.py {} {} {}'
system('module load R')

for o in list_ontology:
    for key, val in dict_to_compare.items():
        cmd = python_cmd.format(o, key, dir_prefix+val+o)
        print(cmd)
        system(cmd)
