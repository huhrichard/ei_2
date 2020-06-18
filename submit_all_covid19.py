import os
import sys
from itertools import chain, combinations
base_path = '/sc/arion/scratch/liy42/covid19_DECEASED_INDICATOR/'


list_of_method = ['EI', 'demographics',
                  'labs', 'medications',
                  'vitals', 'concatenated','comorbidities',
                    # 'medications_binary', 'EI_med_binary', 'concatenated_med_binary'
                  # 'EI_svdImpute', 'EI_svdImpute_rank_5', 'EI_svdImpute_rank_20',
                  # 'concatenated_svdImpute', 'concatenated_svdImpute_rank_5', 'concatenated_svdImpute_rank_20',
                  # 'labs_svdImpute', 'labs_svdImpute_rank_5', 'labs_svdImpute_rank_20'
                  ]

outcome_list = ['DECEASED_INDICATOR']

calling_script = str(sys.argv[-1])

base_command = 'python {} --path {}'

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

list_of_data = ['demographics',
                  'labs', 'medications',
                  'vitals','comorbidities']

feature_power_set = powerset(list_of_data)

for s in feature_power_set:
    # print(s, len(s))
    if len(s) > 1 and len(s) < len(list_of_data):
        feat = ''
        for sub in s:
            feat = feat + '+' + sub
        list_of_method.append(feat[1:])

for m in list_of_method:
    for outcome in outcome_list:
        lsf_fn = 'train_all_base_{}_{}.lsf'.format(outcome, m)
        script = open(lsf_fn, 'w')
        script.write(
            '#!/bin/bash\n#BSUB -J train_all_base\n#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 4\n#BSUB -W 10:00\n#BSUB -o train_all_base.stdout\n#BSUB -eo train_all_base.stderr\n#BSUB -R rusage[mem=10000]\n')
        script.write('module purge\nmodule load java\nmodule load groovy\nmodule load selfsched\n')
        dir_name = base_path+outcome+'_'+m
        print(dir_name)
        cmd = base_command.format(calling_script, dir_name)
        print(cmd)
        script.write(cmd)
        script.close()
        os.system('bsub < {}'.format(lsf_fn))
        os.remove(lsf_fn)

