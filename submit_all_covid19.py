import os
import sys

base_path = '/sc/hydra/scratch/liy42/covid19_DECEASED_INDICATOR/'

list_of_method = ['EI', 'demographics',
                  'labs_type', 'medications_frequency', 'vitals_numeric', 'concatenated']

outcome_list = ['DECEASED_INDICATOR']

script = sys.argv[-1]

base_command = 'python {} --path {}'



for m in list_of_method:
    for outcome in outcome_list:
        lsf_fn = 'train_all_base_{}_{}.lsf'.format(outcome, m)
        script = open(lsf_fn, 'w')
        script.write(
            '#!/bin/bash\n#BSUB -J train_all_base\n#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 4\n#BSUB -W 10:00\n#BSUB -o train_all_base.stdout\n#BSUB -eo train_all_base.stderr\n#BSUB -R rusage[mem=10000]\n')
        script.write('module purge\nmodule load java\nmodule load groovy\nmodule load selfsched\n')
        dir_name = base_path+outcome+'_'+m
        print(dir_name)
        cmd = base_command.format(script, dir_name)
        script.write(cmd)
        script.close()
        os.system('bsub < {}'.format(lsf_fn))
        os.remove(lsf_fn)

