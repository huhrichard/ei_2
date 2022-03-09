import sys, os

ontology = sys.argv[-1]
queue_n = 'express'
lsf_fn = 'train_all_base_{}.lsf'.format(ontology)
script = open(lsf_fn, 'w')
script.write('#!/bin/bash\n#BSUB -J train_all_base\n#BSUB -P acc_pandeg01a\n#BSUB -q {}\n'
             '#BSUB -n 2\n#BSUB -W 12:00\n#BSUB -o train_all_base.stdout\n'
             '#BSUB -eo train_all_base.stderr\n#BSUB -R rusage[mem=10000]\n'.format(queue_n))
script.write('module purge\nmodule load java\nmodule load python\nmodule load groovy\n'
             'module load selfsched\n')

python_cmd = 'python minerva_scripts/run_all_different_data_train_base.py {}'.format(ontology)
script.write(python_cmd)
script.close()
os.system('bsub < {}'.format(lsf_fn))
os.remove(lsf_fn)

