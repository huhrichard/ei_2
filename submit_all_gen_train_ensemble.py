import sys, os

# ontology = sys.argv[-1]
lsf_fn = 'submit.lsf'
script = open(lsf_fn, 'w')
script.write('#!/bin/bash\n#BSUB -J submit{}\n#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 4\n#BSUB -W 10:00\n#BSUB -o generate_term.stdout\n#BSUB -eo generate_term.stderr\n#BSUB -R rusage[mem=10000]\n'.format(ontology)
             )
script.write('module purge\nmodule load java\nmodule load python\nmodule load groovy\nmodule load selfsched\n')

python_cmd = 'python all_gen_train_ensemble.py'
script.write(python_cmd)
script.close()
os.system('bsub < {}'.format(lsf_fn))
os.remove(lsf_fn)