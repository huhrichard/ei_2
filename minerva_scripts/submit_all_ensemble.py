import sys, os

ontology = sys.argv[-1]
lsf_fn = 'ensemble_all_{}.lsf'.format(ontology)
script = open(lsf_fn, 'w')
script.write('#!/bin/bash\n#BSUB -J ensemble_all\n#BSUB -P acc_pandeg01a\n#BSUB -q express\n#BSUB -n 1\n#BSUB -W 10:00\n#BSUB -o ensemble_all.stdout\n#BSUB -eo ensemble_all.stderr\n#BSUB -R rusage[mem=10000]\n')
script.write('module purge\nmodule load java\nmodule load python\nmodule load groovy\nmodule load selfsched\n')

python_cmd = 'python minerva_scripts/run_all_different_data_ensemble.py --path {}'.format(ontology)
script.write(python_cmd)
script.close()
os.system('bsub < {}'.format(lsf_fn))
os.remove(lsf_fn)

