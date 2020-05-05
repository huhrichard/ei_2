import sys, os

# ontology = sys.argv[-1]
lsf_fn = 'rwrImpute.lsf'
script = open(lsf_fn, 'w')
script.write('#!/bin/bash\n#BSUB -J train_all_base\n#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 4\n#BSUB -W 10:00\n#BSUB -o ensemble_all.stdout\n#BSUB -eo ensemble_all.stderr\n#BSUB -R rusage[mem=64000]\n')
python_cmd = 'python rwrImpute_adj_txt.py'
script.write(python_cmd)
script.close()
os.system('bsub < {}'.format(lsf_fn))
os.remove(lsf_fn)