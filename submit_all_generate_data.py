import sys, os

ontology = sys.argv[-1]
lsf_fn = 'generate_term_{}.lsf'.format(ontology)
script = open(lsf_fn, 'w')
script.write('#!/bin/bash\n#BSUB -J enemble_{}\n#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 1\n#BSUB -W 10:00\n#BSUB -o generate_term.stdout\n#BSUB -eo generate_term.stderr\n#BSUB -R rusage[mem=10000]\n'.format(ontology)
             )
script.write('module purge\nmodule load java\nmodule load python\nmodule load groovy\nmodule load selfsched\n')

python_cmd = 'python call_all_generate_script.py {}'.format(ontology)
script.write(python_cmd)
script.close()
os.system('bsub < {}'.format(lsf_fn))
os.remove(lsf_fn)

