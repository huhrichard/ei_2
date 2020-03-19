from os import system
import sys

jobs_file = sys.argv[-1]
jobs_list = open(jobs_file, 'r').readlines()

for job in jobs_list:
    term = job.split(' ')[-2]
    dir = job.split(' ')[-1]
    lsf_fn = 'run_{}_generate.lsf'.format(term)
    fn = open(lsf_fn, 'w')
    fn.write('#!/bin/bash\n')
    fn.write('#BSUB -J {}_{}\n'.format(dir, term))
    fn.write('#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 6\n#BSUB -W 1:00\n')
    fn.write('#BSUB -o {}_%J_{}.stdout\n'.format(dir, term))
    fn.write('#BSUB -eo {}_%J_{}.stderr\n'.format(dir, term))
    fn.write('#BSUB -R rusage[mem=20480]\nmodule purge\nmodule load java\nmodule load python\nmodule load groovy\nmodule load selfsched\n')
    fn.write(job)
    fn.close()
    system('bsub < %s' % lsf_fn)
    system('rm %s' % lsf_fn)