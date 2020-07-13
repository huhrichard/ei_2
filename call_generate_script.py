from os import system
import sys

jobs_file = sys.argv[-1]
jobs_list = open(jobs_file, 'r').readlines()

for job in jobs_list:
    term = job.split(' ')[2]
    data_dir = job.split(' ')[3][:-1]
    lsf_fn = 'run_{}_{}_generate.lsf'.format(data_dir, term.split(':')[1])
    fn = open(lsf_fn, 'w')
    fn.write('#!/bin/bash\n')
    fn.write('#BSUB -J {}_{}\n'.format(data_dir, term))
    fn.write('#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 6\n#BSUB -W 10:00\n')
    fn.write('#BSUB -o gen_{}.stdout\n'.format(data_dir))
    fn.write('#BSUB -eo gen_{}.stderr\n'.format(data_dir))
    fn.write('module purge\nmodule load java\nmodule load python\nmodule load groovy\nmodule load selfsched\n')
    fn.write(job)

    fn.close()
    system('bsub < %s' % lsf_fn)
    system('rm %s' % lsf_fn)