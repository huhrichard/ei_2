from os import system
import sys

jobs_file = sys.argv[-1]
jobs_list = open(jobs_file, 'r').readlines()

# for job in jobs_list:
#     term = job.split(' ')[2]
#     data_dir = job.split(' ')[3][:-1]
#     lsf_fn = 'run_{}_{}_generate.lsf'.format(data_dir, term.split(':')[1])
#     fn = open(lsf_fn, 'w')
#     fn.write('#!/bin/bash\n')
#     fn.write('#BSUB -J {}_{}\n'.format(data_dir, term))
#     fn.write('#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 64\n#BSUB -W 10:00\n')
#     fn.write('#BSUB -o gen_{}.stdout\n'.format(data_dir))
#     fn.write('#BSUB -eo gen_{}.stderr\n'.format(data_dir))
#     fn.write('module purge\nmodule load java\nmodule load python\nmodule load groovy\nmodule load selfsched\n')
#     fn.write(job)
#
#     fn.close()
#     system('bsub < %s' % lsf_fn)
#     system('rm %s' % lsf_fn)

data = jobs_file.split('/')[-1].split('.')[0].split('_')[-1]
lsf_fn = 'run_{}_generate.lsf'.format(jobs_file.split('/')[-1].split('.')[0].split('_')[-1])
nc = 64
fn = open(lsf_fn, 'w')
fn.write('#!/bin/bash\n')
fn.write('#BSUB -J {}_{}\n'.format(data, 'generate'))
fn.write('#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n {}\n#BSUB -W 10:00\n'.format(nc))
fn.write('#BSUB -o gen_{}.stdout\n'.format(data))
fn.write('#BSUB -eo gen_{}.stderr\n'.format(data))
fn.write('module purge\nmodule load java\nmodule load python\nmodule load groovy\nmodule load selfsched\n')
fn.write('mpirun -np {} selfsched < {}'.format(nc, jobs_file))
# fn.write(job)

fn.close()
system('bsub < %s' % lsf_fn)
system('rm %s' % lsf_fn)