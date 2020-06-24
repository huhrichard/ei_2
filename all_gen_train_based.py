from os import system
import sys

import os, fnmatch
from os.path import abspath
import sys
from os import system

def find_dir(pattern, path):
    result = []
    # for root, dirs, files in os.walk(path):
    dirs = os.listdir(path)
    for dir in dirs:
        # print(dir)
        if fnmatch.fnmatch(dir, pattern):
            result_dir = os.path.join(path, dir)
            # result_dir = dir
            print(result_dir)
            result.append(result_dir)

    return result


if __name__ == "__main__":
    ontology = sys.argv[-1]
    dir_list = find_dir('EIdata_*_{}*.jobs'.format(ontology), './jobs/')
    scratch_path = '/sc/arion/scratch/liy42/'
    # dir_list = find_dir('GO0071704', sys.argv[-1])
    for jobs_file in dir_list:
        jobs_list = open(jobs_file, 'r').readlines()

        for job in jobs_list:
            term = job.split(' ')[2]
            data_dir = job.split(' ')[3][:-1]
            term_dir = term.split(':')[0]+term.split(':')[1]
            lsf_fn = 'run_{}_{}.lsf'.format(data_dir, term.split(':')[1])
            fn = open(lsf_fn, 'w')
            fn.write('#!/bin/bash\n')
            fn.write('#BSUB -J {}_{}\n'.format(data_dir, term))
            fn.write('#BSUB -P acc_pandeg01a\n#BSUB -q premium\n#BSUB -n 6\n#BSUB -W 10:00\n')
            fn.write('#BSUB -o term_{}.stdout\n'.format(data_dir))
            fn.write('#BSUB -eo term_{}.stderr\n'.format(data_dir))
            fn.write('#BSUB -R rusage[mem=20480]\nmodule purge\nmodule load java\nmodule load python\nmodule load groovy\nmodule load selfsched\n')
            fn.write(job)
            train_base_cmd = 'python train_base.py --path {}\n'.format(scratch_path+data_dir+'/'+term_dir)
            # ensemble_cmd = 'python ensemble.py --path {}\n'.format(data_dir+'/'+term_dir)
            fn.write(train_base_cmd)
            # fn.write(ensemble_cmd)
            fn.close()
            system('bsub < %s' % lsf_fn)
            system('rm %s' % lsf_fn)