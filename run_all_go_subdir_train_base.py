import os, fnmatch
import sys
from os import remove, system
from os.path import abspath
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import glob

def find_dir(pattern, path):
    result = []
    # for root, dirs, files in os.walk(path):
    dirs = os.listdir(path)
    for dir in dirs:
        # print(dir)
        if fnmatch.fnmatch(dir, pattern):
            result_dir = abspath(os.path.join(path, dir))
            print(result_dir)
            result.append(result_dir)

    return result

parser = argparse.ArgumentParser(description='Feed some bsub parameters')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--queue', '-Q', type=str, default='premium', help='LSF queue to submit the job')
parser.add_argument('--node', '-N', type=str, default='2', help='number of node requested')
parser.add_argument('--time', '-T', type=str, default='12:00', help='number of hours requested')
parser.add_argument('--memory', '-M', type=str,default='10000', help='memory requsted in MB')
parser.add_argument('--classpath', '-CP', type=str,default='./weka.jar', help='path to weka.jar')
parser.add_argument('--hpc', '-MIN', type=str2bool,default='true', help='use hpc cluster or not')
parser.add_argument('--term_prefix', type=str, default='GO', help='term_prefix')

parser.add_argument('--seed', '-S', type=str,default='1', help='the seed use to generate cross-validataion data')
args = parser.parse_args()

def write_submit_del_job_list(scratch_path, jobs_fn):
    # second_sub = scratch_path.split('/')[-1]
    first_sub = scratch_path.split('/')[-1]
    # first_sub

    # lsf_fn = first_sub + '.lsf'
    # lsf_fn = '{}.lsf'.format(first_sub)
    lsf_fn = '{}.lsf'.format(first_sub)
    # print('submitting EI ensemble job to hpc...')
    ####### Write the lsf fileqn1
    script = open(lsf_fn, 'w')
    script.write(
        '#!/bin/bash\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -J %s\n#BSUB -W %s\n#BSUB -R rusage[mem=%s]\n#BSUB -n %s\n#BSUB -sp 100\n' % (
            args.queue, first_sub, args.time, args.memory, args.node))
    script.write('#BSUB -o train_base_%s.stdout\n#BSUB -eo train_base_%s.stderr\n#BSUB -L /bin/bash\n' % (first_sub, first_sub))
    # script.write('module purge')
    # script.write('conda activate largegopred')
    script.write(
        #     # 'module load python\n'+
        #     # 'module load py_packages\n'
        'module load java\nmodule load groovy\nmodule load selfsched\nmodule load weka\n')
    script.write('export _JAVA_OPTIONS=\"-XX:ParallelGCThreads=10\"\nexport JAVA_OPTS=\"-Xmx10g\"\n')
    script.write('mpirun selfsched < {}'.format(jobs_fn))
    # script.write(python_cmd)
    # script.write('rm %s.jobs' % second_sub)
    script.close()
    ####### Submit the lsf job and remove lsf script
    system('bsub < %s' % lsf_fn)
    remove(lsf_fn)


def write_submit_del_job(scratch_path, python_cmd):
    second_sub = scratch_path.split('/')[-2]
    first_sub = scratch_path.split('/')[-1]
    # first_sub

    # lsf_fn = first_sub + '.lsf'
    # lsf_fn = '{}.lsf'.format(first_sub)
    lsf_fn = '{}_{}.lsf'.format(second_sub, first_sub)
    # print('submitting EI ensemble job to hpc...')
    ####### Write the lsf fileqn1
    script = open(lsf_fn, 'w')
    script.write(
        '#!/bin/bash\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -J %s\n#BSUB -W %s\n#BSUB -R rusage[mem=%s]\n#BSUB -n %s\n#BSUB -sp 100\n' % (
            args.queue, first_sub, args.time, args.memory, args.node))
    script.write('#BSUB -o train_base_%s.stdout\n#BSUB -eo train_base_%s.stderr\n#BSUB -L /bin/bash\n' % (first_sub, first_sub))
    # script.write('module purge')
    # script.write('conda activate largegopred')
    # script.write(
        #     # 'module load python\n'+
        #     # 'module load py_packages\n'
        # 'module load java\nmodule load groovy\nmodule load selfsched\nmodule load weka\n')
    # script.write('export _JAVA_OPTIONS=\"-XX:ParallelGCThreads=10\"\nexport JAVA_OPTS=\"-Xmx10g\"\n')
    # script.write('mpirun selfsched < {}'.format(jobs_fn))
    script.write(python_cmd)
    # script.write('rm %s.jobs' % second_sub)
    script.close()
    ####### Submit the lsf job and remove lsf script
    system('bsub < %s' % lsf_fn)
    remove(lsf_fn)


if __name__ == "__main__":

    # dir_list = find_dir('{}*'.format(args.term_prefix), args.path)
    # dir_list = find_dir('GO0071704', sys.argv[-1])
    jobs_prefix = args.path.split('/')[-1]
    scratch_path = '/sc/arion/scratch/liy42/'
    jobs_file = find_dir('{}.jobs'.format(jobs_prefix), './jobs')
    print(jobs_file)
    # jobs_list = ['module load groovy']
    # system('module load groovy')
    # jobs_fstream = open(jobs_file[0], "r").read().split('\n')
    cat_dir = args.path
    jobs_fstream = os.listdir(cat_dir)

    # for go_job in jobs_fstream:
    #     term_name = go_job.split(' ')[2].replace(':', '')
    for go_job in os.walk(cat_dir, topdown=True):

        root, dirs, files = go_job
        num_sep = cat_dir.count(os.path.sep)
        num_sep_this = root.count(os.path.sep)
        print(root)
        print(cat_dir)
        if root == cat_dir:
            print(dirs)
            for dir in dirs:
                go_scratch_dir = os.path.join(root, dir)
                print(go_scratch_dir)
                python_cmd_train = 'python train_base.py --path {}'.format(go_scratch_dir)
                # python_cmd_train = 'python train_base.py --path {} --attr_imp True'.format(go_scratch_dir)
                write_submit_del_job(go_scratch_dir, python_cmd=python_cmd_train)
        else:
            break
        # jobs_list.append(python_cmd_train)
        # print(python_cmd_train)
        # system(python_cmd_train)
        # go_dir_splitted = go_dir.split('/')

    # jobs_txt.write('\n'.join(jobs_list))
    # jobs_txt.close()
    # write_submit_del_job(args.path, jobs_fn=jobs_n)

    # jobs_n = 'train_base_{}.jobs'.format(jobs_prefix)
    # jobs_txt = open(jobs_n, 'w')
    # jobs_list = ['module load groovy']
    # system('module load groovy')
    # for go_dir in dir_list:
    #
    #     python_cmd_train = 'python train_base.py --path {} --hpc=False'.format(go_dir)
    #     # write_submit_del_job(go_dir, python_cmd=python_cmd_train)
    #     jobs_list.append(python_cmd_train)
    #     # print(python_cmd_train)
    #     # # system(python_cmd)
    #     # go_dir_splitted = go_dir.split('/')
    #
    # jobs_txt.write('\n'.join(jobs_list))
    # jobs_txt.close()
    # write_submit_del_job_list(args.path, jobs_fn=jobs_n)

