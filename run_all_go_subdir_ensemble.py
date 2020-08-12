import os, fnmatch
import sys
from os import remove, system
from os.path import abspath
from os.path import abspath, isdir
from os import remove, system, listdir
import glob
import argparse
import subprocess

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Feed some bsub parameters')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--queue', '-Q', type=str, default='premium', help='LSF queue to submit the job')
parser.add_argument('--node', '-N', type=str, default='2', help='number of node requested')
parser.add_argument('--time', '-T', type=str, default='10:00', help='number of hours requested')
parser.add_argument('--memory', '-M', type=str,default='10000', help='memory requsted in MB')
parser.add_argument('--classpath', '-CP', type=str,default='./weka.jar', help='path to weka.jar')
parser.add_argument('--hpc', '-MIN', type=str2bool,default='true', help='use hpc cluster or not')
parser.add_argument('--term_prefix', type=str, default='GO', help='term_prefix')

parser.add_argument('--seed', '-S', type=str,default='1', help='the seed use to generate cross-validataion data')

args = parser.parse_args()

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

def write_submit_del_job(ensemble_dir):
    second_sub = ensemble_dir.split('/')[-1]
    first_sub = ensemble_dir.split('/')[-2]


    lsf_fn = first_sub + '_' + second_sub + '.lsf'
    print('submitting EI ensemble job to hpc...')
    ####### Write the lsf fileqn1
    script = open(lsf_fn, 'w')
    script.write(
        '#!/bin/bash\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -J %s\n#BSUB -W %s\n#BSUB -R rusage[mem=%s]\n#BSUB -n %s\n#BSUB -sp 100\n' % (
            args.queue, second_sub, args.time, args.memory, args.node))
    script.write('#BSUB -o %s.stdout\n#BSUB -eo %s.stderr\n#BSUB -L /bin/bash\n' % (second_sub, second_sub))
    # script.write('module purge')
    # script.write('conda activate largegopred')
    script.write(
        #     # 'module load python\n'+
        #     # 'module load py_packages\n'
        'module load java\nmodule load groovy\nmodule load selfsched\nmodule load weka\n')
    script.write('export _JAVA_OPTIONS=\"-XX:ParallelGCThreads=10\"\nexport JAVA_OPTS=\"-Xmx10g\"\n')
    # script.write('mpirun selfsched < %s.jobs\n' % second_sub)
    python_cmd = 'python ensemble.py --path {}\n'.format(ensemble_dir)
    print(python_cmd)
    script.write(python_cmd)
    # script.write('rm %s.jobs' % second_sub)
    script.close()
    ####### Submit the lsf job and remove lsf script
    system('bsub < %s' % lsf_fn)
    remove(lsf_fn)

excluding_folder = ['analysis']

if __name__ == "__main__":
    # file_list = find_dir('GO*',sys.argv[-1])
    dir_list = find_dir('{}*'.format(args.term_prefix), args.path)
    for go_dir in dir_list:

        data = go_dir.split('/')[-1]
        data_dir = go_dir.split('/')[-2]
        if data_dir.split('_')[-1] == 'EI':
            # p = subprocess.Popen('python tcca_projection.py --path {}'.format(go_dir))
            # p.wait()
            # os.system('python tcca_projection.py --path {}'.format(go_dir))
            fns = listdir(go_dir)
            fns = [fn for fn in fns if not fn in excluding_folder]
            fns = [go_dir + '/' + fn for fn in fns]
            feature_folders = [fn for fn in fns if isdir(fn)]
            for f_dir in feature_folders:
                if f_dir[-1] == '/':
                    edited_f_dir = f_dir[:-1]
                else:
                    edited_f_dir = f_dir
                write_submit_del_job(f_dir)

        write_submit_del_job(go_dir)




