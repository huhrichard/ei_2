import os, fnmatch
import sys
from os import remove, system
from os.path import abspath

import glob
import argparse

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
parser.add_argument('--node', '-N', type=str, default='20', help='number of node requested')
parser.add_argument('--time', '-T', type=str, default='10:00', help='number of hours requested')
parser.add_argument('--memory', '-M', type=str,default='10240', help='memory requsted in MB')
parser.add_argument('--classpath', '-CP', type=str,default='./weka.jar', help='path to weka.jar')
parser.add_argument('--hpc', '-MIN', type=str2bool,default='true', help='use hpc cluster or not')
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



if __name__ == "__main__":
    # file_list = find_dir('GO*',sys.argv[-1])
    dir_list = find_dir('GO*', args.path)
    for go_dir in dir_list:
        data = go_dir.split('/')[-1]
        print('submitting largeGOPred ensemble job to hpc...')
        ####### Write the lsf file
        script = open(data + '.lsf', 'w')
        script.write(
            '#!/bin/bash\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -J %s\n#BSUB -W %s\n#BSUB -R rusage[mem=%s]\n#BSUB -n %s\n#BSUB -sp 100\n' % (
            args.queue, data, args.time, args.memory, args.node))
        script.write('#BSUB -o %s.%%J.stdout\n#BSUB -eo %s.%%J.stderr\n#BSUB -L /bin/bash\n' % (data, data))
        script.write('module purge')
        # script.write('conda activate largegopred')
        script.write(
        #     # 'module load python\n'+
        #     # 'module load py_packages\n'
            'module load java\nmodule load groovy\nmodule load python\nmodule load selfsched\nmodule load weka\n')
        script.write('export _JAVA_OPTIONS=\"-XX:ParallelGCThreads=10\"\nexport JAVA_OPTS=\"-Xmx10g\"\n')
        # script.write('mpirun selfsched < %s.jobs\n' % data)
        python_cmd = 'python ensemble.py --path {}\n'.format(go_dir)
        print(python_cmd)
        script.write(python_cmd)
        # script.write('rm %s.jobs' % data)
        script.close()
        ####### Submit the lsf job and remove lsf script
        system('bsub < %s.lsf' % data)
        remove('%s.lsf' % data)
