import os, fnmatch
import sys
from os import remove, system
from os.path import abspath
from os.path import abspath, isdir
from os import remove, system, listdir
import glob
import argparse
import subprocess
sys.path.append('./')
import common

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
parser.add_argument('--time', '-T', type=str, default='20:00', help='number of hours requested')
parser.add_argument('--memory', '-M', type=str,default='5000', help='memory requsted in MB')
parser.add_argument('--classpath', '-CP', type=str,default='./weka.jar', help='path to weka.jar')
parser.add_argument('--hpc', '-MIN', type=str2bool,default='true', help='use hpc cluster or not')
parser.add_argument('--term_prefix', type=str, default='GO', help='term_prefix')
parser.add_argument('--rank', type=str2bool, default='false', help='attribute importance')
parser.add_argument('--by_jobf', type=str2bool, default='True', help='submit jobs by the .jobs file')
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

def write_submit_del_job(ensemble_dir, jobs_fn):
    second_sub = ensemble_dir.split('/')[-1]
    first_sub = ensemble_dir.split('/')[-2]


    lsf_fn = first_sub + '_' + second_sub + '.lsf'
    print(lsf_fn)
    print('submitting EI ensemble job to hpc...')
    ####### Write the lsf fileqn1
    script = open(lsf_fn, 'w')
    script.write(
        '#!/bin/bash\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -J %s\n#BSUB -W %s\n#BSUB -R rusage[mem=%s]\n#BSUB -n %s\n#BSUB -sp 100\n' % (
            args.queue, second_sub, args.time, args.memory, args.node))
    script.write('#BSUB -o ensemble_%s.stdout\n#BSUB -eo ensemble_%s.stderr\n#BSUB -L /bin/bash\n' % (second_sub, second_sub))
    # script.write('module purge')
    # script.write('conda activate largegopred')
    # script.write(
        #     # 'module load python\n'+
        #     # 'module load py_packages\n'
        # 'module load java\nmodule load groovy\nmodule load selfsched\nmodule load weka\n')
    # script.write('export _JAVA_OPTIONS=\"-XX:ParallelGCThreads=10\"\nexport JAVA_OPTS=\"-Xmx10g\"\n')
    script.write('mpirun selfsched < %s' % jobs_fn)
    # python_cmd = "\n".join(python_cmd_list)
    # print(python_cmd)
    # script.write(python_cmd)
    # script.write('rm %s.jobs' % second_sub)
    script.close()
    ####### Submit the lsf job and remove lsf script
    system('bsub < %s' % lsf_fn)
    remove(lsf_fn)

excluding_folder = ['analysis']

if __name__ == "__main__":
    # file_list = find_dir('GO*',sys.argv[-1])
    dir_list = find_dir('{}*'.format(args.term_prefix), args.path)

    jobs_prefix = args.path.split('/')[-1]

    scratch_path = '/sc/arion/scratch/liy42/'
    g_jobs_file = find_dir('{}.jobs'.format(jobs_prefix), './jobs')
    cat_dir = args.path
    jobs_fstream = os.listdir(cat_dir)


    jobs_n = 'ensemble_{}.jobs'.format(jobs_prefix)
    jobs_txt = open(jobs_n, 'w')
    jobs_list = []
    python_cmd_list = []
    if args.by_jobf is True:
        g_jobs_fstream = open(g_jobs_file[0], "r").read().split('\n')
        for go_job in g_jobs_fstream:
            if len(go_job.split(' ')) > 1:
                # term_name = go_job.split(' ')[2].replace(':', '')
                term_name = common.load_cmd_option(go_job, 'outcome').replace(':', '')
                go_scratch_dir = scratch_path+jobs_prefix+'/'+term_name

                        # data = go_scratch_dir.split('/')[-1]
                data_dir = go_scratch_dir.split('/')[-2]
                if data_dir.split('_')[-1] == 'EI':
                    # p = subprocess.Popen('python tcca_projection.py --path {}'.format(go_dir))
                    # p.wait()
                    # python_cmd_list.append('python tcca_projection.py --path {}'.format(go_dir))
                    fns = listdir(go_scratch_dir)
                    fns = [fn for fn in fns if not fn in excluding_folder]
                    fns = [os.path.join(go_scratch_dir, fn) for fn in fns]
                    feature_folders = [fn for fn in fns if isdir(fn)]
                    for f_dir in feature_folders:
                        jobs_list.append('python ensemble.py --path {}'.format(f_dir))

                if args.rank is False:
                    jobs_list.append('python ensemble.py --path {}'.format(go_scratch_dir))
                else:
                    jobs_list.append('python ensemble.py --path {}'.format(go_scratch_dir) + ' --rank True')
            else:
                break
        jobs_txt.write('\n'.join(jobs_list))
        jobs_txt.close()
        write_submit_del_job(args.path, jobs_n)
    else:
        # g_jobs_fstream = open(g_jobs_file[0], "r").read().split('\n')
        # for go_job in g_jobs_fstream:
        for go_job in os.walk(cat_dir, topdown=True):
            # print(go_job)
            root, dirs, files = go_job
            num_sep = cat_dir.count(os.path.sep)
            num_sep_this = root.count(os.path.sep)
            # print(root)
            # print(cat_dir)
            if root == cat_dir:
                # print(dirs)
                for dir in dirs:
                    if (not 'analysis' in dir) and (not 'weka.' in dir):
                        # print(dir)
                        go_scratch_dir = os.path.join(root, dir)
                        if go_scratch_dir != cat_dir:
                            print(go_scratch_dir)
                            data = go_scratch_dir.split('/')[-1]
                            data_dir = go_scratch_dir.split('/')[-2]
                            if data_dir.split('_')[-1] == 'EI':
                                # p = subprocess.Popen('python tcca_projection.py --path {}'.format(go_dir))
                                # p.wait()
                                # python_cmd_list.append('python tcca_projection.py --path {}'.format(go_dir))
                                fns = listdir(go_scratch_dir)
                                fns = [fn for fn in fns if not fn in excluding_folder]
                                fns = [os.path.join(go_scratch_dir, fn) for fn in fns]
                                feature_folders = [fn for fn in fns if isdir(fn)]
                                for f_dir in feature_folders:
                                    jobs_list.append('python ensemble.py --path {}'.format(f_dir))

                            if args.rank is False:
                                jobs_list.append('python ensemble.py --path {}'.format(go_scratch_dir))
                            else:
                                jobs_list.append('python ensemble.py --path {}'.format(go_scratch_dir) + ' --rank True')
            else:
                break
        jobs_txt.write('\n'.join(jobs_list))
        jobs_txt.close()
        write_submit_del_job(args.path, jobs_n)




