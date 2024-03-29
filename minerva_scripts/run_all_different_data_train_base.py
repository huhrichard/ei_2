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
            # result_dir = abspath(os.path.join(path, dir))
            result_dir = dir
            print(result_dir)
            result.append(result_dir)

    return result


if __name__ == "__main__":
    ontology = sys.argv[-1]
    dir_list = find_dir('EIdata*_{}*.jobs'.format(ontology), './jobs/')
    scratch_path = '/sc/arion/scratch/liy42/'
    # dir_list = find_dir('GO0071704', sys.argv[-1])
    if 'go' in ontology:
        prefix = 'GO'
    elif 'hpo' in ontology:
        prefix = 'HP'


    for go_dir in dir_list:

        # python_cmd = 'python train_base.py --path {}'.format(go_dir)
        python_cmd = 'python minerva_scripts/run_all_go_subdir_train_base.py --term_prefix {} --path {}'.format(prefix, scratch_path+go_dir.split('.')[0])
        print(python_cmd)
        system(python_cmd)