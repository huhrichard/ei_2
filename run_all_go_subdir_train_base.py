import os, fnmatch
import sys
from os import remove, system
from os.path import abspath

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

if __name__ == "__main__":
    dir_list = find_dir('HP*', sys.argv[-1])
    # dir_list = find_dir('GO0071704', sys.argv[-1])
    for go_dir in dir_list:
        python_cmd = 'python train_base.py --path {}'.format(go_dir)
        print(python_cmd)
        system(python_cmd)
