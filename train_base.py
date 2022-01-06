'''
	Scripts to train base classifiers in a nested cross-validation structure by Weka.
	See README.md for detailed information.
	@author: Yan-Chak Li
'''
from os.path import isdir
from os import listdir
import argparse
from itertools import product
from os import system
from os.path import abspath, dirname, exists
from sys import argv
from common import load_properties, read_arff_to_pandas_df
from time import time



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    ### parse arguments
    parser = argparse.ArgumentParser(description='Feed some bsub parameters')
    parser.add_argument('--path', '-P', type=str, required=True, help='data path')
    parser.add_argument('--queue', '-Q', type=str, default='premium', help='LSF queue to submit the job')
    parser.add_argument('--node', '-N', type=str, default='16', help='number of node requested')
    parser.add_argument('--time', '-T', type=str, default='30:00', help='number of hours requested')
    parser.add_argument('--memory', '-M', type=str, default='40000', help='memory requsted in MB')
    parser.add_argument('--classpath', '-CP', type=str, default='./weka.jar', help='default weka path')
    parser.add_argument('--hpc', type=str2bool, default='true', help='use HPC cluster or not')
    parser.add_argument('--fold', '-F', type=int, default=5, help='number of cross-validation fold')
    parser.add_argument('--attr_imp', type=str2bool, default='False', help='getting attribute importance')
    args = parser.parse_args()
    ### record starting time
    start = time()
    ### get the data path
    data_path = abspath(args.path)
    data_source_dir = data_path.split('/')[-2]
    data_name = data_path.split('/')[-1]
    working_dir = dirname(abspath(argv[0]))

    ### get weka properties from weka.properties
    p = load_properties(data_path)
    bag_values = range(int(p['bagCount']))

    ### get the list of base classifiers
    classifiers_fn = data_path + '/classifiers.txt'
    assert exists(classifiers_fn)
    classifiers = filter(lambda x: not x.startswith('#'), open(classifiers_fn).readlines())
    classifiers = [_.strip() for _ in classifiers]

    ### get paths of the list of features
    fns = listdir(data_path)
    excluding_folder = ['analysis']
    fns = [fn for fn in fns if not fn in excluding_folder]
    fns = [fn for fn in fns if not 'tcca' in fn]
    fns = [data_path + '/' + fn for fn in fns]
    feature_folders = [fn for fn in fns if isdir(fn)]
    from functools import partial


    assert len(feature_folders) > 0

    # get fold, id and label attribute

    if 'foldAttribute' in p:
        df = read_arff_to_pandas_df(feature_folders[0] + '/data.arff')
        fold_values = list(df[p['foldAttribute']].unique())
        # pca_fold_values = ['pca_' + fv for fv in fold_values]
    else:
        fold_values = range(int(p['foldCount']))
    id_col = p['idAttribute']
    label_col = p['classAttribute']
    jobs_fn = "temp_train_base_{}_{}.jobs".format(data_source_dir, data_name)
    job_file = open(jobs_fn, 'w')
    if not args.hpc:
        job_file.write('module load groovy\n')


    def preprocessing(jf):
        classpath = args.classpath
        all_parameters = list(product(feature_folders, classifiers, fold_values, bag_values))

        for parameters in all_parameters:
            project_path, classifier, fold, bag = parameters
            jf.write('groovy -cp %s %s/base_predictors.groovy %s %s %s %s %s %s\n' % (
                classpath, working_dir, data_path, project_path, fold, bag, args.attr_imp, classifier))

        if not args.hpc:
            jf.write('python combine_individual_feature_preds.py %s %s\npython combine_feature_predicts.py %s %s\n' % (
                data_path, args.attr_imp, data_path, args.attr_imp))

        return jf

    job_file = preprocessing(job_file)
    job_file.close()

    ### submit to hpc if args.hpc != False
    if args.hpc:
        lsf_fn = 'run_%s_%s.lsf' % (data_source_dir, data_name)
        fn = open(lsf_fn, 'w')
        fn.write(
            '#!/bin/bash\n#BSUB -J EI-%s\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -n %s\n#BSUB -sp 100\n#BSUB -W %s\n#BSUB -o %s.stdout\n#BSUB -eo %s.stderr\n#BSUB -R rusage[mem=20000]\n' % (
            data_name, args.queue, args.node, args.time, data_source_dir, data_source_dir))
        # fn.write('module load python/2.7.14\n')
        # fn.write('module load py_packages\n')
        # fn.write('module purge')
        # fn.write('conda activate ei')
        fn.write('module load java\nmodule load python\nmodule load groovy\nmodule load selfsched\nmodule load weka\n')
        fn.write('export _JAVA_OPTIONS="-XX:ParallelGCThreads=10"\nexport JAVA_OPTS="-Xmx30g"\nexport CLASSPATH=%s\n' % (
            args.classpath))

        fn.write('mpirun selfsched < {}\n'.format(jobs_fn))
        fn.write('rm {}\n'.format(jobs_fn))
        fn.write('python combine_individual_feature_preds.py %s %s\npython combine_feature_predicts.py %s %s\n' % (
        data_path, args.attr_imp, data_path, args.attr_imp))
        fn.close()
        system('bsub < %s' % lsf_fn)
        system('rm %s' % lsf_fn)

    ### run it sequentially otherwise
    else:
        system('sh %s' % jobs_fn)
        system('rm %s' % jobs_fn)
    end = time()
    if not args.hpc:
        print('Elapsed time is: %s seconds' % (end - start))
