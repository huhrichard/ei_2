'''
    Scripts to train base classifiers in a nested cross-validation structure.
    See README.md for detailed information.
    @author: Linhua Wang
'''
from os.path import abspath, isdir
from os import remove, system, listdir
import argparse
from common import load_properties
from itertools import product
from os import environ, system
from os.path import abspath, dirname, exists
from sys import argv
from common import load_arff_headers, load_properties
from time import time
from scipy.io import arff
import pandas as pd
import tcca_projection
import numpy as np
import generate_data
from sklearn.decomposition import PCA

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_arff_to_pandas_df(arff_path):
    # loadarff doesn't support string attribute
    # data = arff.loadarff(arff_path)
    df = pd.read_csv(arff_path, comment='@')
    num_col = df.shape[1]
    columns = []
    file1 = open(arff_path, 'r')
    Lines = file1.readlines()

    count = 0
    # Strips the newline character
    for line_idx, line in enumerate(Lines):
        # if line_idx > num_col
        if '@attribute' in line:
            columns.append(line.strip().split(' ')[1])

    df.columns = columns
    return df


### parse arguments
parser = argparse.ArgumentParser(description='Feed some bsub parameters')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--queue', '-Q', type=str, default='premium', help='LSF queue to submit the job')
parser.add_argument('--node', '-N', type=str, default='4', help='number of node requested')
parser.add_argument('--time', '-T', type=str, default='20:00', help='number of hours requested')
parser.add_argument('--memory', '-M', type=str,default='20000', help='memory requsted in MB')
parser.add_argument('--classpath', '-CP', type=str,default='./weka.jar', help='default weka path')
parser.add_argument('--hpc', type=str2bool,default='true', help='use HPC cluster or not')
parser.add_argument('--fold', '-F', type=int, default=5, help='number of cross-validation fold')
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
fold_values = range(int(p['foldCount']))
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
fns = [data_path  + '/' + fn for fn in fns]
feature_folders = [fn for fn in fns if isdir(fn)]
assert len(feature_folders) > 0

### TODO: Read OuterCV and perform TCCA here?

fold_list = np.array(range(args.fold))+10000
id_col = p['idAttribute']
label_col = p['classAttribute']
arff_list = [read_arff_to_pandas_df(f_path+'/data.arff') for f_path in feature_folders]
data_path_list = [f_path+'/data_{}.arff' for f_path in feature_folders]

print(arff_list[0].shape)
print(arff_list[0]['fold'])
import torch
import tensorly as tl
context_dict = {}
# if torch.cuda.is_available():
#     # print('using pytorch as TL backend')
#     # tl.set_backend('pytorch')
#     # tl_pytorch = True
#     # context_dict['device'] = torch.device('cuda')
# else:
tl_pytorch = False

rdim = 10

if ('foldAttribute' in p) and (len(feature_folders) > 1):
    fold_col = p['foldAttribute']
    column_non_feature = [fold_col, id_col, label_col]
    for outer_fold in fold_list:
        test_split_list = [df[df[fold_col]==outer_fold] for df in arff_list]
        train_split_list = [df[df[fold_col] != outer_fold] for df in arff_list]
        test_nf = test_split_list[0][column_non_feature]
        train_nf = train_split_list[0][column_non_feature]
        test_X_raw = [tl.tensor(t.drop(columns=column_non_feature).values,
                                **context_dict) for t in test_split_list]
        train_X_raw = [tl.tensor(t.drop(columns=column_non_feature).values,
                                 **context_dict) for t in train_split_list]
        # print(train_split_list[0].columns, train_split_list[0].shape)
        # print(train_split_list[0].columns, train_split_list[0].shape)
        # print(len(train_X_raw), train_X_raw[0].shape)
        pca_list = [PCA(n_components=min([10, df.shape[0]]),
                        svd_solver='arpack', random_state=64) for df in arff_list]
        Z_train = []
        Z_test = []

        for idx, pca_obj in enumerate(pca_list):
            Z_train_n = pca_obj.fit_transform(train_X_raw[idx])
            Z_test_n = pca_obj.transform(test_X_raw[idx])
            Z_train.append(Z_train_n)
            Z_test.append(Z_test_n)

        # H_train, Z_train = tcca_projection.project(train_X_raw, rDim=rdim)
        # Z_test = []
        # for v in range(len(H_train)):
        #     if tl_pytorch:
        #         Z_test.append(torch.matmul(test_X_raw[v], H_train[v]))
        #     else:
        #         Z_test.append(np.matmul(test_X_raw[v], H_train[v]))

        feat_col = ['ProjectedFeature{}'.format(i) for i in range(rdim)]
        projected_train_df_list = [pd.DataFrame(data=z_t,
                                                columns=feat_col) for z_t in Z_train]
        projected_train_df_list = [pd.concat([df.reset_index(drop=True), train_nf.reset_index(drop=True)],
                                                axis=1) for df in projected_train_df_list]

        projected_test_df_list = [pd.DataFrame(data=z_t,
                                                columns=feat_col) for z_t in Z_test]
        projected_test_df_list = [pd.concat([df.reset_index(drop=True), test_nf.reset_index(drop=True)],
                                                axis=1) for df in projected_test_df_list]

        projected_df_with_nf = [pd.concat([test_df,
                                           train_df], ignore_index=True) for test_df, train_df in zip(projected_test_df_list, projected_train_df_list)]

        # final_columns_list = projected_df_with_nf.columns

        arff_fn_list = [f_path+'/data_pca_fold_{}.arff'.format(outer_fold) for f_path in feature_folders]
        for v_fn, projected_df in zip(arff_fn_list, projected_df_with_nf):
            # projected_df.columns = feat_col + []
            print(projected_df.columns)
            generate_data.convert_to_arff(projected_df, v_fn)



### write the individual tasks
classpath = args.classpath
all_parameters = list(product(feature_folders, classifiers,fold_values,bag_values))




jobs_fn = "temp_{}_{}.jobs".format(data_source_dir, data_name)
job_file = open(jobs_fn,'w')
for parameters in all_parameters:
    project_path, classifier, fold, bag = parameters
    # job_file.write('groovy -cp %s %s/base_model.groovy %s %s %s %s %s\n' % (classpath, working_dir,data_path, project_path, fold, bag,classifier))
    tcca_bool = True
    job_file.write('groovy -cp %s %s/base_predictors_enable_tcca.groovy %s %s %s %s %s %s\n' % (classpath, working_dir,data_path, project_path, fold, bag, tcca_bool, classifier))
if not args.hpc:
    job_file.write('python combine_individual_feature_preds.py %s\npython combine_feature_predicts.py %s %s\n' %(data_path,data_path,args.fold))
job_file.close()

### submit to hpc if args.hpc != False
if args.hpc:
    lsf_fn = 'run_%s_%s.lsf' %(data_source_dir, data_name)
    fn = open(lsf_fn,'w')
    fn.write('#!/bin/bash\n#BSUB -J EI-%s\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -n %s\n#BSUB -sp 100\n#BSUB -W %s\n#BSUB -o %s.stdout\n#BSUB -eo %s.stderr\n#BSUB -R rusage[mem=20000]\n' %(data_name,args.queue,args.node,args.time,data_name,data_name))
    # fn.write('module load python/2.7.14\n')
    # fn.write('module load py_packages\n')
    fn.write('module purge')
    # fn.write('conda activate ei')
    fn.write('module load java\nmodule load python\nmodule load groovy\nmodule load selfsched\nmodule load weka\n')
    fn.write('export _JAVA_OPTIONS="-XX:ParallelGCThreads=10"\nexport JAVA_OPTS="-Xmx15g"\nexport CLASSPATH=%s\n' %(args.classpath))

    fn.write('mpirun selfsched < {}\n'.format(jobs_fn))
    fn.write('rm {}\n'.format(jobs_fn))
    fn.write('python combine_individual_feature_preds.py %s\npython combine_feature_predicts.py %s %s\n' %(data_path,data_path,args.fold))
    fn.close()
    system('bsub < %s' %lsf_fn)
    system('rm %s' %lsf_fn)

### run it sequentially otherwise
else:
    system('sh %s.jobs' %data_name)
    system('rm %s.jobs' %data_name)
end = time()
if not args.hpc:
    print('Elapsed time is: %s seconds' %(end - start))
