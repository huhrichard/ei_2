import pandas as pd
import datetime
import numpy as np
import os
from os.path import exists, abspath, isdir
from os import mkdir
from sys import argv
from glob import glob
from multiprocessing import Pool
from itertools import product
import arff
from soft_impute import SoftImpute



def convert_to_arff(df, path):
    fn = open(path, 'w')
    fn.write('@relation yanchakli\n')
    col_counter = 0
    for col in df.columns:
        if col not in ['cls', 'seqID']:
            fn.write('@attribute %s numeric\n' % col)
            col_counter += 1
        elif col == 'cls':
            fn.write('@attribute cls {pos,neg}\n')
            col_counter += 1
        elif col == 'seqID':
            fn.write('@attribute seqID string\n')
            col_counter += 1
    print('col counter:', col_counter)

    fn.write('@data\n')
    print(path, 'start to write df')
    cont = df.to_csv(index=False, header=None)
    fn.write(cont)
    print(path, 'finished writing df')
    fn.close()


# def processTermFeature(param):
# 	term, feature, labels = param
# 	feature_df =  pd.read_csv('%s.csv' %feature,index_col=0)
# 	before_shape = feature_df.shape
# 	cols = [ c for c in feature_df.columns if c in seqs]
# 	feature_df = feature_df.loc[seqs,cols]
# 	feature_df.fillna(0,inplace=True)
# 	feature_df = feature_df.loc[:,(feature_df !=0).any(axis=0)]
# 	feature_df = feature_df.round(3)
# 	feature_df['cls'] = labels
# 	del feature_df.index.name
# 	feature_df['seqID'] = feature_df.index
#
#         p = '%s/%s/%s' %(scratch_data_dir, t, feature)
#
#         if not exists(p):
#             mkdir(p)
#
#         path = p + '/' + t + '.arff'
# #	arff.dump(path, feature_df.values, relation='linhuaw', names = feature_df.columns)
#         convert_to_arff(feature_df,path)

def processTermFeature_3(param, impute):
    term, feature, go_hpo_df, csv_file = param
    feature_df = pd.read_csv('{}{}.csv'.format(csv_file, feature), index_col=0)
    before_shape = feature_df.shape
    # go_hpo_df.fillna(0, inplace=True)
    go_hpo_df = go_hpo_df[go_hpo_df != 0]
    go_hpo_df.replace(-1, 'neg', inplace=True)
    go_hpo_df.replace(1, 'pos', inplace=True)

    if impute:
        # from fancyimpute import SoftImpute
        imp = SoftImpute()

        f = feature_df.values
        imp.fit(f)
        imputed_f = imp.predict(f)
        # imputed_f = imp.transform(f)
        feature_df[:] = imputed_f
    else:
        feature_df.fillna(0, inplace=True)
    cols = (feature_df == 0).all(axis=0)
    cols = cols.loc[cols == False].index.tolist()
    feature_df = feature_df[cols]
    feature_df = feature_df.round(3)
    # merged_df = pd.merge(feature_df, go_hpo_df, how='inner')
    merged_df = pd.concat([feature_df, go_hpo_df], axis=1, join='inner')
    merged_df.rename(columns={term: 'cls'}, inplace=True)
    merged_df['seqID'] = merged_df.index
    print('before', merged_df.shape)
    merged_df.dropna(inplace=True)
    print('after', merged_df.shape)
    del merged_df.index.name
    # print(term, 'merged df')
    p = os.path.join(scratch_data_dir, feature)
    if not exists(p):
        mkdir(p)
    path = os.path.join(p, 'data.arff')
    convert_to_arff(merged_df, path)



# def processTermFeature_2(param):
#     term, feature, go_hpo_df, csv_file = param
#     feature_df = pd.read_csv('{}{}.csv'.format(csv_file, feature), index_col=0)
#     before_shape = feature_df.shape
#     go_hpo_df.fillna(0, inplace=True)
#     print('before', go_hpo_df.shape)
#     go_hpo_df = go_hpo_df[go_hpo_df != 0]
#     print('after', go_hpo_df.shape)
#     term_inds = go_hpo_df.index.tolist()
#     sel_inds = [ind for ind in feature_df.index.tolist() if ind in term_inds]
#     feature_df.fillna(0, inplace=True)
#     feature_df = feature_df.loc[sel_inds,]
#     go_hpo_df = go_hpo_df.loc[sel_inds]
#
#
#     print(term, feature, before_shape, feature_df.shape)
#     labs = []
#     print(go_hpo_df)
#     # print(feature_df)
#
#
#     counter = 0
#     for l in go_hpo_df:
#
#         print(counter, l)
#         counter += 1
#
#         if l == -1:
#             labs.append('neg')
#         elif l == 1:
#             labs.append('pos')
#
#         else:
#             print('invalid labels', l)
#             exit(0)
#
#     feature_df = feature_df.round(3)
#     feature_df['cls'] = labs
#     del feature_df.index.name
#     p = os.path.join(scratch_data_dir, feature)
#     if not exists(p):
#         mkdir(p)
#     path = os.path.join(p, t+'.arff')
#     convert_to_arff(feature_df, path)


if __name__ == "__main__":

    scratch_data_dir = '/sc/hydra/scratch/liy42/'
    group_number_goterm = argv[2]

    if 'Impute' in group_number_goterm:
        impute_graph = True
    else:
        impute_graph = False

    csv_dir = './not_on_github/csv/'
    tsv_dir = './not_on_github/tsv/'
    go_to_hpo_file = 'GO2HPO_binary.tsv'
    # go_to_hpo_file = 'pos-neg-O-10.tsv'
    print(len(argv))
    if len(argv) == 4:
        features = [argv[3]]
    else:
        features = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']
    # features = ['deepNF']

    term = argv[1]

    t = term.split(':')[0] + term.split(':')[1]
    scratch_data_dir = scratch_data_dir + group_number_goterm

    if not exists(scratch_data_dir):
        mkdir(scratch_data_dir)

    scratch_data_dir = os.path.join(scratch_data_dir, t+'/')


    if not exists(scratch_data_dir):
        mkdir(scratch_data_dir)

    os.system('cp sample_data/classifiers.txt {}'.format(scratch_data_dir[:-1]))
    os.system('cp sample_data/weka.properties {}'.format(scratch_data_dir[:-1]))

    for feature in features:
        f_dir = os.path.join(scratch_data_dir, feature+'/')
        if not exists(f_dir):
            mkdir(f_dir)

    # deepNF_net = pd.read_csv('/sc/hydra/scratch/liy42/deepNF/%s/%s.arff' %(t,t), header=None,comment='@')
    # seqs = deepNF_net.iloc[:,-1].tolist()
    # labels = deepNF_net.iloc[:,-2].tolist()
    go_to_hpo_df = pd.read_csv(tsv_dir + go_to_hpo_file, sep='\t', index_col=0)
    print(term)
    go_to_hpo_df_with_specific_term = go_to_hpo_df[[term]]


    params = list(product([term], features, [go_to_hpo_df_with_specific_term], [csv_dir]))
    print(
        '[STARTED: %s] start multithreads computing to generate feature files for GO term: %s...............................' % (
        str(datetime.datetime.now()), term))
    for param in params:
        processTermFeature_3(param, impute=impute_graph)

    # p = Pool(6)

    # p.map(processTermFeature_3, params)
    print(
        '[FINISHED: %s] completed the generation of feature files for GO term: %s...........................................' % (
        str(datetime.datetime.now()), term))
