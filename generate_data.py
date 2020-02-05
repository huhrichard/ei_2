import pandas as pd
import datetime
import numpy as np
from os.path import exists,abspath
from os import mkdir
from sys import argv
from glob import glob
from multiprocessing import Pool
from itertools import product
import arff

def convert_to_arff(df,path):
    fn = open(path, 'w')
    fn.write('@relation yanchakli\n')
    for col in df.columns:
        if col not in ['cls','seqID']:
            fn.write('@attribute %s numeric\n' %col)
        elif col == 'cls':
            fn.write('@attribute cls {pos,neg}\n')

	elif col == 'seqID':
		fn.write('@attribute seqID string\n')
    fn.write('@data\n')
    cont = df.to_csv(index=False,header=None)
    fn.write(cont)
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

def processTermFeature_2(param):
	term, feature, go_hpo, csv_file = param
	feature_df = pd.read_csv('{}{}.csv'.format(csv_file, feature), index_col=0)
	before_shape = feature_df.shape
	go_hpo = go_hpo[go_hpo != 0]
	term_inds = go_hpo.index.tolist()
	sel_inds = [ind for feature_df.index.tolist() if ind in term_inds]
	feature_df = feature_df.loc[sel_inds,]
	go_hpo = go_hpo.loc[sel_inds]
	cols = (feature_df == 0).all(axis=0)
	cols  = cols.loc[cols == False].index.tolist()
	feature_df = feature_df[cols]

	print term, feature, before_shape, feature_df.shape
	labs = []
	for l in go_hpo:
		if l == -1:
			labs.append('neg')
		elif l == 1:
			labs.append('pos')
		else:
			print 'invalid labels'
			exit(0)
	feature_df = feature_df.round(3)
	feature_df['cls'] = labs
	del feature_df.index.name
	p = '%s/%s' % (scratch_data_dir, t)
	if not exists(p):
		mkdir(p)
	path = '%s/%s.arff' % (p, t)
	convert_to_arff(feature_df, path)

if __name__== "__main__":

	scratch_data_dir = abspath('/sc/hydra/scratch/liy42/')
	group_number_goterm = argv[2]
	scratch_data_dir = scratch_data_dir + group_number_goterm
	csv_dir = './not_on_github/csv/'
	tsv_dir = './not_on_github/tsv/'
	go_to_hpo_file = 'GO2HPO_binary.tsv'
	if not exists(scratch_data_dir):
		mkdir(scratch_data_dir)
	features = ['coexpression','cooccurence','database','experimental','fusion','neighborhood']
	term = argv[1]
	t = term[:2] + term[3:]

	if not exists(scratch_data_dir + '/' + t):
		mkdir(scratch_data_dir + '/' + t)

	for feature in features:
	    if not exists(scratch_data_dir + '/' + t + '/' + feature):
	        mkdir(scratch_data_dir + '/' + t + '/' + feature)
	
	# deepNF_net = pd.read_csv('/sc/hydra/scratch/liy42/deepNF/%s/%s.arff' %(t,t), header=None,comment='@')
	# seqs = deepNF_net.iloc[:,-1].tolist()
	# labels = deepNF_net.iloc[:,-2].tolist()
	go_to_hpo_df = pd.read_csv(tsv_dir+go_to_hpo_file, sep='\t', index_col=0)
	go_to_hpo_df_with_specific_term = go_to_hpo_df[[term]]


	params = list(product([term], features, [go_to_hpo_df_with_specific_term], [csv_dir]))
	print '[STARTED: %s] start multithreads computing to generate feature files for GO term: %s...............................' %(str(datetime.datetime.now()), term)
	p = Pool(6)
	p.map(processTermFeature_2,params)
	print '[FINISHED: %s] completed the generation of feature files for GO term: %s...........................................' %(str(datetime.datetime.now()), term)

