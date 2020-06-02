'''
Combine predictions of models using different feature sets.
Author: Linhua (Alex) Wang
Date:  12/27/2018
'''
from os.path import exists,abspath,isdir,dirname
from sys import argv
from os import listdir,environ
from common import load_properties, load_arff_headers
import pandas as pd
import numpy as np

data_folder = abspath(argv[1])

fns = listdir(data_folder)
fns = [fn for fn in fns if fn != 'analysis']
fns = [data_folder  + '/' + fn for fn in fns]
feature_folders = [fn for fn in fns if isdir(fn)]

# foldValues = range(int(argv[2]))
p = load_properties(data_folder)
# fold_count = int(p['foldCount'])
if 'foldAttribute' in p:
	input_fn = '%s/%s' % (feature_folders[0], 'data.arff')
	assert exists(input_fn)
	headers = load_arff_headers(input_fn)
	fold_values = headers[p['foldAttribute']]
else:
	fold_values = range(int(p['foldCount']))


prediction_dfs = []
validation_dfs = []

# for value in foldValues:
for value in fold_values:
	prediction_dfs = []
	validation_dfs = []
	for folder in feature_folders:
		feature_name = folder.split('/')[-1]
		prediction_df = pd.read_csv(folder + '/predictions-%s.csv.gz' %value,compression='gzip')
		prediction_df.set_index(['id','label'],inplace=True)
		prediction_df.columns = ['%s.%s' %(feature_name,col) for col in prediction_df.columns]

		validation_df = pd.read_csv(folder + '/validation-%s.csv.gz' %value,compression='gzip')
		validation_df.set_index(['id','label'],inplace=True)
		validation_df.columns = ['%s.%s' %(feature_name,col) for col in validation_df.columns]

		prediction_dfs.append(prediction_df)
		validation_dfs.append(validation_df)

	prediction_dfs = pd.concat(prediction_dfs,axis=1)
	validation_dfs = pd.concat(validation_dfs,axis=1)

	prediction_dfs.to_csv(data_folder + '/predictions-%s.csv.gz' %value,compression='gzip')
	validation_dfs.to_csv(data_folder + '/validation-%s.csv.gz' %value,compression='gzip')
