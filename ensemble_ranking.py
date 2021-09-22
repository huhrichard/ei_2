import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from os.path import abspath, exists
import numpy as np
import os

parser = argparse.ArgumentParser(description='')
# parser.add_argument('--bppath', '-bpp', type=str, required=True, help='data path of importance of base predictors')
# parser.add_argument('--bppath', '-bpp', type=str, required=True, help='data path of importance of base predictors')
parser.add_argument('--path', '-p', type=str, required=True, help='data path of importance of base features')
parser.add_argument('--stacker', '-s', type=str, default='none', help='stacker which performs the best')
# parser.add_argument('--outcome', '-o', type=str, default='none', help='data path of importance of base features')
# parser.add_argument('--perfdf', '-pdf', type=str, default='none', help='data path of importance of base features')

args = parser.parse_args()
# bppath = abspath(args.bppath)
# bfpath = abspath(args.bfpath)
bfpath = os.path.join(args.path, 'attribute_imp-1.csv.gz')
bppath = os.path.join(args.path, 'analysis/pi_stackers.csv')
stacker = args.stacker
# if args.stacker == 'none':
#     if args.outcome != 'none':
#         perf_df = abspath(args.perfdf)
#         outcome = args.outcome
#     else:
#         print('There is no specified stacker/outcome')

imp_base_predictors = pd.read_csv(bppath)
imp_base_predictors = imp_base_predictors.loc[imp_base_predictors['stacker'] == stacker]
imp_base_predictors.drop(columns=['stacker'], inplace=True)
imp_base_predictors = imp_base_predictors.T
imp_base_predictors.rename(columns={imp_base_predictors.columns[0]: 'bp_imp'}, inplace=True)
imp_base_predictors = imp_base_predictors.iloc[1:]
imp_base_predictors['bp_name'] = imp_base_predictors.index
print(imp_base_predictors)
imp_base_features = pd.read_csv(bfpath, compression = 'gzip')



bpdf_imp_col = 'bp_imp'
bfdf_imp_col = 'attribute_importance'
bfdf_feature_col = 'attribute'
bp_name_col_bpdf = 'bp_name'
bp_name_col_bfdf = 'base_predictor'
# classifier,modality
imp_base_features['bag'] = '0'
imp_base_features[bp_name_col_bfdf] = imp_base_features[['modality','classifier','bag']].agg('-'.join, axis=1)

multiplied_rank_col = 'bfxbp_rank'

imp_base_predictors['bp_rank'] = imp_base_predictors[bpdf_imp_col].rank(pct=True)
imp_base_features['bf_rank'] = imp_base_features[bfdf_imp_col].rank(pct=True)
imp_base_features[multiplied_rank_col] = 0.0

for bp_idx, bp in imp_base_predictors.iterrows():
    bp_name = bp[bp_name_col_bpdf]
    bp_rank = bp['bp_rank']
    bf_df_matched_bool = imp_base_features[bp_name_col_bfdf] == bp_name
    bf_ranks = imp_base_features.loc[bf_df_matched_bool, 'bf_rank']
    imp_base_features.loc[bf_df_matched_bool, multiplied_rank_col] = bf_ranks*bp_rank

base_features_list = imp_base_features[bfdf_feature_col].unique().tolist()
base_feature_rank_agg = {}

for base_feature in base_features_list:
    ranks = imp_base_features.loc[imp_base_features[bfdf_feature_col] == base_feature, multiplied_rank_col]
    avg_ranks = np.mean(list(ranks))
    base_feature_rank_agg[base_feature] = avg_ranks

base_feature_rank_df = pd.DataFrame(base_feature_rank_agg).T
base_feature_rank_df.to_csv('base_feature_rank.csv')






