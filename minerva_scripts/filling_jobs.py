import pandas as pd
import sys
import argparse
import os

# term = sys.argv[-2]
# annotation = sys.argv[-1]
parser = argparse.ArgumentParser(description='')
parser.add_argument('--prepro_path', type=str, required=True, help='data path')
parser.add_argument('--go_tsv', type=str, required=True, help='')
parser.add_argument('--ontology', type=str, default='go', help='')
parser.add_argument('--annotation', type=str, required=True, help='')



args = parser.parse_args()
annotation_understroke = args.annotation.replace('-', '_')
term = args.ontology
cd_df = pd.read_csv('./plot/cd_csv/cd_input_{}_{}_fmax.csv'.format(term, args.annotation))
data_list = ['EI',
             # 'mashup', 'deepNF'
             ]
path = os.path.join(args.prepro_path, args.go_tsv)

for data in data_list:
    fn = 'EIdata_{}_{}_{}.jobs'.format(annotation_understroke, term, data)
    jobs_fn = './jobs/'+fn
    f = open(jobs_fn, 'w')

    missed_term = cd_df[cd_df.isnull().any(axis=1)].index
    print(missed_term)
    go_group_dir = os.path.join('/sc/arion/scratch/liy42/', fn.split('.')[0])
    cmd_str = 'python minerva_scripts/generate_data.py --outcome {} --output_dir {} --method {} --feature_csv_path {} --outcome_tsv_path {}\n'
    for t in missed_term:
        f.write(cmd_str.format(t, go_group_dir, data, args.prepro_path, path))
    f.close()