import pandas as pd
import sys

term = sys.argv[-2]
annotation = sys.argv[-1]

cd_df = pd.read_csv('./plot/cd_csv/cd_input_{}_{}.csv'.format(term, annotation))

annotation_understroke = annotation.replace('-', '_')

data = 'EI'

fn = 'EIdata_{}_{}_{}.jobs'.format(annotation_understroke, term, data)
jobs_fn = './jobs/'+fn
f = open(fn, 'w')

missed_term = cd_df.isnull().any(axis=1).index
cmd_str = 'python generate_data.py {} {}/ {}\n'
for t in missed_term:
    f.write(cmd_str.format(t, fn.split('.')[0], data))
f.close()