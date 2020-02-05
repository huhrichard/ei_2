import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = './not_on_github/tsv/GO2HPO_binary.tsv'
# path = 'GO_annotations_Sept2017_EI_experiments.tsv'
df = pd.read_csv(path, sep='\t',index_col=0)
# print(df.shape)
# print(df.df.column
# print(df.shape)
# np_array = df.values
pos_entry = (df == 1).values
go_terms = df.columns
go_pos_count = sum(pos_entry)
# plt.figure()
# plt.hist(go_pos_count, bins=100)
# plt.show()

go_by_count_dict = {'EIdata_500_1000_go.jobs':np.logical_and((go_pos_count>500), (go_pos_count<=1000)),
                    'EIdata_1000_go.jobs': go_pos_count > 1000,
                    'EIdata_200_500_go.jobs': np.logical_and((go_pos_count>200), (go_pos_count<500))}
# print(pos_count)

# go_annotated_500_to_1000_jobs_fn = 'EIdata_500_1000_go.jobs'
# go_500_to_1000_count = np.logical_and((pos_count>500), (pos_count<=1000))
#
# go_annotated_more_1000_jobs_fn = 'EIdata_1000_go.jobs'
# go_1000_count = pos_count > 1000
#
# go_annotated_200_to_500_jobs_fn = 'EIdata_200_500_go.jobs'
# go_200_to_500_count = np.logical_and((pos_count>200), (pos_count<500))

for fn, bool_array in go_by_count_dict.items():
    f = open(fn, 'w')

    print(fn, np.sum(bool_array))
    go_by_groups = go_terms[bool_array]
    for go in go_by_groups:
        f.write('python generate_data.py {}\n'.format(go))

    f.close()