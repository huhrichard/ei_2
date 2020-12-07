import os


outcome_list = ['on_off',
                 'dyskinesia',
                 'tremor']

base_path = './not_on_github/beat_pd_challenge/processed_data_for_EI'

for o in outcome_list:
    python_cmd = "python ensemble.py --regression True --path {}"
    o_path = os.path.join(base_path, o)
    os.system(python_cmd.format(o_path))