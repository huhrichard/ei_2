
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

templates = {'on_off': "syn21344933",
              'dyskinesia': "syn21344934",
              'tremor' :"syn21344949"}

measurement_id_col = 'measurement_id'
subject_id_col = 'subject_id'

"""
Change the path if needed
"""
measurement_subject_path = 'processed_data/measurement_subject_table.csv'
measurement_subject_df = pd.read_csv(measurement_subject_path)
# print(measurement_subject_df)

def weighted_mse(groundtruth_df, pred_df):
    """

    :param pred_df: dataframe of predictions
    :param groundtruth_df: dataframe of groundtruth
    :return: weighted_mse_score
    """

    m_index = pred_df.index
    """
    Checking the subject id by the measurement id
    """
    subset_ms_df = measurement_subject_df.loc[measurement_subject_df[measurement_id_col].isin(m_index)]
    m_count_per_subject = subset_ms_df[subject_id_col].value_counts()
    subject_scores = []
    weights = []

    for subject_id, m_count in m_count_per_subject.items():
        """
        Get the measurement id of the subject
        """
        measurements_of_subject = subset_ms_df.loc[(subset_ms_df[subject_id_col]==subject_id), measurement_id_col]
        measurements_bool_of_subject = m_index.isin(measurements_of_subject)
        """
        MSE
        """
        subject_score = mean_squared_error(pred_df.loc[measurements_bool_of_subject], groundtruth_df.loc[measurements_bool_of_subject])
        subject_scores.append(subject_score)
        """
        Weight by patients
        """
        weights.append(np.sqrt(m_count))

    np_subject_scores = np.array(subject_scores)
    np_weights = np.array(weights)
    weighted_score = np.dot(np_subject_scores,np_weights)/sum(np_weights)

    return weighted_score

def test_code():
    templates = {'on_off': "syn21344933",
                 'dyskinesia': "syn21344934",
                 'tremor': "syn21344949"}
    fold_pred_train_list = {1: [2, 3, 4, 5],
                            2: [1, 3, 4, 5],
                            3: [1, 2, 4, 5],
                            4: [1, 2, 3, 5],
                            5: [1, 2, 3, 4],
                            'validation': [1, 2, 3, 4, 5]}
    base_df_fn_list = ['hecky_cross_validation', 'dbmi_crossvalidations',
                       'haProzdor_crossvalidations', 'yuanfang_crossvalication']
    fold_list = [k for k in fold_pred_train_list]
    for t in templates:
        t_dict = {}
        for base_df_fn in base_df_fn_list:
            t_dict[base_df_fn] = []
            for fold in fold_list:
                test_df = pd.read_csv('processed_data/{}/{}/validating_file-{}.csv.gz'.format(t, base_df_fn, fold), index_col=0, compression = 'gzip')
                """
                Example of usage of the function
                """
                score = weighted_mse(test_df.loc[:,'label'], test_df.iloc[:,-1])
                t_dict[base_df_fn].append(score)
        t_df = pd.DataFrame(t_dict, index=fold_list)
        t_df.to_csv('processed_data/weighted_mse_score_{}.csv'.format(t), index_label='cv_fold')



if __name__ == "__main__":
    test_code()


