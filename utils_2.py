import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools


def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df['generic_drug_name'] = df.apply(lambda x: ndc_df[ndc_df['NDC_Code'] == x['ndc_code']]['Non-proprietary Name'].iloc[0] if x['ndc_code'] in ndc_df['NDC_Code'].values else np.nan , axis=1)
    return df


def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    # conver df to encounter level
    #grouping_field_list= ['encounter_id', 'patient_nbr' ]
    #non_grouping_field_list= [c for c in df.columns if c not in grouping_field_list]
    #df = df.groupby(grouping_field_list)[non_grouping_field_list].agg(lambda x: list([y for y in x if y is not np.nan])).reset_index()
    
    #grouping_field_list =['encounter_id','patient_nbr','gender','age','admission_type_id','discharge_disposition_id','admission_source_id','time_in_hospital','number_outpatient','number_inpatient','number_emergency','num_lab_procedures','number_diagnoses','num_medications','num_procedures','change','readmitted']
    
    #non_grouped_field_list= [c for c in df.columns if c not in grouping_field_list]
    #encounter_df = df.groupby(grouping_field_list)[non_grouped_field_list].agg(lambda x: 
                                                     #   list([y for y in x if y is not np.nan ] ) ).reset_index()
    
    df_sorted = df.sort_values('encounter_id')
    last_encounter_values = df_sorted.groupby(['patient_nbr'])['encounter_id'].head(1).values
    
    return df_sorted[df_sorted['encounter_id'].isin(last_encounter_values)]


def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len (unique_values)
    sample_size = round(total_values * 0.6)
    train = df[df[patient_key].isin(unique_values[:sample_size])].reset_index(drop = True)
    validation = df[df[patient_key].isin(unique_values[sample_size:round(total_values * 0.8)])].reset_index(drop = True)
    test = df[df[patient_key].isin(unique_values[round(total_values * 0.8):])].reset_index(drop = True)
    
    return train, validation, test

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...'''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key = c, vocabulary_file = vocab_file_path, num_oov_buckets=0)
        
        one_hot = tf.feature_column.indicator_column(tf_categorical_feature_column)
       
        output_tf_list.append(one_hot)
    return output_tf_list

def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean = MEAN, std = STD)
    
    return tf.feature_column.numeric_column(key=col, default_value = default_value, normalizer_fn = normalizer, dtype = tf.float64)

def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

def get_student_binary_prediction(col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    
    return np.array([ 1 if abs(x)>=5 else 0 for x in col], dtype=int)

def check_for_missing_and_null(df):
    null_df = pd.DataFrame({'columns':df.columns,
                           'percent_null':df.isnull().sum()*100/len(df),
                           'percent_Zero':df.isin([0]).sum()*100/len(df)})
    return null_df

def count_unique_values(df, cat_col_list):
    cat_df = df[cat_col_list]
    val_df = pd.DataFrame({'columns':cat_df.columns,
                           'cadinality':cat_df.nunique()})
    return val_df
