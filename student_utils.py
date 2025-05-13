import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    # Renommer les colonnes de ndc_df temporairement pour correspondre à df
    ndc_df_renamed = ndc_df.rename(columns={
        'NDC_Code': 'ndc_code',
        'Non-proprietary Name': 'generic_drug_name'  # adapte exactement au nom présent dans ton jeu de données
    })

    # Fusionner sur la colonne ndc_code pour ajouter la colonne generic_drug_name
    df = df.merge(ndc_df_renamed[['ndc_code', 'generic_drug_name']], on='ndc_code', how='left')

    return df



#Question 4
def select_first_encounter(df):
    """
    df: pandas dataframe contenant toutes les rencontres.
    return:
        - first_encounter_df: dataframe contenant uniquement la première rencontre pour chaque patient.
    """
    df_sorted = df.sort_values(by=['patient_nbr', 'encounter_id'])
    first_encounter_df = df_sorted.groupby('patient_nbr', as_index=False).first()
    return first_encounter_df




#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    np.random.seed(42)
    unique_patients = df[patient_key].unique()
    np.random.shuffle(unique_patients)

    total = len(unique_patients)
    train_end = int(0.6 * total)
    val_end = int(0.8 * total)

    train_patients = unique_patients[:train_end]
    val_patients = unique_patients[train_end:val_end]
    test_patients = unique_patients[val_end:]

    train = df[df[patient_key].isin(train_patients)]
    validation = df[df[patient_key].isin(val_patients)]
    test = df[df[patient_key].isin(test_patients)]

    return train, validation, test



#Question 7

def create_tf_categorical_feature_cols(categorical_col_list, d_train):
    output_tf_list = []
    for c in categorical_col_list:
        
        vocab = list(d_train[c].dropna().unique())  # Enlever les NaN avant de récupérer les valeurs uniques

        lookup = tf.keras.layers.StringLookup(
            vocabulary=vocab,
            output_mode='int',
            num_oov_indices=1,  # Indice pour les valeurs inconnues
            mask_token=None,
            name=f"{c}_lookup"
        )

        embedding = tf.keras.layers.Embedding(input_dim=len(vocab) + 2, output_dim=10)  # +2 = OOV + padding

        output_tf_list.append((c, lookup, embedding))

    return output_tf_list



#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    Normalizes a column using Z-score normalization
    '''
    return (col - mean) / std



def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    '''
    Create TensorFlow numeric features with normalization layers for each numeric column.
    '''
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TensorFlow Probability prediction object
    '''
    # Déplace les prédictions sur le CPU explicitement avec tf.identity
    if diabetes_yhat.device != '/device:CPU:0':
        diabetes_yhat = tf.identity(diabetes_yhat)  # Utilisation de tf.identity pour forcer le déplacement sur le CPU

    preds = diabetes_yhat.numpy().flatten()
    m = np.mean(preds)
    s = np.std(preds)
    return m, s



# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    probs = df[col].values
    binary_labels = (probs >= 0.5).astype(int)
    return pd.DataFrame(binary_labels, columns=['PredictedLabel'])

