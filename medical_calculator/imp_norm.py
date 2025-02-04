import numpy as np
import pickle

def imp_norm(df, clinical_endpoint):

    for col in df.columns:
        # Replace 'unknown' and 'Unknown' with np.nan
        df[col] = df[col].replace({'unknown': np.nan, 'Unknown': np.nan}, inplace=False)

        


    # replace the time-to-event by culmulative hazard to avoid bias
    # for the test set, create two pseudo columns to accomodate fitted imputer 
    # groundtruth of test set cannot be used as during real inference stage, there will be no groundtruth. 
    # groudtruth is only used for measuring model performances.
    df[f'{clinical_endpoint} Censor'] = np.nan
    df['cumulative_hazard'] = np.nan 


    test_df_all_cols = df.columns
    # test_df_variable_cols = test_df.drop(['Centre Number', 'OS Censor', 'OS months', 'DSS Censor', 'DSS months', 'DFS months', 'DFS_Censor_original', 'DFS Censor', 'cumulative_hazard', 'ENSURE ID - filter to get centres in order', 'Survival status'], inplace=False, axis=1).columns
    test_df_variable_cols = ['Age at diagnosis', 'Number of nodes analyzed', 'Length of stay', 'GDP USD per cap', 'H volutary', 'HE Compulsory' ]

    with open(f"/Users/mirandazheng/Desktop/first/ensure/medical_calculator/imputer_{clinical_endpoint}.pkl", "rb") as imputer_file:
        loaded_imputer = pickle.load(imputer_file)

    # Load the scaler
    with open(f"/Users/mirandazheng/Desktop/first/ensure/medical_calculator/scaler_{clinical_endpoint}.pkl", "rb") as scaler_file:
        loaded_scaler = pickle.load(scaler_file)


    # Transform the test data using the same imputer
    df.loc[:, test_df_all_cols] = loaded_imputer.transform(df[test_df_all_cols])
    # normalise the test data using the mean and std of training data to avoid bias
    df.loc[:, test_df_variable_cols] = loaded_scaler.transform(df[test_df_variable_cols]) 


    df.drop(columns=['cumulative_hazard', f'{clinical_endpoint} Censor'], inplace=True)

    return df
