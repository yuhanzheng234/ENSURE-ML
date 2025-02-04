import xgboost as xgb #  brew install libomp for Mac users if error 'You are running 32-bit Python on a 64-bit OS' appears
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
import io

def compute_survival_function(cumhaz, predicted_log_hazards, times_total_unique, x_test): 
    # enter baseline h(t) and predicted output(i.e. exp(beta*x) in h(t)=h_0(t)exp(beta*x) on test set from model 
    # also enter times_total_unique and x_test for extracting discrete time points and patient id for indexing pandas frame output

    # function to obtain survival functions (n times x n samples) given output prediction of xgboost model and computed baseline hazard

    cumhaz_matrix = cumhaz[:, np.newaxis] * predicted_log_hazards[np.newaxis, :] # convert to matrix multiplication to achiveve one-time multiplication with all elements in predictions and h_0(t)
    surv_matrix = np.exp(-cumhaz_matrix)

    surv_df = pd.DataFrame(surv_matrix, index=times_total_unique, columns=x_test.index)

    return surv_df

def xgboost_model(df, clinical_endpoint):

    cat_cols = ['Clinical T stage', 'Clinical N stage',  'Clinical Differentiation', 'Tumour site', 'Pathologic T stage', 'Pathologic N stage', 'Differentiation', 'Treatment protocol', 'Operation type', 'Approach']

    # Convert categorical columns to category type
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    if clinical_endpoint == 'DFS':

        # Load XGBoost model inside the route
        bst = xgb.Booster()
        bst.load_model('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_dfs.model')
        # Obtain prediction on validation dataset
        dtest = xgb.DMatrix(df, enable_categorical=True)
        predicted_log_hazards = bst.predict(dtest) # of type numpy array

         # Load the variables from the file
        with open('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_baseline_hazards_dfs.pkl', 'rb') as f:
            data = pickle.load(f)

        haz = data['haz']
        cumhaz = data['cumhaz']
        times_total_unique = data['times_total_unique']

        surv_df = compute_survival_function(cumhaz, predicted_log_hazards, times_total_unique, df)
        ax = surv_df.iloc[:, :].plot()
        ax.legend().remove()
        plt.title(f'XGBoost predicted survival functions for {clinical_endpoint}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.grid()


    if clinical_endpoint == 'OS':

         # init model
        bst = xgb.Booster()
        bst.load_model('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_os.model')
        # Obtain prediction on validation dataset
        dtest = xgb.DMatrix(df, enable_categorical=True)
        predicted_log_hazards = bst.predict(dtest) # of type numpy array


         # Load the variables from the file
        with open('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_baseline_hazards_os.pkl', 'rb') as f:
            data = pickle.load(f)

        haz = data['haz']
        cumhaz = data['cumhaz']
        times_total_unique = data['times_total_unique']

        surv_df = compute_survival_function(cumhaz, predicted_log_hazards, times_total_unique, df)
        ax = surv_df.iloc[:, :].plot()
        ax.legend().remove()
        plt.title(f'XGBoost predicted survival functions for {clinical_endpoint}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.grid()


    if clinical_endpoint == 'DSS':

         # init model
        bst = xgb.Booster()
        bst.load_model('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_dss.model')
        # Obtain prediction on validation dataset
        dtest = xgb.DMatrix(df, enable_categorical=True)
        predicted_log_hazards = bst.predict(dtest) # of type numpy array


         # Load the variables from the file
        with open('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_baseline_hazards_dss.pkl', 'rb') as f:
            data = pickle.load(f)

        haz = data['haz']
        cumhaz = data['cumhaz']
        times_total_unique = data['times_total_unique']

        surv_df = compute_survival_function(cumhaz, predicted_log_hazards, times_total_unique, df)
        ax = surv_df.iloc[:, :].plot()
        ax.legend().remove()
        plt.title(f'XGBoost predicted survival functions for {clinical_endpoint}')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival probability')
        plt.grid()

    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Encode the image to base64
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    return encoded_image

