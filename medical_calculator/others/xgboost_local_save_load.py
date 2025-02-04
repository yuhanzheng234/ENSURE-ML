#%%

import xgboost as xgb #  brew install libomp for Mac users if error 'You are running 32-bit Python on a 64-bit OS' appears
from sksurv.datasets import load_aids
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import sys
print(sys.path)
sys.path.append('/gpfs3/well/papiez/users/pea322/ensure')
print(sys.path)
import pycox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
# from pycox.models import CoxPH
from torchtuple_cox import _CoxPHBase
# from pycox.evaluation import EvalSurv
from eva_surv import EvalSurv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import msoffcrypto
import io
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from sklearn.utils import resample
import pickle

random_seed = 100

filepath= '/Users/mirandazheng/Desktop/ensure_processed_files_round2/ensure_model1_DSS_2.xlsx'
password = 'se9Ngufu!'

def load_password_protected_excel(filepath, password):
     temp = io.BytesIO()  # create a file-like object i.e. output file handle which can be accessed as a file but stored in RAM (i.e. don't occupy physical space on disk)
 
     with open(filepath, 'rb') as f: 
         # rb stands for read in binary format and r reads in text format. here should be rb as excel is a binary file.
         # for csv, generally read in text as it is a direct display, but binary is faster and also preserve accuracy
         excel = msoffcrypto.OfficeFile(f)  # create a file handle to the encrypted file
         excel.load_key(password)
         excel.decrypt(temp)  # decrypt the file using the password
    
     temp.seek(0)  # explictly move the pointer from the end of the file to the start
     df = pd.read_excel(temp)
     
     del temp 
 
     return df


 # call the function
df = load_password_protected_excel(filepath, password)

new_header = df.iloc[0]  # Grab the first row for the header
df = df[1:]  # Take the data less the header row
df.columns = new_header  # Set the header row as the df header

# Optionally, reset the index of the DataFrame
df.reset_index(drop=True, inplace=True)


for col in df.columns:
    # Replace 'unknown' and 'Unknown' with np.nan
    df[col] = df[col].replace({'unknown': np.nan, 'Unknown': np.nan}, inplace=False)

# Show the DataFrame to verify
print(df)

# without dummy coding, but use internal categorical handler
# List of categorical columns
cat_cols = ['Clinical T stage', 'Clinical N stage',  'Clinical Differentiation', 'Tumour site', 'Pathologic T stage', 'Pathologic N stage', 'Differentiation', 'Treatment protocol', 'Operation type', 'Approach']

# Convert categorical columns to category type
for col in cat_cols:
    df[col] = df[col].astype('category')



train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:,0], random_state=random_seed)
print(train_df, test_df)

def obtain_baseline_hazards(best_model, d_x_train, y_train): # input fitted model, DMatrix format of x_train, and original y_train in pd format

    # customized breslow estimation function, return baseline hazard function h(t)
    # ref: https://github.com/IyarLin/survXgboost/tree/master/R 
        
    # defining several time variables for breslow calculation
    label = y_train.to_numpy()  # convert label from pd to numpy format
    times_total = np.abs(label)
    times_total_unique = np.sort(np.unique(times_total))
    events = label > 0
    times_events = times_total[events]
    times_events_unique = np.sort(np.unique(times_events))

    # print(times_total)
    # print(times_total_unique)
    # print(events)
    # print(times_events)
    # print(times_events_unique)

    # get predicted log hazards (i.e. exp(beta*x) in h(t)=h_0(t)exp(beta*x)) of the fitted model on the training set for breslow calculation
    predicted_log_hazards = best_model.predict(d_x_train)
    

    # Calculate hazard for each unique time in events
    haz = np.zeros(len(times_total_unique)) # consider all time points exist in dataset
    for i, t in enumerate(times_total_unique): 
        if t in times_events_unique: # if there is >=1 event cases at this time
            event_mask = times_events == t
            risk_set_mask = times_total >= t
            haz[i] = np.sum(event_mask) / np.sum(np.exp(predicted_log_hazards[risk_set_mask]))
        else:
            haz[i] = 0 # no event at this time, hazard = 0, hence cumulative hazard remains unchanged as previous time step

    # Calculate cumulative hazard
    cumhaz = np.cumsum(haz)

    # Ensure that time zero is included if not already present
    if 0 not in times_total_unique:
        haz = np.insert(haz, 0, 0)
        cumhaz = np.insert(cumhaz, 0, 0)
        times_total_unique = np.insert(times_total_unique, 0, 0)

    # print(haz)
    # print(cumhaz)
    # print(times_total_unique)
        
    return haz, cumhaz, times_total_unique
def compute_survival_function(cumhaz, predicted_log_hazards, times_total_unique, x_test): 
    # enter baseline h(t) and predicted output(i.e. exp(beta*x) in h(t)=h_0(t)exp(beta*x) on test set from model 
    # also enter times_total_unique and x_test for extracting discrete time points and patient id for indexing pandas frame output

    # function to obtain survival functions (n times x n samples) given output prediction of xgboost model and computed baseline hazard

    cumhaz_matrix = cumhaz[:, np.newaxis] * predicted_log_hazards[np.newaxis, :] # convert to matrix multiplication to achiveve one-time multiplication with all elements in predictions and h_0(t)
    surv_matrix = np.exp(-cumhaz_matrix)

    surv_df = pd.DataFrame(surv_matrix, index=times_total_unique, columns=x_test.index)

    return surv_df

# Features and target
x_train = train_df.drop(columns=['Centre Number', 'DFS Censor', 'DFS months', 'OS Censor', 'OS months', 'DSS Censor', 'DSS months', 'ENSURE ID - filter to get centres in order', 'Survival status']) # 785 x 64
y_train = train_df[['DSS months', 'DSS Censor']]  # 3136 x 2 # Make sure 'time_to_event' is the first column
y_train = y_train.copy() # to avoid pandas warning
y_train.loc[:, 'label'] = y_train.apply(lambda row: row['DSS months'] if row['DSS Censor'] == 1 else -row['DSS months'], axis=1) # 3136 x 3

# print(x_train.shape)  # (341,14)
# print(y_train[0].shape)  # (341,)

# # Select a random row
# random_row = train_df.sample(n=1, random_state=random_seed)
# print(train_df)
# # Drop the selected row
# train_df = train_df.drop(random_row.index) # drop one row randomly, otherwise will incur an error due to batch size of one for this code
# print(random_row.index)
# print(train_df)

# Convert data to DMatrix, which is a high-performance XGBoost data structure
dtrain = xgb.DMatrix(x_train, label=y_train['label'], enable_categorical=True)




# Load the model from disk
bst = xgb.Booster()  # init model
bst.load_model('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_dss.model')

x_test = test_df.drop(columns=['Centre Number', 'DFS Censor', 'DFS months', 'OS Censor', 'OS months', 'DSS Censor', 'DSS months', 'ENSURE ID - filter to get centres in order', 'Survival status']) # 785 x 64
y_test = test_df[['DSS months', 'DSS Censor']]  # 785 x 2 # Make sure 'time_to_event' is the first column
y_test = y_test.copy() # to avoid pandas warning
y_test.loc[:, 'label'] = y_test.apply(lambda row: row['DSS months'] if row['DSS Censor'] == 1 else -row['DSS months'], axis=1) # 785 x 3

# Obtain prediction on validation dataset
dtest = xgb.DMatrix(x_test, enable_categorical=True)
predicted_log_hazards = bst.predict(dtest) # of type numpy array

haz, cumhaz, times_total_unique = obtain_baseline_hazards(bst, dtrain, y_train['label'])
# Save the variables to a file
with open('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_baseline_hazards_dss.pkl', 'wb') as f:
    pickle.dump({'haz': haz, 'cumhaz': cumhaz, 'times_total_unique': times_total_unique}, f)

# Load the variables from the file
with open('/Users/mirandazheng/Desktop/first/ensure/medical_calculator/xgb_baseline_hazards_dss.pkl', 'rb') as f:
    data = pickle.load(f)

haz = data['haz']
cumhaz = data['cumhaz']
times_total_unique = data['times_total_unique']

surv_df = compute_survival_function(cumhaz, predicted_log_hazards, times_total_unique, x_test)
# print(surv_df)
surv_df.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
plt.xlabel('Time')

# evaluation
get_target = lambda df: (df['DSS months'].values, df['DSS Censor'].values)
durations_test, events_test = get_target(y_test)

ev = EvalSurv(surv_df, durations_test, events_test, censor_surv='km')
point_concordance = ev.concordance_td()
print(point_concordance)







df = pd.read_excel('/Users/mirandazheng/Desktop/medical_calculator/2937_OS.xlsx')
for col in df.columns:
    # Replace 'unknown' and 'Unknown' with np.nan
    df[col] = df[col].replace({'unknown': np.nan, 'Unknown': np.nan}, inplace=False)
cat_cols = ['Clinical T stage', 'Clinical N stage',  'Clinical Differentiation', 'Tumour site', 'Pathologic T stage', 'Pathologic N stage', 'Differentiation', 'Treatment protocol', 'Operation type', 'Approach']

# Convert categorical columns to category type
for col in cat_cols:
    df[col] = df[col].astype('category')

# Load the model from disk
bst = xgb.Booster()  # init model
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
# print(surv_df)
surv_df.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
plt.xlabel('Time')
# %%
