#%%
#  has to be done locally otherwise some cuda problem (jupyter by defualt use some gpu, which will run into problem if you run it locally)
import sys
sys.path.append('/Users/mirandazheng/Desktop/first/ensure/medical_calculator')
sys.path.append('/Users/mirandazheng/Desktop/medical_calculator')
import pycox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
from pycox.datasets import metabric
# from pycox.models import CoxPH
from torchtuple_cox import _CoxPHBase
from pycox.models import loss as pycox_loss
# from pycox.evaluation import EvalSurv
from eva_surv import EvalSurv
import pandas as pd
from coxphloss import CoxPHLoss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import pickle
import base64
import io
import dill

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CoxPH_new(_CoxPHBase):
    def __init__(self, net, optimizer=None, device=None, loss=None, scheduler=None):
        if loss is None:
            loss = pycox_loss.CoxPHLoss()
        super().__init__(net, loss, optimizer, device)
        self.scheduler = scheduler

df = pd.read_excel('/Users/mirandazheng/Desktop/medical_calculator/2937_imp_norm_OS.xlsx') # note: can only check curve consistency for OS; as DFS has different imputations

with open("/Users/mirandazheng/Desktop/first/ensure/medical_calculator/deepsurv_os.pkl", "rb") as file:
    loaded_deepsurv = pickle.load(file)
    
cols_leave = df.columns
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(leave)
x_test = x_mapper.fit_transform(df).astype('float32') 

surv = loaded_deepsurv.predict_surv_df(x_test)
surv.iloc[:,:].plot() # attributes of pandas.dataframe.plot
plt.title(f'DeepSurv predicted survival functions for OS')
plt.xlabel('Time (months)')
plt.ylabel('Survival probability')
plt.grid()




df = pd.read_csv('/Users/mirandazheng/Desktop/ensure_processed_files_round2/imp_norm_OS_dc.csv')
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:,0], random_state=100)
print(train_df, test_df)

test_df_new = test_df.drop(columns=['Centre Number', 'DFS Censor', 'DFS months', 'DSS months', 'DSS Censor', 'ENSURE ID - filter to get centres in order', 'Survival status']) 

# prepare test data to deepsurv required format
cols_leave = test_df_new.columns[:-2]  # to exclude relapse and rfs column
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(leave)
x_test = x_mapper.fit_transform(test_df_new).astype('float32') 
get_target = lambda df: (df['OS months'].values, df['OS Censor'].values)
durations_test, events_test = get_target(test_df_new)

surv = loaded_deepsurv.predict_surv_df(x_test)  # S(t) =  exp(-H(x,t)) = baseline * exp(log_h)) # pandas frame type, n_times x n_samples
# print(type(surv))  # 295,98 of type df
surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
plt.grid()

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
point_concordance = ev.concordance_td()
print(point_concordance)



# %%
