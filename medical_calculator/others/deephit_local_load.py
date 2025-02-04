#%%
#  has to be done locally otherwise some cuda problem (jupyter by defualt use some gpu, which will run into problem if you run it locally)
import sys
sys.path.append('/Users/mirandazheng/Desktop/first/ensure')
sys.path.append('/Users/mirandazheng/Desktop/medical_calculator')
import pycox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
from pycox.datasets import metabric
# from pycox.models import DeepHitSingle
from torchtuple_pmf import PMFBase
from pycox.models import loss as pycox_loss
from pycox import models
# from pycox.evaluation import EvalSurv
from eva_surv import EvalSurv
import pandas as pd
from coxphloss import CoxPHLoss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from sklearn.utils import resample
import pickle
import base64
import io
import dill

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class DeepHitSingle_new(PMFBase):

    def __init__(self, net, optimizer=None, device=None, duration_index=None, alpha=0.2, sigma=0.1, loss=None, scheduler=None):
        if loss is None:
            loss = pycox_loss.DeepHitSingleLoss(alpha, sigma)
        super().__init__(net, loss, optimizer, device, duration_index)
        self.scheduler = scheduler

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=models.data.DeepHitDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

df = pd.read_excel('/Users/mirandazheng/Desktop/medical_calculator/2937_imp_norm_OS.xlsx')

with open("/Users/mirandazheng/Desktop/first/ensure/medical_calculator/deephit_os.pkl", "rb") as file:
    loaded_deephit = pickle.load(file)
    
cols_leave = df.columns
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(leave)
x_test = x_mapper.fit_transform(df).astype('float32') 

surv = loaded_deephit.predict_surv_df(x_test)
surv = loaded_deephit.interpolate(50).predict_surv_df(x_test) # optimal interpolate number is 50
surv.iloc[:,:].plot(drawstyle='steps-post') # attributes of pandas.dataframe.plot
plt.title(f'DeepSurv predicted survival functions for OS')
plt.xlabel('Time (months)')
plt.ylabel('Survival probability')




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

surv = loaded_deephit.predict_surv_df(x_test)  # S(t) =  exp(-H(x,t)) = baseline * exp(log_h)) # pandas frame type, n_times x n_samples
surv = loaded_deephit.interpolate(50).predict_surv_df(x_test) # optimal interpolate number is 50
# print(type(surv))  # 295,98 of type df
surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
point_concordance = ev.concordance_td()
print(point_concordance)



# %%
