# has to be done locally otherwise some cuda problem (jupyter by defualt use some gpu, which will run into problem if you run it locally)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = ('cpu')
print(device)

random_seed = 100

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


dataset = '/Users/mirandazheng/Desktop/ensure_processed_files_round2/imp_norm_DSS_dc.csv'
df = pd.read_csv(dataset)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:,0], random_state=random_seed)
print(train_df, test_df)

# network
num_durations = 10 # [5, 10, 20] # the number of output
num_nodes =  [32, 32]  # [[32, 32], [64, 64], [32, 32, 32], [32, 32, 32, 32]]
out_features = num_durations # or = labtrans.out_features

batch_norm = True
dropout = 0.1  # [0.1, 0.2, 0.3]
# assume alpha=0.2, sigma=0.1 unchanged for now

# other hyperparameters
batch_size = 64 # [2400, 1800, 1000, 400, 200, 100, 10]
epochs = 75 # [25, 50, 100, 150, 200, 250, 300, 350, 400, 500]
# callbacks = [tt.callbacks.EarlyStopping()]
# verbose = True
learning_rate = 0.01  # [0.1, 0.01, 0.001, 0.0001]
optimiser = torch.optim.Adam
gamma = 0.7  # [0.9, 0.8, 0.7, 0.6]
weight_decay = 0.05  # [0, 0.01, 0.1]
scheduler = torch.optim.lr_scheduler.ExponentialLR

surv_interpolate_number = 50  #[10, 50]


#-------------------------------prediction of testing dataset--------------------------------------------


# train the best model using the whole training set ------------------


train_df_new = train_df.drop(columns=['Centre Number', 'DFS Censor', 'DFS months', 'OS months', 'OS Censor', 'ENSURE ID - filter to get centres in order', 'Survival status']) 

# print(x_train.shape)  # (341,14)
# print(y_train[0].shape)  # (341,)

# # Select a random row
# random_row = train_df.sample(n=1, random_state=random_seed)
# print(train_df)
# # Drop the selected row
# train_df = train_df.drop(random_row.index) # drop one row randomly, otherwise will incur an error due to batch size of one for this code
# print(random_row.index)
# print(train_df)

cols_leave = train_df_new.columns[:-2]  # to exclude relapse and rfs column
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(leave)
x_train = x_mapper.fit_transform(train_df).astype('float32') # of type numpy array and shape unchanged e.g. (312x16)
get_target = lambda df: (df['DSS months'].values, df['DSS Censor'].values)
labtrans = DeepHitSingle_new.label_transform(out_features)
y_train = labtrans.fit_transform(*get_target(train_df_new))


#-------------------------------list hyperparameters--------------------------------------------------------


# Instantiate the deephit model
net = tt.practical.MLPVanilla(x_train.shape[1], num_nodes, out_features, batch_norm,
                        dropout, output_activation=nn.Softmax(dim=1))  # default activation function is nn.Relu

optimiser_instance = optimiser(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  # optimiser on the right is a class, left is class instance / object
scheduler_instance = scheduler(optimiser_instance, gamma=gamma)

model = DeepHitSingle_new(net, optimiser_instance, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts, device=device, scheduler=scheduler_instance)



# bootstrap: obtain point estimate by running on whole test dataset -----------------------

model.load_model_weights('/Users/mirandazheng/Desktop/best_model_dss.pth')


# Save the trained CoxPHFitter model to a file
with open("//Users/mirandazheng/Desktop/deephit_dss.pkl", "wb") as file:
    pickle.dump(model, file)

# torch.save(model, '/Users/mirandazheng/Desktop/deepsurv_dfs.pkl')
