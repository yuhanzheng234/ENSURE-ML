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
# from pycox.models import CoxPH
from torchtuple_cox import _CoxPHBase
from pycox.models import loss as pycox_loss
# from pycox.evaluation import EvalSurv
from eva_surv import EvalSurv
import pandas as pd
from coxphloss import CoxPHLoss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from sklearn.utils import resample
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = ('cpu')
print(device)

random_seed = 100

class CoxPH_new(_CoxPHBase):
    def __init__(self, net, optimizer=None, device=None, loss=None, scheduler=None):
        if loss is None:
            loss = pycox_loss.CoxPHLoss()
        super().__init__(net, loss, optimizer, device)
        self.scheduler = scheduler

dataset = '/Users/mirandazheng/Desktop/ensure_processed_files_round2/imp_norm_DSS_dc.csv'
df = pd.read_csv(dataset)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:,0], random_state=random_seed)
print(train_df, test_df)

#-------------------------------prediction of testing dataset--------------------------------------------


# train the best model using the whole training set ------------------


train_df_new = train_df.drop(columns=['Centre Number', 'OS Censor', 'OS months', 'DFS months', 'DFS Censor', 'ENSURE ID - filter to get centres in order', 'Survival status']) 

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
y_train = get_target(train_df) # tuple of length 2, each element is a numpy array of size 312


#-------------------------------list hyperparameters--------------------------------------------------------

# network
num_nodes =  [32, 32]  # [[32, 32], [64, 64], [32, 32, 32], [32, 32, 32, 32]]
out_features = 1
batch_norm = True
dropout = 0.1 # [0.1, 0.2, 0.3]
output_bias = False # i.e. Wx (no +b)

# other hyperparameters
batch_size = 256 # [2400, 1800, 1000, 400, 200, 100, 10]
epochs = 100 # [25, 50, 100, 150, 200, 250, 300, 350, 400, 500]
# callbacks = [tt.callbacks.EarlyStopping()]
# verbose = True
learning_rate = 0.05 # [0.1, 0.01, 0.001, 0.0001]
optimiser = torch.optim.Adam
gamma = 0.7 # [0.9, 0.8, 0.7, 0.6]
weight_decay = 0.05 # [0, 0.01, 0.1]
scheduler = torch.optim.lr_scheduler.ExponentialLR

# Instantiate the deepsurv model
net = tt.practical.MLPVanilla(x_train.shape[1], num_nodes, out_features, batch_norm,
                        dropout, output_bias=output_bias)  # default activation function is nn.Relu

optimiser_instance = optimiser(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  # optimiser on the right is a class, left is class instance / object
scheduler_instance = scheduler(optimiser_instance, gamma=gamma)

       
model = CoxPH_new(net, optimiser_instance, loss=CoxPHLoss(), scheduler=scheduler_instance, device='cpu')


# bootstrap: obtain point estimate by running on whole test dataset -----------------------

model.load_model_weights('/Users/mirandazheng/Desktop/best_model_dss.pth')



_ = model.compute_baseline_hazards(x_train, y_train)

# Save the trained CoxPHFitter model to a file
with open("//Users/mirandazheng/Desktop/deepsurv_dss.pkl", "wb") as file:
    pickle.dump(model, file)

# torch.save(model, '/Users/mirandazheng/Desktop/deepsurv_dfs.pkl')

