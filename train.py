# %%
from models import *
from model_utils import *
from pymatreader import read_mat
from matplotlib import pyplot as plt 
from datetime import datetime
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from dataset import *
import torch
torch.set_default_dtype(torch.float64)
from datetime import datetime 
from os.path import join, split, splitext
from glob import glob
from visualization import *
import os  
now = datetime.now()

# %% [markdown]
# # Hyperparameters

# %%
IF_DIFFERENT_OPTIMIZERS = 1
TRAIN_EPOCH = 1500
LR = 1e-3
LR_REG = 1e-4
USE_ABS = 1
MODEL_CLASS = ECGI_Denoiser
LOSS_FN = MSELoss(reduction="mean")
str_MODEL_CLASS = MODEL_CLASS.__name__
TRAIN_BATCHSIZE = 64
N_UNROLL = 3

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %% [markdown]
# # Load reordered forward model and form datasets

# %%
A = read_mat("ForwMat_HT.mat")
A = torch.from_numpy(A['Trf_HT_leads']).to(device)
reordering = read_mat('newnode_order_3.mat')['node_order'].astype(int)
A = A[:,reordering-1]

# %%
train_dataset = ECGBeats('torso_concat.npy','heart_concat_reordered.npy',device=device, if_train=True, validation_ratio=0.05)
train_dataloader = DataLoader(train_dataset,batch_size=TRAIN_BATCHSIZE)


# %%
val_dataset = ECGBeats('torso_concat.npy','heart_concat_reordered.npy',device=device, if_train=False, validation_ratio=0.05)
VAL_BATCHSIZE = val_dataset.__len__()
val_dataloader = DataLoader(val_dataset,batch_size=VAL_BATCHSIZE)

# %% [markdown]
# # Take the reg param from the last trained model 

# %%
last_model_path = get_the_latest_file_in_path(join('Regs','*.pt'),-9)
last_model = torch.load(last_model_path)
last_reg = last_model['reg_params'].item()
EXP_NAME = splitext(os.path.split(last_model_path)[-1])[0]
EXP_NAME = '_'.join(EXP_NAME.split('_')[5:])
whole_date_time = str(now.year) + "_"+str(now.month) + "_"+str(now.day) + '_' +str(now.hour) + '_' + str(now.minute) + '_'
SAVE_PATH = whole_date_time + EXP_NAME
model = ECGI_Denoiser().to(device)
reg_params = nn.Parameter(torch.Tensor([last_reg/2**i  for i in range(N_UNROLL)]).requires_grad_(True))
# %%
model = train_model(model, reg_params, LOSS_FN, train_dataloader, val_dataloader, LR, LR_REG, TRAIN_EPOCH, A, device, True, EXP_NAME, join("Trains", SAVE_PATH), model_load_path=None, different_optimizers=True, one_zero_normalization=True)
# %%
last_model = get_the_latest_file_in_path(join('Trains','*.pt'))
EXP_NAME = splitext(os.path.split(last_model)[-1])[0]
EXP_NAME = '_'.join(EXP_NAME.split('_')[5:])
result_figure = visualize_results(last_model)

# %%
Experiment_name = "AbsUsed_" + str(USE_ABS) + "_DifferentOptims_" + str(IF_DIFFERENT_OPTIMIZERS) + "LR_" + str(LR) + "_LRReg_" + str(LR_REG) \
    + "_Unrolling_" + str(N_UNROLL) + "_model_" + str_MODEL_CLASS
Experiment_name = ''.join([item for item in Experiment_name if not item == '.'])
fig_path = Experiment_name+".png"
fig_path = join('Figs',fig_path)
result_figure.savefig(fig_path)
