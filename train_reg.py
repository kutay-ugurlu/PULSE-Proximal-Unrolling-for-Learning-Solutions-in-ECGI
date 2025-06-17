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
import argparse


# %%
IF_DIFFERENT_OPTIMIZERS = 1
REG_EPOCH = 600
LR_REG = 1e-5
USE_ABS = 1
MODEL_CLASS = LSTM_UNet
LOSS_FN = MSELoss(reduction="mean")
str_MODEL_CLASS = MODEL_CLASS.__name__
parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size',type=int, default=32)
args = parser.parse_args()


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
A_HLT = read_mat(join("GeometricVariation","GeometricModels","AlreadyReordered_HLT_Original.mat"))
A_HLT = torch.from_numpy(A_HLT['Trf_HLT_coarse']).to(device)
A_HT = read_mat(join("GeometricVariation","GeometricModels","AlreadyReordered_HT_Original.mat"))
A_HT = torch.from_numpy(A_HT['Trf_HT_coarse']).to(device)
reordering = read_mat('newnode_order_3.mat')['node_order'].astype(int) - 1 # MATLAB Python conversion 

# %%
training_files = glob('TrainingData/*.mat')

TRAIN_BATCHSIZE = args.batch_size
    
dataset = ECGBeats_Batched(files=training_files, A=A_HLT, batch_size=TRAIN_BATCHSIZE, if_train=True, device=device, reordering=reordering, validation_ratio=0.05)
train_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
dataset = ECGBeats_Batched(files=training_files, A=A_HLT, batch_size=TRAIN_BATCHSIZE, if_train=False, device=device, reordering=reordering, validation_ratio=0.05)
val_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
N_UNROLL = 1
first_reg = 1.75e-4

# %%
for LR in [1e-4]:
    for model in [MODEL_CLASS(batch_size=TRAIN_BATCHSIZE).to(device)]:
        now = datetime.now()
        EXP_NAME = "AbsUsed_" + str(USE_ABS) + "_Unrolling_" + str(N_UNROLL) + "_model_" + str_MODEL_CLASS
        whole_date_time = str(now.year) + "_"+str(now.month) + "_"+str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)
        SAVE_PATH = whole_date_time + EXP_NAME

        reg_params = nn.Parameter(torch.Tensor([first_reg for i in range(N_UNROLL)]).requires_grad_(True))
        reg_model = train_model(model, reg_params, LOSS_FN, train_dataloader,
                                val_dataloader, LR, LR_REG, REG_EPOCH, A_HT, device, USE_ABS, EXP_NAME, join("Regs", SAVE_PATH))

        last_model = get_the_latest_file_in_path(join('Regs','*.pt'))
        EXP_NAME = splitext(os.path.split(last_model)[-1])[0]
        EXP_NAME = '_'.join(EXP_NAME.split('_')[5:])
        result_figure = visualize_results(last_model)
        fig_path = EXP_NAME+".png"
        fig_path = join('Figs',fig_path)
        result_figure.savefig(fig_path)

        whole_date_time = str(now.year) + "_"+str(now.month) + "_"+str(now.day) + '_' +str(now.hour) + '_' + str(now.minute) + '_'
        subject = "Reg Training completed (" + whole_date_time[:-1]   
        msg = EXP_NAME + ")\n" + str(model.__class__)
