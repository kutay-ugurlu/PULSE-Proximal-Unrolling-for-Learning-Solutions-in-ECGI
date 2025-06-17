from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pymatreader import read_mat
from utils import add_noise

def in_data_batch_completer(voltage, batch_size):
    
    # Completes the each beats training data to the specified batches of samples
    n_t = voltage.shape[1]
    
    if batch_size == n_t:
        return voltage
    
    elif batch_size > n_t:
        repetition_number = batch_size//n_t + 1
        return np.tile(voltage, (1,repetition_number))[:,0:batch_size]
    
    else: # The most probable case where n_t >> batch_size 
        n_remainder = n_t % batch_size
        n_complete_batch = n_t//batch_size

        if n_remainder:
            complete_batches = np.split(voltage[:,:-1*n_remainder],n_complete_batch,axis=1)
            complete_batches.append(voltage[:,-1*batch_size:])
            return np.hstack(complete_batches)
        
        else:
            complete_batches = np.split(voltage,n_complete_batch,axis=1)
            return np.hstack(complete_batches)
        
        
def create_training_set(batch_size, training_files, reordering):
    
    container = np.empty((490,0))
    for item in training_files:
        data = read_mat(item)
        qrs_begin = data['features']['QRSbegin'] - 1
        qrs_end = data['features']['QRSend'] - 1 
        potvals = data['ts']['potvals'][reordering,qrs_begin:qrs_end]

        # Complete it to the nearest integer multiple of batch_size or slice it accordingly 
        batches = in_data_batch_completer(voltage=potvals, batch_size=batch_size)
                
        container = np.concatenate((container,batches),axis=1)
    return container

def create_test_set(batch_size, test_files, reordering):
    
    container = np.empty((490,0))
    for item in test_files:
        data = read_mat(item)
        potvals = data['ep']['potvals']

        # Complete it to the nearest integer multiple of batch_size or slice it accordingly 
        batches = in_data_batch_completer(voltage=potvals[reordering, :], batch_size=batch_size)
                
        container = np.concatenate((container,batches),axis=1)
    return container

class ECGBeats(Dataset):
    def __init__(self, torso_file, heart_file, if_train, device, transform=None, target_transform=None, validation_ratio=0.05):
        self.heart = torch.from_numpy(np.load(heart_file))
        self.torso = torch.from_numpy(np.load(torso_file))
        self.train_heart, self.val_heart, self.train_torso, self.val_torso = train_test_split(self.heart.T,self.torso.T,random_state=101,test_size=validation_ratio)
        self.if_train = if_train
        self.device = device
    
    def __len__(self):
        if self.if_train:
            return self.train_heart.shape[0] 
        else: # if validation
            return self.val_heart.shape[0]
    
    def __getitem__(self, index):
        if self.if_train:
            return self.train_heart[index,:].to(self.device), self.train_torso[index,:].to(self.device)
        else:
            return self.val_heart[index,:].to(self.device), self.val_torso[index,:].to(self.device)
        
    
class ECGBeats_Batched(Dataset):
    def __init__(self, files, A, batch_size, if_train, device, reordering, transform=None, target_transform=None, validation_ratio=0.05):
        
        self.batch_size = batch_size
        self.if_train = if_train
        self.device = device
        
        files = np.array(files)
        train_idx, validation_idx = train_test_split(np.arange(0,files.__len__()),random_state=101,test_size=validation_ratio)
        training_files = files[train_idx]
        validation_files = files[validation_idx]
        
        if self.if_train:
        
            train_heart = create_training_set(training_files=training_files,batch_size=self.batch_size, reordering=reordering)
            self.noise_free_torso = np.matmul(A.cpu().numpy(),train_heart)
            noisy_torso,_,_ = add_noise(self.noise_free_torso,30)
            batch_no = noisy_torso.shape[1]//batch_size
            self.train_torso = torch.reshape(torch.from_numpy(noisy_torso).T,(batch_no,batch_size,noisy_torso.shape[0]))
            self.train_heart = torch.reshape(torch.from_numpy(train_heart).T,(batch_no,batch_size,train_heart.shape[0]))
            
        else:
        
            val_heart = create_training_set(training_files=validation_files,batch_size=self.batch_size, reordering=reordering)
            self.noise_free_torso = np.matmul(A.cpu().numpy(),val_heart)
            noisy_torso,_,_ = add_noise(self.noise_free_torso,30)
            batch_no = noisy_torso.shape[1]//batch_size
            self.val_torso = torch.reshape(torch.from_numpy(noisy_torso).T,(batch_no,batch_size,noisy_torso.shape[0]))
            self.val_heart = torch.reshape(torch.from_numpy(val_heart).T,(batch_no,batch_size,val_heart.shape[0]))
        
    def __len__(self):
        if self.if_train:
            return self.train_heart.shape[0] 
        else: # if validation
            return self.val_heart.shape[0]
    
    def __getitem__(self, index):
        if self.if_train:
            return torch.squeeze_copy(self.train_heart[index,...].to(self.device)), self.train_torso[index,...].to(self.device)
        else:
            return self.val_heart[index,...].to(self.device), self.val_torso[index,...].to(self.device)

class ECGBeats_Test_Batched(Dataset):
    def __init__(self, files, A, batch_size, device, reordering, transform=None, target_transform=None, validation_ratio=0.05):
        
        self.batch_size = batch_size
        self.device = device
                
        test_heart = create_test_set(test_files=files,batch_size=self.batch_size, reordering=reordering)
        self.noise_free_torso = np.matmul(A.cpu().numpy(),test_heart)
        noisy_torso,_,_ = add_noise(self.noise_free_torso,30)
        self.test_torso = torch.from_numpy(noisy_torso).T
        self.test_heart = torch.from_numpy(test_heart).T

        
    def __len__(self):
        return self.test_heart.shape[0] 

    
    def __getitem__(self, index):
        return self.test_heart[index,...].T.to(self.device), self.test_torso[index,...].T.to(self.device)


class TestBeats(Dataset):
    
    def __init__(self, torso, heart, device, transform=None, target_transform=None):
        self.heart = heart
        self.torso = torso
        self.device = device
        
    def __len__(self):
        return self.heart.shape[0]
    
    def __getitem__(self, index):
        return self.heart[index,:].to(self.device), self.torso[index,:].to(self.device)
