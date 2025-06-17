from torch import nn
import torch
import numpy as np
from tqdm import tqdm
from metrics import correlation_coefficient, relative_error, four_metrics, metrics_with_bad_leads
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from models import * 
from dataset import *
import traceback
from dataset import TestBeats
import time
try:
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath('D://BayesianECGI/Utilities/Codebase/')
except:
    print("MATLAB not imported in this machine.")


def DFBlock(A, reg_param, y, z, device, use_abs=True):
    
    '''DFBlock: Analytical block operation that solves the Tikhonov-proximal solution 

    Arguments:
        A -- Forward operator matrix
        reg_param -- regularization parameter
        y -- measurements
        z -- proximal point from whose distance the solution is penalized
        device -- Tensor.device
        use_abs -- to circumvent the negative square root, if you make sure that the reg_param always returns positive from optimization

    Returns:
        the solution to the inverse problem 
    '''
    eps = 1e-24
    if use_abs:
        A_tilde = torch.cat(
            [A, torch.sqrt(torch.abs(reg_param+eps))*torch.eye(A.shape[1]).to(device)], 0)
        b_tilde = torch.cat(
            [y, torch.sqrt(torch.abs(reg_param+eps))*z.to(device)], 0)
    else:
        A_tilde = torch.cat([A, torch.sqrt((reg_param+eps))
                                * torch.eye(A.shape[1]).to(device)], 0)
        b_tilde = torch.cat([y, torch.sqrt((reg_param+eps))*z.to(device)], 0)
    return torch.linalg.multi_dot([torch.linalg.inv(torch.mm(A_tilde.T, A_tilde)), A_tilde.T, b_tilde]).to(device)


def train_model(model, reg_params, loss_fn, train_dataloader, val_dataloader, LR, LR_reg, n_epoch, forward_model_A, device, use_abs, experiment_name, model_save_path, model_load_path=None):
    '''train_model Trains the model in HQS scheme.

    Arguments:
        model -- empty PyTorch model
        reg_params -- a Tensor of Pytorch params that contains the lambda's for DFBlocks
        loss_fn -- Loss function for training
        train_dataloader -- train dataloader that outputs heart and corresponding torso tensors 
        val_dataloader -- validation dataloader that outputs heart and corresponding torso tensors 
        LR -- Learning rate for the model
        LR_reg -- Learning rate for the regularization parameter
        n_epoch -- number of epochs
        forward_model_A -- Forward model A required for DFBlock
        device -- Torch tensor device (if you can somehow drop this, do it)
        use_abs -- Used this for keeping lambda positive, if you can find any other way change it. 
        experiment_name -- ID for experiments to save the results
        model_save_path -- path to save the result dictionary

    Keyword Arguments:
        model_load_path -- pretrained model path if you want to use (default: {None})

    Raises:
        ValueError: written for nans in the gradient computation 

    Returns:
        result dictionary containing the training metrics and the trained model 
    '''
    try:
        
        ## If a model load path is provided, load state dict
        if model_load_path:
            pretrained_results = torch.load(model_load_path)
            state_dict = pretrained_results['model']
            model.load_state_dict(state_dict, strict=False)
            
        optimizer1 = torch.optim.Adam(model.parameters(),lr=LR)
        optimizer2 = torch.optim.Adam([{'params':reg_params, 'lr':LR_reg}])

        scheduler1 = ReduceLROnPlateau(optimizer1,mode='min',patience=50,factor=0.5,verbose=True)

        train_loss_container = np.zeros(n_epoch)
        val_loss_container = np.zeros(n_epoch)
        train_re_container = np.zeros(n_epoch)
        val_re_container = np.zeros(n_epoch)
        train_cc_container = np.zeros(n_epoch)
        val_cc_container = np.zeros(n_epoch)
        reg_container = np.empty_like(reg_params.unsqueeze(axis=0).detach().cpu())
        
        ## Training starts
        for epoch in tqdm(range(n_epoch)):
            model.train()
            train_losses = np.zeros(len(train_dataloader))
            val_losses = np.zeros(len(val_dataloader))
            train_cc = np.zeros(len(train_dataloader))
            val_cc = np.zeros(len(val_dataloader))
            train_re = np.zeros(len(train_dataloader))
            val_re = np.zeros(len(val_dataloader))
            
            ## Looping on batches in one epoch for training 
            for batch_idx, (train_heart, train_torso) in enumerate(train_dataloader):
                
                (train_heart, train_torso) = (train_heart.squeeze(), train_torso.squeeze())
                z = DFBlock(forward_model_A, reg_params[0], train_torso.T,
                    torch.zeros_like(train_heart.T), device, use_abs).T # This corresponds to the Tikhonov solution
                train_heart = train_heart.unsqueeze(axis=1)

                # Train Unroll
                for reg_param in reg_params:
                    heart_hat = DFBlock(
                        forward_model_A, reg_param, train_torso.T, z.squeeze().T, device, use_abs)
                    z = model(heart_hat.T.unsqueeze(axis=1))
                    
                optimizer1.zero_grad()
                batch_loss = loss_fn(z, train_heart)
                
                if torch.sum(torch.isnan(batch_loss)) > 0:
                    raise ValueError
                
                train_losses[batch_idx] = batch_loss.item()
                mean_batch_cc = torch.mean(
                    correlation_coefficient(train_heart, z))
                train_cc[batch_idx] = mean_batch_cc.item()
                mean_batch_re = torch.mean(
                    relative_error(train_heart, z))
                train_re[batch_idx] = mean_batch_re.item()
                
                # batch_loss = mean_batch_re - mean_batch_cc
                batch_loss.backward()
                optimizer1.step()
                                
                ## Update the reg param after the model sees every sample in the epoch 
                if batch_idx == (len(train_dataloader)-1):
                    reg_params.grad = reg_params.grad / len(train_dataloader)
                    optimizer2.step()
                    optimizer2.zero_grad()
                    

                # Validation Unroll (Just compute the reconstructions and calculate losses)

            with torch.no_grad():
                model.eval()
                
                for batch_idx, (val_heart, val_torso) in enumerate(val_dataloader):

                    (val_heart, val_torso) = (val_heart.squeeze(), val_torso.squeeze())
                    z = DFBlock(forward_model_A, reg_params[0], val_torso.T,
                    torch.zeros_like(val_heart.T), device, use_abs).T
                    val_heart = val_heart.unsqueeze(axis=1)

                    # Val Unroll
                    for reg_param in reg_params:
                        heart_hat = DFBlock(
                            forward_model_A, reg_param, val_torso.T, z.squeeze().T, device, use_abs)
                        z = model(heart_hat.T.unsqueeze(axis=1))
                    val_loss = loss_fn(z, val_heart)
                    val_losses[batch_idx] = val_loss.item()
                    val_cc[batch_idx] = torch.mean(
                        correlation_coefficient(val_heart, z)).item()
                    val_re[batch_idx] = torch.mean(
                        relative_error(val_heart, z)).item()
                        
            val_cc_container[epoch] = np.mean(val_cc).item()
            val_re_container[epoch] = np.mean(val_re).item()
            train_cc_container[epoch] = np.mean(train_cc).item()
            train_re_container[epoch] = np.mean(train_re).item()
            train_loss_container[epoch] = np.mean(train_losses).item()
            val_loss_container[epoch] = np.mean(val_losses).item()
            reg_container = np.vstack((reg_container, reg_params.unsqueeze(axis=0).detach().cpu()))
            
            scheduler1.step(val_loss_container[epoch])

        save_model(model,reg_params,experiment_name, train_loss_container, val_loss_container, train_cc_container,
                val_cc_container, train_re_container, val_re_container, reg_container[1:,:], LR, LR_reg, model_save_path)
        
    except Exception as e:
        now = datetime.now()
        date_time = str(now.month) + "_" + str(now.day)
        error = traceback.format_exc()
        print(e)
        
    return model

def test_model(model, reg_params, test_data_list, forward_model_A, device, use_abs, batch_size=64):
    '''test_model: Model evaluation script with test data 

    Arguments:
        model -- PyTorch model to test
        reg_params -- PyTorch tensor that contains the regularization parameters for unrolling iterations 
        test_data_list -- a Python list that contains the dictionaries of the test data, refer to create_training_data notebook 
        forward_model_A -- Forward model operator
        device -- Tensor.device
        use_abs -- **see DFBLock**

    Keyword Arguments:
        batch_size -- batch size to run the in_data_batch_completer (default: {64})

    Returns:
        the results dictionary with evaluation results
    '''
    model.eval()
    results = {}
    for i,_ in enumerate(test_data_list):
        results[str(i)] = {}
    # Test data is given in the form of list of dictionaries, each of them contains heart,torso and badleads
    times = []
    for batch_idx, test_dict in tqdm(enumerate(test_data_list)):
        x = test_dict['x'].T.to(device)
        y = test_dict['y'].T.to(device)
        at = test_dict['at']
        paceloc = test_dict['paceloc']
        
        t,n = x.shape
        x_batched = torch.Tensor(in_data_batch_completer(x.T.cpu().numpy(),batch_size)).T
        y_batched = torch.Tensor(in_data_batch_completer(y.T.cpu().numpy(),batch_size)).T
        
        bad_leads = test_dict['badleads'].to(device)
        n = n-len(bad_leads)
        
        test_dataset = TestBeats(torso=y_batched, heart=x_batched, device=device)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        L_batch = test_dataloader.__len__()
        Reconstruction = torch.Tensor(np.zeros((x.shape[1],L_batch*batch_size))).to(device) # A tensor to dump the reconstructions until the inference is finished
        CCs, CCt, REt, REs = np.zeros((L_batch,batch_size)),np.zeros((L_batch,n)),np.zeros((L_batch,n)),np.zeros((L_batch,batch_size))
        
        t1 = time.time()
        for i,(test_heart, test_torso) in enumerate(test_dataloader):
            z = DFBlock(forward_model_A, reg_params[0], test_torso.T,
                torch.zeros_like(test_heart.T), device, use_abs).T # This corresponds to the Tikhonov solution
            test_heart = test_heart.unsqueeze(axis=1)

            # Test Unroll
            for reg_param in reg_params:
                heart_hat = DFBlock(
                    forward_model_A, reg_param, test_torso.T, z.squeeze().T, device, use_abs)
                z = model(heart_hat.T.unsqueeze(axis=1))
            Reconstruction[:,i*batch_size:(i+1)*batch_size] = z.squeeze().T
        t2 = time.time()
        times.append(t2-t1)
        
        # It periodically reconstructs the test data based on the batch_size, the last batch_size instance is reconstructed together. 
        if batch_size < t:
            Reconstruction[:,t-batch_size:t] = z.squeeze().T
            Reconstruction = Reconstruction[:,0:t].T
        else:
            Reconstruction = Reconstruction[:,0:t].T
            
            
        CCs, REs, CCt, REt = metrics_with_bad_leads(x, Reconstruction, bad_leads)
        AT_Results = eng.AT_and_LE(np.array(Reconstruction.T.detach().cpu()),paceloc.item(),np.asmatrix(at).T,bad_leads.cpu().numpy(),nargout=2)
        
        results[str(int(batch_idx))]['Rec'] = Reconstruction.detach().cpu().numpy()
        results[str(int(batch_idx))]['spat_cc'], results[str(int(batch_idx))]['spat_re'], \
        results[str(int(batch_idx))]['temp_cc'], results[str(int(batch_idx))]['temp_re'] = CCs, REs, CCt, REt
        results[str(int(batch_idx))]['time'] = t
        results[str(int(batch_idx))]['AT_CC'] = AT_Results[0]
        results[str(int(batch_idx))]['LE'] = AT_Results[1]
        results[str(int(batch_idx))]['AvgTime'] = np.mean(times)
 
    return results 

def test_model_dataloader(model, reg_params, test_dataloader, forward_model_A, device, use_abs):
    '''test_model_dataloader: Evaluating the model with data loaders, not used any more. 

    Arguments:
        see **test_model**

    Returns:
        see **test_model**
    '''
    model.eval()
    
    # Test data is given in the form of list of dictionaries, each of them contains heart,torso and badleads
    L = len(test_dataloader)
    REs = np.zeros(L)
    REt = np.zeros(L)
    CCt = np.zeros(L)
    CCs = np.zeros(L)
    
    for batch_idx, (test_heart, test_torso) in tqdm(enumerate(test_dataloader)):
        
        z = DFBlock(forward_model_A, reg_params[0], test_torso.T,
            torch.zeros_like(test_heart.T), device, use_abs).T # This corresponds to the Tikhonov solution
        test_heart = test_heart.unsqueeze(axis=1)

        # Test Unroll
        for reg_param in reg_params:
            heart_hat = DFBlock(
                forward_model_A, reg_param, test_torso.T, z.squeeze().T, device, use_abs) 
            z = model(heart_hat.T.unsqueeze(axis=1))

            
        spat_cc, spat_re, temp_cc, temp_re = four_metrics(test_heart,z)

        CCs[batch_idx] = torch.mean(spat_cc).item()
        CCt[batch_idx] = torch.mean(temp_cc).item()
        REs[batch_idx] = torch.mean(spat_re).item()
        REt[batch_idx] = torch.mean(temp_re).item()
                       
    return CCs,CCt,REs,REt 

def save_model(model, reg_params, experiment_name, train_loss, val_loss, train_cc, val_cc, train_re, val_re, reg_container, LR, LR_reg, path):
    '''save_model saving the model progress during the training

    Arguments:
        model -- see **train_model**
        reg_params -- last optimized regularization parameters 
        experiment_name -- see **train_model**
        train_loss -- training loss with loss_fn 
        val_loss -- validation loss with loss_fn 
        train_cc -- correlation coefficients for each epoch during training 
        val_cc -- correlation coefficients for each epoch during training 
        train_re -- correlation coefficients for each epoch during training 
        val_re -- correlation coefficients for each epoch during training 
        reg_container -- regularization parameters for the n_epoch iterations 
        LR -- learning rate for the model
        LR_reg -- learning rate for the regularization parameter
        path -- save path for the results dict 
    '''
    results = {}
    path = path + ".pt"
    
    # Find the keys that will be given arguments to model initializer
    essential_keys = [item for item in list(model.__dict__.keys()) if not (item.startswith('_') or item == 'training')]
    essential_dict = {}
    for item in essential_keys:
        essential_dict[item] = model.__dict__[item]
    results['essential_dict'] = essential_dict
    results["experiment"] = experiment_name
    results["model"] = model.state_dict()
    results["modelclass"] = model.__class__
    results['reg_params'] = reg_params
    results["train_loss"] = train_loss
    results["val_loss"] = val_loss
    results["train_cc"] = train_cc
    results["val_cc"] = val_cc
    results["train_re"] = train_re
    results["val_re"] = val_re
    results["reg_container"] = reg_container
    results['LR'] = LR
    results['LR_reg'] = LR_reg
    torch.save(results, path)
