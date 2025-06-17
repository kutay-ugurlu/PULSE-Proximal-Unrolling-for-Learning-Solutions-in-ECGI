import torch
from torch.linalg import vector_norm


def correlation_coefficient(a, b):
    ## Assumes batch-first ordered inputs
    a = torch.squeeze(a,1)
    b = torch.squeeze(b,1)
    features = a.shape[-1]
    a = a - torch.mean(a, axis=1, keepdim=True)
    b = b - torch.mean(b, axis=1, keepdim=True)
    var_a = torch.var(a, axis=1, keepdim=True)
    var_b = torch.var(b, axis=1, keepdim=True)
    den = torch.sqrt(torch.multiply(var_a, var_b))
    num = torch.sum(torch.multiply(a, b), axis=1, keepdim=True)
    return torch.divide(num, den*(features-1))

def relative_error(x_true, x_hat):
    return torch.div(vector_norm(x_true-x_hat, dim=-1), vector_norm(x_true, dim=-1))

def metrics_with_bad_leads(a,b,bad_leads):
    
    features = a.shape[-1]
    valid_leads = torch.tensor(list({int(i) for i in range(features)}.difference(set(bad_leads.tolist()))),dtype=torch.int64)
    temp_cc = correlation_coefficient(a.T,b.T)
    spat_cc = correlation_coefficient(a[...,valid_leads],b[...,valid_leads])
    temp_cc = temp_cc[valid_leads]
    temp_re = relative_error(a.T,b.T)
    spat_re = relative_error(a[...,valid_leads],b[...,valid_leads])
    temp_re = temp_re[valid_leads]
    
    return spat_cc.detach().cpu().numpy(), spat_re.detach().cpu().numpy(), temp_cc.detach().cpu().numpy(), temp_re.detach().cpu().numpy()

def four_metrics(a,b):
    
    features = a.shape[-1]
    temp_cc = correlation_coefficient(a.T,b.T)
    spat_cc = correlation_coefficient(a,b)
    temp_re = relative_error(a.T,b.T)
    spat_re = relative_error(a,b)
    
    return spat_cc, spat_re, temp_cc, temp_re
