import torch
from torch import nn
from metrics import correlation_coefficient, relative_error, four_metrics, metrics_with_bad_leads

def normalize_by_axis(batch,dim=-1):
    '''normalize_by_axis Does one zero normalization along the specified axis, when axis=-1
    the tensor is normalized in all dimensions

    Arguments:
        batch -- Tensor to normalize

    Keyword Arguments:
        dim -- normalization axis (default: {-1})

    Returns:
        The normalized tensor, peak to peak amplitude(s) and the minimum(s) for denormalization
    '''
    if dim != None:
        minimums,_ = torch.min(batch,dim=dim,keepdim=True)
        maximums,_ = torch.max(batch,dim=dim,keepdim=True)
        peak2peaks = maximums - minimums
    else:
        minimums = torch.min(batch)
        maximums = torch.max(batch)
        peak2peaks = maximums - minimums
    
    # When peak2peaks = 0, we get problems with nans
    # so we gotta filter where peak2peaks is not equal to zero
    # Easiest way to do so is to set nans to 0. 
    # This is not a get-around, it should be like that
    
    onezero_batch = (batch-minimums)/peak2peaks
    
    # Create a mask for NaN values
    nan_mask = torch.isnan(onezero_batch)

    # Replace NaN values with 0
    onezero_batch = torch.where(nan_mask, torch.tensor(0.0).to(onezero_batch.device), onezero_batch)
    
    return onezero_batch, peak2peaks, minimums

def denormalize(onezero_batch,peak2peaks, minimums):
    '''denormalize Recover one zero normalization with saved minimum and the peak2peak amplitude

    Arguments:
        onezero_batch -- normalized batch tensor 
        peak2peaks -- see **normalize_by_axis**
        minimums -- **normalize_by_axis**

    Returns:
        denormalized one-zero Tensor
    '''
    return onezero_batch * peak2peaks + minimums

def pad_input(inputs, skip):
    '''pad_input: function to deal with the pooling by 2 when odd size tensor is encountered

    Arguments:
        inputs -- input to decoder
        skip -- skip connection

    Returns:
        input's zero padded version to the size of skip
    '''
    feats_i = inputs.shape[-1]
    feats_s = skip.shape[-1]
    diff = (feats_s-feats_i)
    if diff:
        inputs = torch.nn.ConstantPad1d((0, diff), 0)(inputs)
    return inputs

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=6, padding="same")
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(in_c+out_c, out_c, kernel_size=6, padding="same")
        self.drop = nn.Dropout1d(p=0.1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = torch.cat([x,inputs],axis=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool1d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = pad_input(x, skip)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class old_ECGI_Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        #   Encoder
        self.e1 = encoder_block(1, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)
        #   Bottleneck
        self.b = ConvBlock(128, 128)
        #  Decoder
        self.d1 = decoder_block(128, 128)
        self.d2 = decoder_block(128, 64)
        self.d3 = decoder_block(64, 32)
        self.d4 = decoder_block(32, 16)
        #  Size matching 
        self.outputs = ConvBlock(17, 1)

    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        # Bottleneck
        b = self.b(p4)
        #  Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        # Residual skip connection
        concat = torch.torch.cat([inputs, d4], axis=1)
        #  Output
        outputs = self.outputs(concat)
        return outputs

class ECGI_Denoiser(nn.Module):
    def __init__(self, if_temp_block=False, batch_size=32):
        super().__init__()
        self.if_temp_block = if_temp_block
        self.batch_size = batch_size
        #   Encoder
        self.e1 = encoder_block(1, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)
        #   Bottleneck
        self.b = ConvBlock(128, 128)
        self.b_down = ConvBlock(128, 1)
        # Normally we have batches first, but we want to apply LSTM along batch dimension! If we squeeze it, batch_first does not matter
        self.temporal = torch.nn.LSTM(batch_size,batch_size,batch_first=False) 
        self.b_up = ConvBlock(1, 128)
        #  Decoder
        self.d1 = decoder_block(128, 128)
        self.d2 = decoder_block(128, 64)
        self.d3 = decoder_block(64, 32)
        self.d4 = decoder_block(32, 16)
        #  Size matching 
        self.outputs = ConvBlock(17, 1)

    def forward(self, inputs):
        # Normalize
        if self.if_temp_block:
            inputs, peak2peaks, minimums = normalize_by_axis(inputs, dim=None)
        else:
            inputs, peak2peaks, minimums = normalize_by_axis(inputs, dim=-1)
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        # Bottleneck
        if self.if_temp_block:
            # b is by [batch, channel, features]
            b  = self.b_down(p4)
            b, p2p, minimum = normalize_by_axis(b,dim=None)
            b,(_,_) = self.temporal(b.squeeze(1).T)
            b = denormalize(b,peak2peaks,minimums)
            b = self.b_up(b.T.unsqueeze(1))
        else:
            b = self.b(p4)
        #  Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        # Residual skip connection
        concat = torch.cat([inputs, d4], axis=1)
        #  Output
        outputs = self.outputs(concat)
        # Denormalize
        outputs = denormalize(outputs,peak2peaks,minimums)
        
        return outputs

class LSTM_UNet(nn.Module):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.unet = ECGI_Denoiser(if_temp_block=False, batch_size=self.batch_size)
        self.temporalstart = torch.nn.LSTM(batch_size,batch_size,batch_first=False) 
        self.temporalend = torch.nn.LSTM(batch_size,batch_size,batch_first=False) 
        
    def forward(self,inputs):
        # Normalize the whole batch 
        inputs, peak2peaks, minimums = normalize_by_axis(inputs, dim=None)
        
        # Feed to LSTM-UNET-LSTM network
        temp1,(h_start,c_start) = self.temporalstart(inputs.squeeze(1).T)
        spat1 = self.unet(temp1.T.unsqueeze(1))
        output,(_,_) = self.temporalend(spat1.squeeze(1).T,(h_start,c_start))
        
        # Denormalize
        output = denormalize(output,peak2peaks,minimums)
        return output.T.unsqueeze(1)
        
class plain_LSTM(nn.Module):
    def __init__(self,batch_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(batch_size,batch_size,batch_first=False) 
        
    def forward(self,inputs):
        inputs, peak2peaks, minimums = normalize_by_axis(inputs, dim=None)
        temp1,(_,_) = self.lstm(inputs.squeeze(1).T)
        output = denormalize(temp1,peak2peaks,minimums)
        return output.T.unsqueeze(1)

class ECGI_Denoiser_Auto(nn.Module):
    
    def __init__(self, levels, device):
        super().__init__()
        self.levels = levels
        self.device = device
        # Encoder Loop
        for i in range(levels):
            print('Encoder:'+str(i+1),2**(2*i), 2**(2*(i+1)),'Decoder:'+str(i+1),2**(2*(levels-i)), 2**(2*(levels-1-i)))
            self.__dict__['e'+str(i+1)] = encoder_block(2**(2*i), 2**(2*(i+1))).to(self.device)
            self.__dict__['d'+str(i+1)] = decoder_block(2**(2*(levels-i)), 2**(2*(levels-1-i))).to(self.device)
        self.b = ConvBlock(2**(2*levels),2**(2*levels)).to(self.device)
        self.outputs = ConvBlock(2, 1).to(self.device)

        
    def forward(self, inputs):
        
        container = {}
        
        for i in range(self.levels):
            if i == 0:
                container['s'+str(i+1)], container['p'+str(i+1)] = self.__dict__['e'+str(i+1)](inputs)
            else:
                container['s'+str(i+1)], container['p'+str(i+1)] = self.__dict__['e'+str(i+1)](container['p'+str(i)])
                
        container['b'] = self.b(container['p'+str(i+1)])
        
        for i in range(self.levels):
            if i == 0:
                container['d'+str(i+1)] = self.__dict__['d'+str(i+1)] (container['b'],container['s'+str(self.levels-i-1)])
            elif i == self.levels-1 :
                container['d'+str(i+1)] = self.__dict__['d'+str(i+1)] (container['d'+str(i)],inputs)
            else:
                container['d'+str(i+1)] = self.__dict__['d'+str(i+1)] (container['d'+str(i)],container['s'+str(self.levels-i-1)])
                
        concat = torch.cat([inputs, container['d'+str(i+1)]], axis=1)
        #  Output
        outputs = self.outputs(concat)
        del container
        return outputs
