import torch
import torch.nn as nn

from torch_timeseries.addon import *


class Model(nn.Module):
    def __init__(self, f_model_type, forecast_model: nn.Module, addon_model: nn.Module):
        super().__init__()
        self.f_model_type = f_model_type
        self.fm = forecast_model
        self.am = addon_model

    
    def preprocess(self, batch_x, batch_x_enc=None, dec_inp=None,dec_inp_enc=None):
        # normalize
        # input: B T N
        # output: B, T, N
        dec_inp = dec_inp
        if  isinstance(self.am, No):
            pass
        else:
            batch_x = self.am(batch_x)
        
        return batch_x, dec_inp
            
    def postprocess(self, pred):
        if isinstance(self.am, No):
            pass
        else:
            pred = self.am(pred, 'd')
        
        return pred
    
    
    def forward(self, batch_x, batch_x_enc=None, dec_inp=None,dec_inp_enc=None):
        
        if 'former' in self.f_model_type:
            pred = self.fm(batch_x, batch_x_enc, dec_inp, dec_inp_enc)
        else:
            pred = self.fm(batch_x)

        return pred

