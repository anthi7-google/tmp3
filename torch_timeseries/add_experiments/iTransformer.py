
import fire
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_timeseries.add_experiments.experiment import AddExperiment
import torch.nn as nn
from dataclasses import asdict,dataclass
from torch_timeseries.models import iTransformer

from dataclasses import dataclass, asdict


@dataclass
class iTransformerExperiment(AddExperiment):
    model_type: str = "iTransformer"

    individual : bool = False

    lr: float = 0.0001

    def _init_f_model(self):
        self.label_len = self.windows

        self.f_model = iTransformer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            #e_layers=2,
            #n_heads=8,
            #d_model=128,
            #d_ff=128,
            #dropout=0.1,
        )
        self.f_model = self.f_model.to(self.device)

    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, self.label_len :, :], batch_y_date_enc], dim=1
        )
        
        batch_x , dec_inp = self.model.preprocess(batch_x, dec_inp=dec_inp) # (B, T, N)   # (B,L,N)
        
        pred = self.model.fm(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)
        
        pred = self.model.postprocess(pred)

        return pred, batch_y # (B, O, N), (B, O, N)



def main():
    exp = iTransformerExperiment(
        dataset_type="DummyContinuous",
        data_path="./data",
        add_type='No',  
        optm_type="Adam",
        batch_size=128,
        device="cuda:1",
        windows=96,
        pred_len=96,
        horizon=1,
        epochs=100,
    )
        
    exp.run()
    
    
def cli():
    fire.Fire(iTransformerExperiment)
    

if __name__ == "__main__":
    cli()
