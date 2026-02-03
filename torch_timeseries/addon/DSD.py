import time
import torch
import torch.nn as nn
import torch.nn.functional as F

#DSD

class DSD (nn.Module):
    """
    Args:
        nn (_type_): _description_
    """
    def __init__(self,  seq_len, pred_len, enc_in, lv=2, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.epsilon = 1e-8
        self.lv = lv        

        self.model_freq = nn.ModuleList([
            MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)
            for _ in range(lv)
        ])

        self.lorenz_curve_lin_len = max(seq_len, pred_len) // 2 + 1
        self.lorenz_curve_hidden = 32

        self.lorenz_curve_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.lorenz_curve_lin_len, self.lorenz_curve_hidden),
                nn.ReLU(),
                nn.Linear(self.lorenz_curve_hidden, 1),
                nn.Tanh()
            )
            for _ in range(lv)
        ])

    def main_freq_part(self, x):
        # start = time.time()
        B, L, C = x.shape
        xf_init = torch.fft.rfft(x, dim=1)                        #(B, F+1, C)

        filtered_list = []

        xf_curr = xf_init

        for i in range(self.lv):
            magnitude = xf_curr[:, 1:].abs()                      #(B, F, C)
            total_energy = magnitude.sum(dim=1, keepdim=True)     #(B, 1, C)

            sorted_magnitude_asc, indices_asc = torch.sort(magnitude, dim=1, descending=False)  # (B, F, C)
   
            # Gini coefficient
            magnitude4g = sorted_magnitude_asc.permute(0, 2, 1)       # (B, C, F)
            cumulative_values = torch.cumsum(magnitude4g, dim=2)      # (B, C, F)
            total_values = cumulative_values[:, :, -1].unsqueeze(2).clamp_min(1e-8)  # (B, C, 1)
            lorenz_curve = cumulative_values / total_values       #(B, C, F)
            gini = 1 - 2 * lorenz_curve.mean(dim=2)               #(B, C)

            pad_len = self.lorenz_curve_lin_len - lorenz_curve.size(-1)
            if pad_len > 0:
                lorenz_curve_pad = F.pad(lorenz_curve, pad=(pad_len, 0), mode='constant', value=0.0)
            else:
                lorenz_curve_pad = lorenz_curve[:, :, -self.lorenz_curve_lin_len:]
            g_mlp = self.lorenz_curve_mlp[i](lorenz_curve_pad)[:, :, -lorenz_curve.size(-1):].mean(dim=2)

            energy_threshold = gini + g_mlp
            energy_threshold = energy_threshold + (energy_threshold.clamp(0.0, 1.0) - energy_threshold).detach()

            sorted_mag_desc = torch.flip(sorted_magnitude_asc, dims=[1])   # (B, F, C)
            indices_desc = torch.flip(indices_asc, dims=[1])               # (B, F, C)
            cumulative_energy = torch.cumsum(sorted_mag_desc, dim=1)       # (B, F, C)

            # energy_threshold
            threshold = energy_threshold.view(B, 1, C) * total_energy  # (B, 1, C)
            threshold = threshold.expand_as(cumulative_energy)  # (B, F, C)
            sorted_filter = torch.sigmoid(threshold - cumulative_energy)  # (B, F, C)

            # filter
            filter_wo_dc = torch.zeros_like(sorted_filter).scatter(1, indices_desc, sorted_filter)
            dc_component = torch.ones((B, 1, C), device=filter_wo_dc.device)
            filt = torch.cat([dc_component, filter_wo_dc], dim=1)

            xf_filtered_i = xf_curr * filt
            x_filtered_i = torch.fft.irfft(xf_filtered_i, dim=1).real.float()      # (B, L, C)
            filtered_list.append(x_filtered_i)

            xf_curr = xf_curr * (1.0 - filt)
    
        x_res = torch.fft.irfft(xf_curr, dim=1).real.float() 
        # print(f"decompose take:{ time.time() - start} s")
        return x_res, filtered_list

    def loss(self, true):
        lf = F.mse_loss
        x_res_gt, gt_filtered_list = self.main_freq_part(true)
        loss_main = sum(lf(self.extracted_parts_pred[i], gt_filtered_list[i]) for i in range(self.lv))
        loss_res = lf(self.pred_residual, x_res_gt)
        return loss_main + loss_res
        
    def extract(self, x):
        B, L, C = x.shape
        residual, extracted_parts = self.main_freq_part(x)

        parts_pred = []
        x_tran = x.transpose(1, 2)                                   # (B, C, L)
        for i in range(self.lv):
            part = extracted_parts[i].transpose(1, 2)                # (B, C, L)
            pred = self.model_freq[i](part, x_tran).transpose(1, 2)  # (B, L, C)
            parts_pred.append(pred)

        self.extracted_parts_pred = parts_pred
        return residual


    def reconstruct(self, pred_res):
        self.pred_residual = pred_res
        extracted_sum = sum(self.extracted_parts_pred)
        output = self.pred_residual + extracted_sum 
        return output
    

    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.extract(batch_x)
        elif mode =='d':
            return self.reconstruct(batch_x)


class MLPfreq(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, hidden_freq=64, hidden_all=128):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        
        self.model_freq = nn.Sequential(
            nn.Linear(seq_len, hidden_freq),
            nn.ReLU(),
        )
        self.model_all = nn.Sequential(
            nn.Linear(hidden_freq + seq_len, hidden_all),
            nn.ReLU(),
            nn.Linear(hidden_all, pred_len)
        )

    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)
        
        
        
