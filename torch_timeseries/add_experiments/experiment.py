import datetime
import json
import os
import random
import time
import hashlib
from prettytable import PrettyTable
import sys
####
from typing import Dict, List, Type, Union
import numpy as np
import torch
from torchmetrics import MeanSquaredError, MetricCollection, MeanAbsoluteError, MeanAbsolutePercentageError
from tqdm import tqdm
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.dataloader import (
    ChunkSequenceTimefeatureDataLoader, ETTHLoader, ETTMLoader
)
from torch.nn import MSELoss, L1Loss

from torch.optim import Optimizer, Adam
from dataclasses import asdict, dataclass, field
from torch_timeseries.addon import *

from torch_timeseries.nn.metric import R2, Corr, TrendAcc, RMSE, compute_corr, compute_r2
from torch_timeseries.metrics.masked_mape import MaskedMAPE
from torch_timeseries.add_experiments.Model import Model
from torch_timeseries.utils.early_stopping import EarlyStopping
import json
import codecs

def kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2):
    k = mu1.size(1)
    
    Sigma2_inv = torch.linalg.inv(Sigma2)
    
    tr_term = torch.einsum('bij,bjk->bi', Sigma2_inv, Sigma1)
    
    mu_diff = mu2 - mu1
    mu_term = torch.einsum('bi,bij,bj->b', mu_diff, Sigma2_inv, mu_diff)
    
    det_term = torch.log(torch.linalg.det(Sigma2) / torch.linalg.det(Sigma1))
    
    kl_div = 0.5 * (tr_term + mu_term - k + det_term)
    
    return kl_div.sum()

@dataclass
class ResultRelatedSettings:
    dataset_type: str
    optm_type: str = "Adam"
    model_type: str = ""
    scaler_type: str = "StandarScaler"
    loss_func_type: str = "mse"
    batch_size: int = 32
    lr: float = 0.0003
    l2_weight_decay: float = 0.0005
    epochs: int = 100

    horizon: int = 3
    windows: int = 384
    pred_len: int = 1

    patience: int = 5
    max_grad_norm: float = 5.0
    invtrans_loss: bool = False
    
    add_type : str = ''
    split_type : str = "custom"

@dataclass
class Settings(ResultRelatedSettings):
    data_path: str = "./data"
    device: str = "cuda:0"
    num_worker: int = 8
    save_dir: str = "./results"
    experiment_label: str = str(int(time.time()))
    lv: int = 2

def count_parameters(model, print_fun=print):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print_fun(table)
    print_fun(f"Total Trainable Params: {total_params}")
    return total_params


class AddExperiment(Settings):   
    def _run_print(self, *args, **kwargs):
        time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
        
        print(*args, **kwargs)
        if hasattr(self, "run_setuped") and getattr(self, "run_setuped") is True:
            with open(os.path.join(self.run_save_dir, 'output.log'), 'a+') as f:
                print(time, *args, flush=True, file=f)

    def _init_loss_func(self):
        loss_func_map = {"mse": MSELoss, "l1": L1Loss}
        self.loss_func = loss_func_map[self.loss_func_type]()

    def _init_metrics(self):
        if self.pred_len == 1:
            self.metrics = MetricCollection(
                metrics={
                    "r2": R2(self.dataset.num_features, multioutput="uniform_average"),
                    "r2_weighted": R2(
                        self.dataset.num_features, multioutput="variance_weighted"
                    ),
                    "mse": MeanSquaredError(),
                    "corr": Corr(),
                    "mae": MeanAbsoluteError(),
                }
            )
        elif self.pred_len > 1:
            self.metrics = MetricCollection(
                metrics={
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError(),
                }
            )

        self.metrics.to(self.device)

    
    
    @property
    def result_related_config(self):
        ident = asdict(self)
        keys_to_remove = [
            "data_path",
            "device",
            "num_worker",
            "save_dir",
            "experiment_label",
        ]
        for key in keys_to_remove:
            if key in ident:
                del ident[key]
        return ident

    def _run_identifier(self, seed) -> str:
        ident = self.result_related_config
        ident["seed"] = seed
        # only influence the evluation result, not included here
        ident['invtrans_loss'] = False

        ident_md5 = hashlib.md5(
            json.dumps(ident, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return str(ident_md5)

    def _init_data_loader(self):
        self.dataset : TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        if self.split_type == "popular" and self.dataset_type[0:3] == "ETT":
            if self.dataset_type[0:4] == "ETTh":
                self.dataloader = ETTHLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=True,
                    freq="h",
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                )
            elif  self.dataset_type[0:4] == "ETTm":
                self.dataloader = ETTMLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=True,
                    freq="h",
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                )
        else:
            self.dataloader = ChunkSequenceTimefeatureDataLoader(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=False,
                shuffle_train=True,
                freq="h",
                batch_size=self.batch_size,
                train_ratio=0.7,
                val_ratio=0.2,
                num_worker=self.num_worker,
            )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = self.dataloader.train_size
        self.val_steps = self.dataloader.val_size
        self.test_steps = self.dataloader.test_size

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")

    def _init_sep_optimizer(self):
        self.a_model_optim = Adam(
            self.model.am.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        
        self.f_model_optim = Adam(
            self.model.fm.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.f_model_optim, T_max=self.epochs
        )

    def _init_optimizer(self):
        self.model_optim = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optim, T_max=self.epochs
        )

    def _init_a_model(self):
        print(f"using {self.add_type} as add-on model.,")
        Ty = self._parse_type(self.add_type) 
        if self.add_type == 'No':
            self.a_model : torch.nn.Module = No()
        else:
            self.a_model : torch.nn.Module = Ty(self.windows, self.pred_len, self.dataset.num_features, self.lv)
        self.a_model = self.a_model.to(self.device)
        

    def _init_f_model(self) -> torch.nn.Module:
        self.f_model = None
        raise NotImplementedError()

        
    def _init_model(self):
        self.model =   Model(self.model_type, self.f_model,self.a_model).to(self.device)

    def _setup(self):
        
        # init data loader
        self._init_data_loader()

        # init metrics
        self._init_metrics()

        # init loss function based on given loss func type
        self._init_loss_func()

        self.current_epochs = 0
        self.current_run = 0

        self.setuped = True

    def _setup_run(self, seed):
        # setup experiment  only once
        if not hasattr(self, "setuped"):
            self._setup()
        # setup torch and numpy random seed
        self.reproducible(seed)
        # init model, optimizer and loss function

        self._init_a_model()
        
        self._init_f_model()
        
        self._init_model()

        self._init_optimizer()
        self.current_epoch = 0
        ## changed
        self.run_save_dir = os.path.join(
            self.save_dir,
            "runs",
            self.model_type,
            self.dataset_type,
            f"w{self.windows}h{self.horizon}s{self.pred_len}",
        )
        '''
        self.run_save_dir = os.path.join(
            self.save_dir,
            "runs",
            self.model_type,
            self.dataset_type,
            f"w{self.windows}h{self.horizon}s{self.pred_len}",
            self._run_identifier(seed),
        )
        '''
        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "best_model.pth"
        )

        self.run_checkpoint_filepath = os.path.join(
            self.run_save_dir, "run_checkpoint.pth"
        )

        self.early_stopper = EarlyStopping(
            self.patience, verbose=True, path=self.best_checkpoint_filepath
        )
        self.run_setuped = True
        
        
        
    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            raise RuntimeError(f"{str_or_type} should be string or type")


    def _save(self, seed=0):
        self.checkpoint_path = os.path.join(
            self.save_dir, f"{self.model_type}/{self.dataset_type}"
        )
        self.checkpoint_filepath = os.path.join(
            self.checkpoint_path, f"{self.experiment_label}.pth"
        )

        if not os.path.exists(self.checkpoint_path):

            os.makedirs(self.checkpoint_path)
            print(f"Directory '{self.checkpoint_path}' created successfully.")

        self.app_state = {
            "model": self.model,
            "optimizer": self.model_optim,
        }

        self.app_state.update(asdict(self))

        # now only save the newest state
        torch.save(self.app_state, f"{self.checkpoint_filepath}")

    def get_run_state(self):
        run_state = {
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }
        return run_state

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = self.get_run_state()


        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")

    def reproducible(self, seed):
        # for reproducibility
        # torch.set_default_dtype(torch.float32)
        print("torch.get_default_dtype()", torch.get_default_dtype())
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _process_input(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, dec_inp=None, dec_input_date=None):
        # inputs:
            # batch_x:  (B, T, N)
            # batch_y:  (B, Steps,T)
            # batch_x_date_enc:  (B, T, N)
            # batch_y_date_enc:  (B, T, Steps)
        # outputs:
            # pred: (B, O, N)
        raise NotImplementedError()
        # batch_x = batch_x.transpose(1,2) # (B, N, T)
        # batch_x_date_enc = batch_x_date_enc.transpose(1,2) # (B, N, T)
        # pred = self.model(batch_x) # (B, O, N)
        # pred = pred.transpose(1,2) # (B, O, N)
        # return pred
    


    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
            # batch_x:  (B, T, N)
            # batch_y:  (B, Steps,T)
            # batch_x_date_enc:  (B, T, N)
            # batch_y_date_enc:  (B, T, Steps)
            
        # outputs:
            # pred: (B, O, N)
            # label:  (B,O,N)
        raise NotImplementedError()
        # label_len = 48
        # dec_inp_pred = torch.zeros(
        #     [batch_x.size(0), self.pred_len, self.dataset.num_features]
        # ).to(self.device)
        # dec_inp_label = batch_x[:, label_len :, :].to(self.device)

        # dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        # dec_inp_date_enc = torch.cat(
        #     [batch_x_date_enc[:, label_len :, :], batch_y_date_enc], dim=1
        # )
        
        # pred = self._process_input(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, dec_inp, dec_inp_date_enc)

        # return pred, batch_y # (B, O, N), (B, O, N)

    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()
        
        length = 0
        if dataloader is self.train_loader:
            length = self.dataloader.train_size
        elif dataloader is self.val_loader:
            length = self.dataloader.val_size
        elif dataloader is self.test_loader:
            length = self.dataloader.test_size

        with torch.no_grad():
            with tqdm(total=length,position=0, leave=True) as progress_bar:
                for batch_x, batch_y,batch_origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                    batch_size = batch_x.size(0)
                    batch_x = batch_x.to(self.device, dtype=torch.float32)
                    batch_y = batch_y.to(self.device, dtype=torch.float32)
                    batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                    batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                    preds, truths = self._process_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    ) # preds : (B, O, N) truths: (B O N)
                    # the result should be the same
                    # self.metrics.update(preds.view(batch_size, -1), truths.view(batch_size, -1))
                    batch_origin_y = batch_origin_y.to(self.device)
                    if self.invtrans_loss:
                        preds = self.scaler.inverse_transform(preds)
                        truths = batch_origin_y

                    if self.pred_len == 1:
                        self.metrics.update(preds.view(batch_size, -1), truths.view(batch_size, -1))
                    else:
                        self.metrics.update(preds.contiguous(), truths.contiguous())

                    progress_bar.update(batch_x.shape[0])
           
            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result

    
    
    def _test(self) -> Dict[str, float]:
        print("Testing .... ")
        test_result = self._evaluate(self.test_loader)
        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        print("Evaluating .... ")
        val_result = self._evaluate(self.val_loader)
        self._run_print(f"vali_results: {val_result}")
        return val_result

    def _train(self):
        with torch.enable_grad(), tqdm(total=self.train_steps,position=0, leave=True) as progress_bar:
            self.model.train()
            times = []
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device).float()
                self.model_optim.zero_grad()
                bs = batch_x.size(0)
                batch_x = batch_x.to(self.device, dtype=torch.float32).float()
                batch_y = batch_y.to(self.device, dtype=torch.float32).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                start = time.time()

                pred, true = self._process_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred).float()
                    true = origin_y
                
                loss = self.loss_func(pred, true) + self.model.am.loss(true)
                    
                if self.scaler_type is NoScaler:
                    loss = 10000*self.loss_func(pred, true) + 10000*self.model.am.loss(true)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

                self.model_optim.step()
                
                
                end = time.time()
                
                
                times.append(end-start)
                
            print(f"average iter: {np.mean(times)*1000}ms")
                
            return train_loss
        
        
    def _sep_train(self):
        with torch.enable_grad(), tqdm(total=self.train_steps,position=0, leave=True) as progress_bar:
            self.model.train()
            times = []
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device)
                self.a_model_optim.zero_grad()
                self.f_model_optim.zero_grad()
                bs = batch_x.size(0)
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                start = time.time()

                pred, true = self._process_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                
                loss = self.loss_func(pred, true)
                lossn = self.model.am.loss(true)

                
                loss.backward(retain_graph=True)
                lossn.backward(retain_graph=True)

                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.f_model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

                self.a_model_optim.step()
                self.f_model_optim.step()
                
                
                end = time.time()
                times.append(end-start)
                
            print("average iter: {}ms", np.mean(times)*1000)
                
            return train_loss

    def _check_run_exist(self, seed: str):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        else:
            print(f"result directory exists: {self.run_save_dir}")
        with codecs.open(os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

        exists = os.path.exists(self.run_checkpoint_filepath)
        return exists

    def _resume_run(self, seed):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        run_checkpoint_filepath = os.path.join(self.run_save_dir, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)


        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])
       
    def _resume_from(self, path):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        run_checkpoint_filepath = os.path.join(path, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

       
    def _load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.device))


    def single_step_forecast(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self.experiment_label = f"{self.model_type}-w{self.windows}-h{self.horizon}"
    
    
    def run_more_epochs(self, seed=42, epoches=200) -> Dict[str, float]:
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self.epoches = epoches

        self._run_print(f"run : {self.current_run} in seed: {seed}")
        
        self.model_parameters_num = self.count_parameters(self._run_print)
        self._run_print(
            f"model parameters: {self.model_parameters_num}"
        )

        epoch_time = time.time()
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"loss no decreased for {self.patience} epochs,  early stopping ...."
                )
                break

            self.reproducible(seed + self.current_epoch)

            train_losses =  self._train()

            self._run_print(
                "Epoch: {} cost time: {}".format(
                    self.current_epoch + 1, time.time() - epoch_time
                )
            )
            self._run_print(
                f"Traininng loss : {np.mean(train_losses)}"
            )
            
            val_result = self._val()

            # test
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result[self.loss_func_type], model=self.model)
            
            self._save_run_check_point(seed)

            self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        self.run_setuped = False

        return best_test_result
        

    def count_parameters(self, print_fun):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print_fun(table)
        print_fun(f"Total Trainable Params: {total_params}")
        return total_params

    
    def run(self, seed=42) -> Dict[str, float]:
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return {}

        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")
        
        self.model_parameters_num = self.count_parameters(self._run_print)
        self._run_print(
            f"model parameters: {self.model_parameters_num}"
        )

        epoch_time = time.time()
        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"loss no decreased for {self.patience} epochs,  early stopping ...."
                )
                break

            self.reproducible(seed + self.current_epoch)

            train_losses =  self._train()

            self._run_print(
                "Epoch: {} cost time: {}".format(
                    self.current_epoch + 1, time.time() - epoch_time
                )
            )
            self._run_print(
                f"Traininng loss : {np.mean(train_losses)}"
            )
            
            # evaluate on val set
            result = self._val()
            # test
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(result[self.loss_func_type], model=self.model)
            
            self._save_run_check_point(seed)

            self.scheduler.step()
            

        self._load_best_model()
        best_test_result = self._test()
        self.run_setuped = False

        #End train
        with open(os.path.join(self.run_save_dir, 'output.log'), 'a+') as f:
            try:
                print(self.model.am.weight, flush=True, file=f)
                print(self.model.am.weight.mean().item(), flush=True, file=f)
            except: pass
        ################
        ##saving csv 


        import pandas as pd

        result_dir = os.path.join(self.save_dir,"runs",self.model_type)
        csv_path = os.path.join(result_dir, str(self.model_type)+"_result.csv")

        if os.path.exists(csv_path):
            result_df = pd.read_csv(csv_path, index_col=0)

        else:
            result_df = pd.DataFrame()

        #add col: model+metric
        model_name = str(self.model_type)+str(self.add_type)
        if model_name+list(best_test_result.keys())[0] not in result_df.columns:
            for met in list(best_test_result.keys()):
                result_df = result_df.assign(**{f"{model_name}{met}": -1})

        #add ind: data+pred_len
        ind = str(self.pred_len)+'_'+str(self.dataset_type)
        if ind not in result_df.index:
            result_df.loc[ind] = [-1] * len(result_df.columns)

        #value
        for met in list(best_test_result.keys()):
           result_df.loc[ind, f"{model_name}{met}"] = best_test_result[met]

        result_df.to_csv(csv_path)

        #########################################

        return best_test_result

    def dp_run(self, seed=42, device_ids: List[int] = [0, 2, 4, 6], output_device=0):
        self._setup_dp_run(seed, device_ids, output_device)
        print(f"run : {self.current_run} in seed: {seed}")
        print(
            f"model parameters: {sum([p.nelement() for p in self.model.parameters()])}"
        )
        epoch_time = time.time()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self._train()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # evaluate on vali set
            self._val()

            self._save(seed=seed)

        return self._test()

    def runs(self, seeds: List[int] = [42,43,44]):
        if hasattr(self, "finished") and self.finished is True:
            print("Experiment finished!!!")
            return 
        
        results = []
        for i, seed in enumerate(seeds):
            self.current_run = i
            torch.cuda.empty_cache()
            result = self.run(seed=seed)
            torch.cuda.empty_cache()

            results.append(result)

        df = pd.DataFrame(results)
        self.metric_mean_std = df.agg(["mean", "std"]).T
        print(
            self.metric_mean_std.apply(
                lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
            )
        )


def main():
    exp = Experiment(
        dataset_type="ETTm1",
        data_path="./data",
        optm_type="Adam",
        model_type="DLinear",
        batch_size=32,
        device="cuda:0",
        windows=10,
        epochs=1,
        lr=0.001,
        pred_len=3,
        scaler_type="MaxAbsScaler",
    )

if __name__ == "__main__":
    main()
