# coding=utf-8
import argparse
from re import X

import CustomOT as ot
import numpy as np
from responses import target
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from data_processor import MyDataset
from models import *
from utils import *
import random


def cal_wass(x_0, x_1, rep_0, rep_1, hparams):

    dist = ot.dist(rep_0, rep_1) 
    c_x_0 = ot.dist(x_0, x_0)
    c_rep_0 = ot.dist(rep_0, rep_0)
    c_x_1 = ot.dist(x_1, x_1)
    c_rep_1 = ot.dist(rep_1, rep_1)
    
    beta = hparams.get('beta')
    loss = 0
    rt_0 = torch.sqrt(torch.sum(torch.pow((c_x_0 - c_rep_0), 2)) / c_rep_0.size(0)**2)
    rt_1 = torch.sqrt(torch.sum(torch.pow((c_x_1 - c_rep_1), 2)) / c_rep_1.size(0)**2)
    fgw_loss, T = ot.gromov.fused_gromov_wasserstein2_our(
        dist,
        c_x_0,
        c_x_1,
        c_rep_0,
        c_rep_1,
        beta=beta,
        loss_fun='square_loss'
    )

    p = torch.ones(len(rep_0), device=device) / len(rep_0)
    q = torch.ones(len(rep_1), device=device) / len(rep_1)
    constC_1, hC1_1, hC2_1 = gromov_matrix(c_rep_0, c_rep_1, p, q, device)
    constC_2, hC1_2, hC2_2 = gromov_matrix(c_x_0, c_rep_1, p, q, device)
    constC_3, hC1_3, hC2_3 = gromov_matrix(c_rep_0, c_x_1, p, q, device)

    gw1 = torch.sum((constC_1 - (torch.mm(torch.mm(hC1_1, T), hC2_1.T))) * T)
    gw2 = torch.sum((constC_2 - (torch.mm(torch.mm(hC1_2, T), hC2_2.T))) * T)
    gw3 = torch.sum((constC_3 - (torch.mm(torch.mm(hC1_3, T), hC2_3.T))) * T)
    w = torch.sum(T * dist)

    fgw = 1.0 / beta * torch.sqrt((1-beta) * w + beta * gw1)
    loss += torch.pow(rt_0 + fgw, 2) - gw2
    loss += torch.pow(rt_1 + fgw, 2) - gw3
    
    return loss


class BaseEstimator:

    def __init__(self, hparams={}):
        data_name = hparams.get('data')
        print("Current data:", data_name)        
        self.train_set = MyDataset(f"Datasets/{data_name}/train.csv", data_name)
        self.traineval_set = MyDataset(f"Datasets/{data_name}/traineval.csv", data_name)
        self.eval_set = MyDataset(f"Datasets/{data_name}/eval.csv", data_name)
        self.test_set = MyDataset(f"Datasets/{data_name}/test.csv", data_name)

        self.device = torch.device(hparams.get('device'))
        if hparams['treat_weight'] == 0:
            self.train_loader = DataLoader(self.train_set, batch_size=hparams.get('batchSize'), drop_last=True)
        else:
            self.train_loader = DataLoader(self.train_set, batch_size=hparams.get('batchSize'), sampler=self.train_set.get_sampler(hparams['treat_weight']), drop_last=True)
        self.traineval_data = DataLoader(self.traineval_set, batch_size=256)  # for test in-sample metric
        self.eval_data = DataLoader(self.eval_set, batch_size=256)
        self.test_data = DataLoader(self.test_set, batch_size=256)

        self.init_model(hparams)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        random.seed(seed)
        np.random.seed(seed)
        
    def init_model(self, hparams):
        self.set_seed(hparams['seed'])
        
        self.train_metric = {
             "mae_ate": np.array([]),
             "mae_att": np.array([]),
             "pehe": np.array([]),
             "r2_f": np.array([]),
             "rmse_f": np.array([]),
             "r2_cf": np.array([]),
             "rmse_cf": np.array([]),
             "auuc": np.array([]),
             "rauuc": np.array([])}
        self.eval_metric = deepcopy(self.train_metric)
        self.test_metric = deepcopy(self.train_metric)

        self.train_best_metric = {
             "mae_ate": None,
             "mae_att": None,
             "pehe": None,
             "r2_f": None,
             "rmse_f": None,
             "r2_cf": None,
             "rmse_cf": None,
             "auuc": None,
             "rauuc": None,}
        self.eval_best_metric = deepcopy(self.train_best_metric)
        self.eval_best_metric['r2_f'] = -10  
        self.eval_best_metric["pehe"] = 100
        self.eval_best_metric['auuc'] = 0
        self.loss_metric = {'loss': np.array([]), 'loss_f': np.array([]), 'loss_c': np.array([])}

        self.epochs = hparams.get('epoch', 200)
        self.model = GMCFR(self.train_set.x_dim, hparams).to(self.device)
        
        self.criterion = torch.nn.MSELoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.get('lr', 1e-3), weight_decay=hparams.get('l2_reg', 1e-4))
        self.hparams = hparams
        self.epoch = 0    

    def fit(self):
        self.init_model(self.hparams)
        
        iter_num = 0
        stop_epoch = 0 
        for epoch in tqdm(range(1, self.epochs)):
            self.epoch = epoch
            self.model.train()
            for batch_idx, data in enumerate(self.train_loader):
                self.model.zero_grad()
                data = data.to(self.device)

                # idx = np.arange(5, 30)
                # new_idx = np.concatenate((idx, [0]))
                _x, _xt, _t, _yf, _ = data[:, :-5], data[:, :-4], data[:, -5], data[:, -4], data[:, -3] 

                _x_0 = _x[_t == 0]
                _x_1 = _x[_t == 1]
                _pred_f = self.model(_xt)
                _loss_fit = self.criterion(_pred_f.view(-1), _yf.view(-1))
                
                _loss_wass = 0

                wass_indicator = (self.hparams['model'] == ['gmcfr'] and epoch > self.hparams['pretrain_epoch'] and len(_t.unique()) > 1)
                if wass_indicator: 
                    if self.hparams['model'] == 'gmcfr':
                        _loss_wass = cal_wass(x_0=_x_0,
                                            x_1=_x_1,
                                            rep_0=self.model.rep_0,
                                            rep_1=self.model.rep_1,
                                            hparams=self.hparams)
                _loss = _loss_fit + self.hparams['lambda'] * _loss_wass
                _loss.backward()
                self.optimizer.step()
                _loss_wass = _loss_wass.item() if wass_indicator else 0
                iter_num += 1
  

            _train_metric = self.evaluation(data='train')
            self.train_metric = metric_update(self.train_metric, _train_metric, self.epoch)
            
            _eval_metric = self.evaluation(data='eval')
            self.eval_metric = metric_update(self.eval_metric, _eval_metric, self.epoch)
        
            if abs(_eval_metric['pehe']) < abs(self.eval_best_metric['pehe']):
                self.eval_best_metric = _eval_metric
                self.train_best_metric = self.evaluation(data='train')
                self.test_best_metric = self.evaluation(data='eval')
                stop_epoch = 0
                print(self.eval_best_metric)
            else:
                stop_epoch += 1
            if stop_epoch >= self.hparams['stop_epoch']:
                print(f'Early stop at epoch {self.epoch}')
                break

            self.epoch += 1
        
            
    def predict(self, dataloader):
        """

        :param dataloader
        :return: np.array, shape: (#sample)
        """
        self.model.eval()
        pred_0 = torch.tensor([], device=self.device)
        pred_1, yf, ycf, t, mu0, mu1 = deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0), deepcopy(pred_0),

        for data in dataloader:
            data = data.to(self.device)


            _x, _xt, _t, _yf, _ycf, _mu_0, _mu_1 = data[:, :-5], data[:, :-4], data[:, [-5]], data[:, -4], data[:, -3], data[:, -2], data[:, -1]
            
            _x_0 = torch.cat([_x, torch.zeros_like((_t), device=self.device)], dim=-1)
            _x_1 = torch.cat([_x, torch.ones_like((_t), device=self.device)], dim=-1)
        
            _pred_0 = self.model(_x_0).reshape([-1])
            _pred_1 = self.model(_x_1).reshape([-1])

            pred_0 = torch.cat([pred_0, _pred_0], axis=-1)
            pred_1 = torch.cat([pred_1, _pred_1], axis=-1)
            yf = torch.cat([yf, _yf], axis=-1)
            ycf = torch.cat([ycf, _ycf], axis=-1)
            
            mu0 = torch.cat([mu0, _mu_0], axis=-1)
            mu1 = torch.cat([mu1, _mu_1], axis=-1)
            
            
            t = torch.cat([t, _t.reshape([-1])], axis=-1)

        pred_0 = pred_0.detach().cpu().numpy()
        pred_1 = pred_1.detach().cpu().numpy()
        yf = yf.cpu().numpy()
        ycf = ycf.cpu().numpy()
        mu0 = mu0.cpu().numpy()
        mu1 = mu1.cpu().numpy()
        t = t.detach().cpu().numpy()
        
        return pred_0, pred_1, yf, ycf, mu0, mu1, t

    def evaluation(self, data: str) -> dict():

        dataloader = {
            'train': self.traineval_data,
            'eval': self.eval_data,
            'test': self.test_data}[data]

        pred_0, pred_1, yf, ycf, mu0, mu1, t = self.predict(dataloader)
        mode = 'in-sample' if data == 'train' else 'out-sample'
        metric = metrics(pred_0, pred_1, yf, ycf, mu0, mu1, t, mode, self.hparams)

        return metric


if __name__ == "__main__":

    hparams = argparse.ArgumentParser(description='hparams')
    hparams.add_argument('--data', type=str, default='IHDP')
    hparams.add_argument('--epoch', type=int, default=200)
    hparams.add_argument('--seed', type=int, default=2)
    hparams.add_argument('--stop_epoch', type=int, default=30, help='tolerance epoch of early stopping')
    hparams.add_argument('--treat_weight', type=float, default=0.0, help='whether or not to balance sample')
    hparams.add_argument('--model', type=str, default='gmcfr')
    hparams.add_argument('--dim_backbone', type=str, default='60,60')
    hparams.add_argument('--dim_task', type=str, default='60,60')
    hparams.add_argument('--batchSize', type=int, default=32)
    hparams.add_argument('--lr', type=float, default=1e-3)
    hparams.add_argument('--l2_reg', type=float, default=1e-4)
    hparams.add_argument('--dropout', type=float, default=0)
    hparams.add_argument('--treat_embed', type=bool, default=True)
    hparams.add_argument('--lambda', type=float, default=0.01, help='weight of wass_loss in loss function')
    hparams.add_argument('--epsilon', type=float, default=1.0, help='Entropic Regularization in sinkhorn. In IHDP, it should be set to 0.5-5.0 according to simulation conditions')
    hparams.add_argument('--kappa', type=float, default=1.0, help='weight of marginal constraint in UOT. In IHDP, it should be set to 0.1-5.0 according to simulation conditions')
    hparams.add_argument('--gamma', type=float, default=0.000005, help='weight of joint distribution alignment. In IHDP, it should be set to 0.0001-0.005 according to simulation conditions')
    hparams.add_argument('--ot_joint_bp', type=bool, default=True, help='weight of joint distribution alignment')
    hparams.add_argument('--beta', type=float, default=0.5, help='trade-off inside Gromov Monge Gap')
    hparams.add_argument('--pretrain_epoch', type=int, default=50, help='pretrain the prediction head')
    hparams.add_argument('--device', type=str, default='cuda:2')
    hparams = vars(hparams.parse_args())
        

    os.nice(0)
    estimator = BaseEstimator(hparams=hparams)    
    estimator.fit()
