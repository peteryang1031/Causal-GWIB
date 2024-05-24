from networkx import dijkstra_predecessor_and_distance
from numpy import argmin
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

class SLearner(nn.Module):
    """
    Single learner with treatment as covariates
    """
    def __init__(self, input_dim, hparams):

        super(SLearner, self).__init__()

        out_backbone = hparams.get('dim_backbone', '100,100').split(',')
        out_task = hparams.get('dim_task', '50').split(',')
        in_backbone = [input_dim + 1] + list(map(int, out_backbone))
        in_task = [in_backbone[-1]] + list(map(int, out_task))
        self.backbone = torch.nn.Sequential()

        for i in range(1, len(in_backbone)):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(in_backbone[i-1], in_backbone[i]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.LeakyReLU())
            self.backbone.add_module(f"backbone_drop{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.tower = torch.nn.Sequential()
        for i in range(1, len(in_task)):
            self.tower.add_module(f"tower_dense{i}", torch.nn.Linear(in_task[i-1], in_task[i]))
            self.tower.add_module(f"tower_relu{i}", torch.nn.LeakyReLU())
            self.tower.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output = torch.nn.Sequential()
        self.output.add_module("output_dense", torch.nn.Linear(in_task[-1], 1))

        self.rep_1 = None
        self.rep_0 = None

    def forward(self, x):
        
        covariates = x[:, :-1]
        t = x[:, -1]
        covariates = torch.cat([covariates, t.reshape([-1, 1])], dim=-1)
        rep = self.backbone(covariates)
        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]
        out = self.tower(rep)
        out = self.output(out)

        return out


class TLearner(nn.Module):
    """
    Two learner with covariates in different groups modeled isolatedly.
    """
    def __init__(self, input_dim, hparams):

        super(TLearner, self).__init__()

        out_backbone= hparams.get('dim_backbone', '32,16').split(',')
        out_task = hparams.get('dim_task', '16').split(',')
        in_backbone = [input_dim] + list(map(int, out_backbone))
        in_task = [in_backbone[-1]] + list(map(int, out_task))
        self.backbone_1 = torch.nn.Sequential()

        for i in range(1, len(in_backbone)):
            self.backbone_1.add_module(f"backbone_dense{i}", torch.nn.Linear(in_backbone[i-1], in_backbone[i]))
            self.backbone_1.add_module(f"backbone_relu{i}", torch.nn.LeakyReLU())
            self.backbone_1.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.tower_1 = torch.nn.Sequential()
        for i in range(1, len(in_task)):
            self.tower_1.add_module(f"tower_dense{i}", torch.nn.Linear(in_task[i-1], in_task[i]))
            self.tower_1.add_module(f"tower_relu{i}", torch.nn.LeakyReLU())
            self.tower_1.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output_1 = torch.nn.Sequential()
        self.output_1.add_module("output_dense", torch.nn.Linear(in_task[-1], 1))

        self.backbone_0 = deepcopy(self.backbone_1)
        self.tower_0 = deepcopy(self.tower_1)
        self.output_0 = deepcopy(self.output_1)

        self.rep_1 = None
        self.rep_0 = None

    def forward(self, x):

        covariates = x[:, :-1]
        t = x[:, -1]  # shape: (-1)
        rep_1 = self.backbone_1(covariates)
        rep_0 = self.backbone_0(covariates)
        out_1 = self.tower_1(rep_1)
        out_0 = self.tower_0(rep_0)
        out_1 = self.output_1(out_1)
        out_0 = self.output_0(out_0)

        self.rep_1 = rep_1[t == 1]
        self.rep_0 = rep_0[t == 0]

        t = t.reshape(-1, 1)
        output_f = t * out_1 + (1 - t) * out_0

        return output_f


class GMCFR(nn.Module):
    """
    Our proposed GMCFR model.
    """
    def __init__(self, input_dim, hparams):

        super(GMCFR, self).__init__()

        out_backbone = hparams.get('dim_backbone', '32,16').split(',')
        out_task = hparams.get('dim_task', '16').split(',')
        self.treat_embed = hparams.get('treat_embed', True)
        in_backbone = [input_dim] + list(map(int, out_backbone))
        print('in_backbone is ' + str(input_dim))
        self.backbone = torch.nn.Sequential()
        for i in range(1, len(in_backbone)):
            self.backbone.add_module(f"backbone_dense{i}", torch.nn.Linear(in_backbone[i-1], in_backbone[i]))
            self.backbone.add_module(f"backbone_relu{i}", torch.nn.ELU())
            self.backbone.add_module(f"backbone_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        in_task = [in_backbone[-1]] + list(map(int, out_task))
        if self.treat_embed is True: 
            in_task[0] += 2

        self.tower_1 = torch.nn.Sequential()
        for i in range(1, len(in_task)):
            self.tower_1.add_module(f"tower_dense{i}", torch.nn.Linear(in_task[i-1], in_task[i]))
            self.tower_1.add_module(f"tower_relu{i}", torch.nn.ELU())
            self.tower_1.add_module(f"tower_dropout{i}", torch.nn.Dropout(p=hparams.get('dropout', 0.1)))

        self.output_1 = torch.nn.Sequential()
        self.output_1.add_module("output_dense", torch.nn.Linear(in_task[-1], 1))

        self.tower_0 = deepcopy(self.tower_1)
        self.output_0 = deepcopy(self.output_1)

        self.rep_1, self.rep_0 = None, None
        self.out_1, self.out_0 = None, None
        self.embedding = nn.Embedding(2, 2)

    def forward(self, x):

        covariates = x[:, :-1]
        t = x[:, -1]
        rep = self.backbone(covariates)
        if self.treat_embed is True:
            t_embed = self.embedding(t.int())
            rep_t = torch.cat([rep, t_embed], dim=-1)
        else:
            rep_t = rep

        self.rep_1 = rep[t == 1]
        self.rep_0 = rep[t == 0]

        self.out_1 = self.output_1(self.tower_1(rep_t))
        self.out_0 = self.output_0(self.tower_0(rep_t))

        t = t.reshape(-1, 1)
        output_f = t * self.out_1 + (1 - t) * self.out_0

        return output_f
