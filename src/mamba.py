import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange, repeat

import sys
import os

from models.s4.pscan_pytorch import pscan

class MambaBlock(nn.Module):
    def __init__(self, d_state=16, dropout=0.1, dt_rank=8, d_model=-1, scale_dt=1.0, transposed=True, dis_mode = "ZOH", **kernel_args):
        super().__init__()

        self.d_model = d_model
        self.d_output = self.d_model
        self.dt_rank = dt_rank
        self.d_state = d_state
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.scale_dt = scale_dt
        self.dis_mode = dis_mode
        
        # delta bias
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=0.001)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.register("log_A_real", torch.log(A), 0.001)
        self.D = nn.Parameter(torch.randn(self.d_model))
        self.activation = nn.GELU()

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": wd}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, x, printflag = False, mode = 'scan', **kwargs):
        # x : (B, L, D)
        
        # y : (B, L, D)


        _, L, _ = x.shape

        y = self.ssm(x)

        return y
    
    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        # First output
        A = -torch.exp(self.log_A_real.float()) # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        delta = delta.transpose(1, 2)
        # delta = F.softplus(0 * delta + self.dt_proj.bias) * self.scale_dt  # For fixdt
        delta = F.softplus(delta + self.dt_proj.bias) * self.scale_dt
        
        y = self.selective_scan(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        if self.dis_mode == "ZOH":
            deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
            deltaB_weight = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)
        else:
            deltaA_cont = delta.unsqueeze(-1) * A                  # (B, L, ED, N)
            denom = 1.0 - 0.5 * deltaA_cont                        # (B, L, ED, N)
            deltaA = (1.0 + 0.5 * deltaA_cont) / denom             # (B, L, ED, N)  # bilinear A_bar
            deltaB_weight = (delta.unsqueeze(-1) / denom) * B.unsqueeze(2)  # (B, L, ED, N)  # bilinear B_bar * B

        BX = deltaB_weight * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = torch.sum(hs, dim=-1) + D * x

        return y