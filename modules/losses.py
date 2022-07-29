#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class L2Loss(nn.Module): # 自作L2
  def __init__(self):
    super().__init__()

  def forward(self, outputs, targets):
    loss = ((outputs - targets)**2).sum()
    return loss

class MSEScore(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, outputs, targets):
    score = []
    for i in range(len(outputs)):
      score.append(((outputs[i] - targets[i])**2).mean().cpu().numpy())
    return score

def get_loss(loss_name):
  if loss_name == 'L2':
    return L2Loss()
  if loss_name == 'MSEScore':
    return MSEScore()
  else:
    return nn.MSELoss()