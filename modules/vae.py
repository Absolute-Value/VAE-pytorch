#!/usr/bin/env python
# coding: utf-8

from typing import ForwardRef
import torch
import torch.nn as nn

class LinearVAE(nn.Module):
  def __init__(self, args):
    super(LinearVAE, self).__init__()
    self.args = args
    self.x_dim = args.resize * args.resize
    self.z_dim = args.hidden_dim
    self.device = args.device
    self.Encoder = nn.Sequential(
      nn.Linear(self.x_dim, 200),
      nn.ReLU(),
      nn.Linear(200, 200),
      nn.ReLU(),
    )
    self.EncMean = nn.Linear(200, self.z_dim)
    self.EncVar  = nn.Sequential(
      nn.Linear(200, self.z_dim),
      nn.Softplus(),
    )
    self.Decoder = nn.Sequential(
      nn.Linear(self.z_dim, 200),
      nn.ReLU(),
      nn.Linear(200, 200),
      nn.ReLU(),
      nn.Linear(200, self.x_dim),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = x.view(-1, self.x_dim)
    ec = self.Encoder(x)
    mean = self.EncMean(ec)
    var  = self.EncVar(ec)
    KL = 0.5 * torch.sum(1 + torch.log(var + 1e-8) - mean**2 - var)
    z = SampleZ(mean, var, self.device)
    x_hat = self.Decoder(z)
    reconstruction = torch.sum(x * torch.log(x_hat + 1e-8) + (1 - x) * torch.log(1 - x_hat + 1e-8))
    lower = -(KL + reconstruction) # 変分下限
    x_hat = x_hat.view(-1, 1, self.args.resize, self.args.resize)
    return x_hat, lower

class EncodeBlock(nn.Module):
  def __init__(self, in_ch, out_ch, kernel, pad, stride):
    super().__init__()
    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad, stride=stride)
    self.bn = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
  def __init__(self, color_dim, pooling_kernel, middle_dim):
    self.middle_dim = middle_dim
    super().__init__()
    self.e1 = EncodeBlock(color_dim, 32, kernel=1, pad=0, stride=1)
    self.e2 = EncodeBlock(32, 64, 3, 1, 1)
    self.e3 = EncodeBlock(64, 128, 3, 1, pooling_kernel[0])
    self.e4 = EncodeBlock(128, 256, 3, 1, pooling_kernel[1])

  def forward(self, x):
    x = self.e4(self.e3(self.e2(self.e1(x))))
    return x.view(-1, self.middle_dim)

class DecodeBlock(nn.Module):
  def __init__(self, in_ch, out_ch, stride, activation='relu'):
    super().__init__()
    self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=stride, stride=stride)
    self.bn = nn.BatchNorm2d(out_ch)
    if activation == 'sigmoid':
      self.activation = nn.Sigmoid()
    else:
      self.activation = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.activation(self.bn(self.conv(x)))

class Decoder(nn.Module):
  def __init__(self, color_dim, pooling_kernel, decoder_in_size):
    self.decoder_in_size = decoder_in_size
    super().__init__()
    self.d1 = DecodeBlock(256, 128, stride=1)
    self.d2 = DecodeBlock(128, 64, stride=pooling_kernel[1])
    self.d3 = DecodeBlock(64, 32, stride=pooling_kernel[0])
    self.d4 = DecodeBlock(32, color_dim, stride=1, activation='sigmoid')

  def forward(self, x):
    x = x.view(-1, 256, self.decoder_in_size, self.decoder_in_size)
    return self.d4(self.d3(self.d2(self.d1(x))))

class ConvVAE(nn.Module):
  def __init__(self, args):
    self.z_dim = args.hidden_dim
    self.device = args.device
    super(ConvVAE, self).__init__()

    if args.dataset in ['mnist', 'fashion']:
      color_dim = 1
      pooling_kernel = [2, 2]
      encoder_out_size = 7

    middle_dim = 256 * encoder_out_size * encoder_out_size

    self.encoder = Encoder(color_dim, pooling_kernel, middle_dim)
    self.enc_mu  = nn.Linear(middle_dim, self.z_dim)
    self.enc_var = nn.Linear(middle_dim, self.z_dim)
    self.var_soft= nn.Softplus()
    self.dec_lin = nn.Linear(self.z_dim, middle_dim)
    self.decoder = Decoder(color_dim, pooling_kernel, encoder_out_size)

  def forward(self, x):
    ec = self.encoder(x)
    mean, var = self.enc_mu(ec), self.var_soft(self.enc_var(ec))
    KL = 0.5 * torch.sum(1 + torch.log(var + 1e-8) - mean**2 - var)
    z = SampleZ(mean, var, self.device)
    z = self.dec_lin(z)
    x_hat = self.decoder(z)
    reconstruction = torch.sum(x * torch.log(x_hat + 1e-8) + (1 - x) * torch.log(1 - x_hat + 1e-8))
    lower = -(KL + reconstruction) # 変分下限
    return x_hat, lower

def SampleZ(mean, var, device):
  epsilon = torch.randn(mean.shape).to(device)
  return mean + torch.sqrt(var+1e-8) * epsilon


def get_model(args):
  if args.model == 'ConvVAE':
    return ConvVAE(args)
  else:
    return LinearVAE(args)