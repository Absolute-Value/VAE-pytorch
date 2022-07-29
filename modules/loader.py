#!/usr/bin/env python
# coding: utf-8

import torch, os, random
from torchvision import transforms, datasets
from PIL import Image

class DatasetLoader(torch.utils.data.Dataset):
  def __init__(self, args, is_train=True):
    self.args = args
    self.is_train = is_train
    self.data_path = '../data/{}'.format(args.dataset)
    self.resize = (args.resize, args.resize)

    phase = 'train' if self.is_train else 'test'
    self.x, self.y, self.mask = [], [], []

    # 画像探索してアドレスを配列に保存
    phase_dir = os.path.join(self.data_path, args.data_type, phase)
    gt_dir = os.path.join(self.data_path, args.data_type, 'ground_truth')

    for data_type in os.listdir(phase_dir):
      data_type_dir = os.path.join(phase_dir, data_type)
      imgs = []
      for img in os.listdir(data_type_dir):
        img_dir = os.path.join(data_type_dir, img)
        if os.path.isfile(img_dir) and ((data_type == 'good' and is_train) or (data_type != 'good' and not is_train)):
          imgs.append(img_dir)
      self.x.extend(sorted(imgs))

      if data_type == 'good':
        self.y.extend([0] * len(imgs))
        self.mask.extend([None] * len(imgs))
      else:
        self.y.extend([1] * len(imgs))
        gt_type_dir = os.path.join(gt_dir, data_type)
        img_names = []
        for im in imgs:
          img_names.append(os.path.splitext(os.path.basename(im))[0])
        for nm in img_names:
          self.mask.append(os.path.join(gt_type_dir, nm + '_mask.png'))

    assert len(self.x) == len(self.y), 'number of x and y should be same'

    self.transform_x = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    self.transform_mask = transforms.Compose([
      transforms.ToTensor()])

  def __getitem__(self, idx):
    x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

    x = Image.open(x).convert('RGB') # 受け取ったアドレスを開く
    x = x.resize(self.resize) # リサイズ
    x = self.transform_x(x)

    if y == 0:
      mask = torch.zeros([1, self.resize, self.resize])
    else: # 推論時はGroundTruthも呼び出す
      mask = Image.open(mask).convert('1')
      mask = y.resize(self.resize)
      mask = self.transform_mask(mask)
    
    return x, y, mask

  def __len__(self):
    return len(self.x)

def get_dataset(args, is_train=True):
  if args.dataset == 'mnist' or args.dataset == 'fashion':
    if args.dataset == 'mnist':
      dataset = datasets.MNIST(
        root='.../data',
        transform=transforms.Compose([
          transforms.Resize(args.resize),
          transforms.ToTensor(),
          #transforms.Normalize([0.5], [0.5])
        ]),
        train=is_train,
        download=True
      )
    else:
      dataset = datasets.FashionMNIST(
        root='.../data',
        transform=transforms.Compose([
          transforms.Resize(args.resize),
          transforms.ToTensor(),
          #transforms.Normalize([0.5], [0.5])
        ]),
        train=is_train,
        download=True
      )
    _dataset = []
    if is_train:
      for data_x, data_y in dataset:
        if data_y == int(args.data_type):
          _dataset.append([data_x, 0])
    else:
      for data_x, data_y in dataset:
        if data_y == int(args.data_type):
          _dataset.append([data_x, 0])
        else:
          _dataset.append([data_x, 1])
    return _dataset
  else:
    return DatasetLoader(args, is_train=True)