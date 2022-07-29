#!/usr/bin/env python
# coding: utf-8

import argparse, os, csv
import torch
import numpy as np
from modules.logger import get_logger
from modules.vae import get_model
from modules.initializer import init_weight
from train import train
from test import test

def main():
    parser = argparse.ArgumentParser(description='VAE_AD')
    parser.add_argument('--model', type=str, default='ConvVAE')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--data_type', type=str, default='0')
    parser.add_argument('--resize', type=int, default=28)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden_dim', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--train_loss', type=str, default='KL', choices=['KL', 'L2', 'SSIM', 'SSIM2', 'MSE'])
    parser.add_argument('--test_loss', type=str, default='MSEScore', choices=['Abs','SSIM2','MSEScore'])
    parser.add_argument('--result', type=str, default='result')
    parser.add_argument('--train', type=bool, default='True')
    args = parser.parse_args()

    args.result_dir = os.path.join(
        f'{args.result}_{args.model}', 
        f'{args.dataset}{args.resize}',
        args.data_type, args.train_loss, 
        f'dim{args.hidden_dim}', 
        args.init_type,
        f'seed{args.seed}'
    )
    args.pic_dir = os.path.join(args.result_dir, 'pic')
    os.makedirs(args.pic_dir, exist_ok=True)

    logger = get_logger(args.result_dir, 'main.log')
    state = {k: v for k, v in args._get_kwargs()}
    logger.info(state)

    args.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    logger.info(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    model = get_model(args).to(args.device)

    if args.train:
        init_weight(model, args.init_type)
        train(args, logger, model)
    test(args, logger, model)

    with open(os.path.join(args.result_dir, f'result_{args.test_loss}_{args.pixel_rocauc:.3f}.csv'), 'w') as f:  
        writer = csv.writer(f)
        for k, v in args._get_kwargs():
            writer.writerow([k, v])
    
    with open(f'result_ConvVAE/{args.dataset}.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        if args.train:
            writer.writerow([
                args.model, args.data_type, args.resize, args.train_loss, 
                args.test_loss, args.hidden_dim, args.seed, 
                args.tr_loss, args.val_loss, args.pixel_rocauc
            ])
        else:
            writer.writerow([
                args.model, args.data_type, args.resize, args.train_loss, 
                args.test_loss, args.hidden_dim, args.seed, 
                '', '', args.pixel_rocauc
            ])

if __name__ == '__main__':
    main()