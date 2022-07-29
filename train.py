#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
from modules.loader import get_dataset
from torchvision import utils
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(args, logger, model):
    dataset = get_dataset(args, is_train=True)
    val_size = int(len(dataset) * args.val_rate)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=False, drop_last=True)

    logger.info(f'Train: {train_size}({len(train_loader)}), Val: {val_size}({len(val_loader)})')

    vis_loader = torch.cat([batch[0] for batch in val_loader], dim=0)
    vis_inputs = vis_loader[:25].to(args.device)
    utils.save_image(vis_inputs, os.path.join(args.pic_dir, 'val_inputs.png'), nrow=5, normalize=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loop = tqdm(train_loader, unit='batch', desc=f'Train [Epoch {epoch:>3}]')

        t_loss = []
        for _, (inputs, _) in enumerate(loop):
            inputs = inputs.to(args.device)
            outputs, loss = model(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss.append(loss.item())

        train_loss_list.append(np.average(t_loss))
        logger.info(f'[Epoch {epoch}/{args.epochs}] Train_loss : {train_loss_list[-1]}')

        with torch.no_grad():
            model.eval()
            loop_val = tqdm(val_loader, unit='batch', desc=f'Val [Epoch {epoch:>3}]')

            v_loss = []
            for _, (inputs, _) in enumerate(loop_val):
                inputs = inputs.to(args.device)
                outputs, loss = model(inputs)

            v_loss.append(loss.item())

        val_loss_list.append(np.average(v_loss))
        logger.info(f'[Epoch {epoch}/{args.epochs}] Val_loss : {val_loss_list[-1]}')

        if epoch % 10 == 0:
            with torch.no_grad():
                outputs, _ = model(vis_inputs)
                utils.save_image(outputs, os.path.join(args.pic_dir, f'val_outputs-{epoch}.png'), nrow=5)
                logger.info(f'Val picture {epoch} exported.')
    
    torch.save(model.state_dict(), os.path.join(args.result_dir, 'model.pt'))
    args.tr_loss  = train_loss_list[-1]
    args.val_loss = val_loss_list[-1]

    y_max = np.max([0.5]+np.median(val_loss_list))
    fig = plt.figure(figsize=(6,6))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, y_max])
    plt.grid()
    fig.savefig(os.path.join(args.result_dir, 'loss.png'))
    plt.clf()
    plt.close()
    print('Loss Graph exported.')