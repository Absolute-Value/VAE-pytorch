#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
from modules.loader import get_dataset
from modules.losses import get_loss
from tqdm import tqdm
from torchvision import transforms
from sklearn import metrics
import matplotlib.pyplot as plt

#import pdb; pdb.set_trace()

def test(args, logger, model):
    args.ad_dir = os.path.join(args.result_dir, args.test_loss)
    if not os.path.isdir(args.ad_dir):
        os.makedirs(args.ad_dir)

    test_dataset = get_dataset(args, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True, shuffle=False, drop_last=False)
    criterion = get_loss(args.test_loss).to(args.device)

    model.load_state_dict(torch.load(os.path.join(args.result_dir, 'model.pt')))
    
    imgs = []
    labels = []
    #gts = []
    abs =[]
    reconsts = []
    scores = []

    with torch.no_grad():
        model.eval()
        loop_test = tqdm(test_loader, unit='batch', desc='Test')
        for i, (inputs, label) in enumerate(loop_test):
            imgs.extend(inputs.cpu().numpy())
            labels.extend(label)
            #gts.extend(mask.cpu().numpy())
            inputs = inputs.to(args.device)
            outputs, _ = model(inputs)
            reconsts.extend(outputs.cpu().numpy())
            
            score = criterion(inputs, outputs)
            scores.extend(score)

            abs.extend(outputs.cpu().numpy() - inputs.cpu().numpy())
    #import pdb; pdb.set_trace()

    #gts = np.asarray(gts)
    labels = np.asarray(labels)
    reconsts = np.asarray(reconsts)
    scores = np.asarray(scores)
    abs = np.asarray(abs)
    scores_min = scores.min()
    scores_max = scores.max()
    scores = (scores - scores_min) / (scores_max - scores_min)
    abs[abs<0]=0
    abs_min = scores.min()
    abs_max = scores.max()
    abs = (abs - abs_min) / (abs_max - abs_min)

    calc_img_pr(args, logger, labels, scores)
    calc_img_roc(args, logger, labels, scores)
    plt_tests(args, imgs[:100], labels, reconsts, abs, scores)

def calc_img_pr(args, logger, labels, scores):
    precision, recall, _ = metrics.precision_recall_curve(labels.flatten(), scores.flatten())
    args.pixel_prauc = metrics.auc(recall, precision)
    logger.info(f'img PRAUC: {args.pixel_prauc:.3f}')
    plt.plot(recall, precision, label=f'{args.data_type} pixel_PRAUC: {args.pixel_prauc:.3f}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.ad_dir, args.data_type + '_pr_curve.png'), dpi=100)
    plt.clf()
    plt.close()

def calc_pix_pr(args, logger, gts, scores):
    precision, recall, thresholds = metrics.precision_recall_curve(gts.flatten(), scores.flatten())
    args.pixel_prauc = metrics.auc(recall, precision)
    logger.info(f'pixel PRAUC: {args.pixel_prauc:.3f}')
    plt.plot(recall, precision, label=f'{args.data_type} pixel_PRAUC: {args.pixel_prauc:.3f}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(args.ad_dir, args.data_type + '_pr_curve.png'), dpi=100)
    plt.clf()
    plt.close()

    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    return threshold

def calc_img_roc(args, logger, gts, scores):
    fpr, tpr, _ = metrics.roc_curve(gts.flatten(), scores.flatten())
    args.pixel_rocauc = metrics.roc_auc_score(gts.flatten(), scores.flatten())
    logger.info(f'pixel ROCAUC: {args.pixel_rocauc:.3f}')
    plt.plot(fpr, tpr, label=f'{args.data_type} pixel_ROCAUC: {args.pixel_rocauc:.3f}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.ad_dir, args.data_type + '_roc_curve.png'), dpi=100)
    plt.clf()
    plt.close()

def plt_tests(args, imgs, labels, reconsts, abs, scores):
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(15,3))
    for i in range(len(imgs)):
        for a in ax:
            a.axes.xaxis.set_visible(False)
            a.axes.yaxis.set_visible(False)
        ax[0].set_title('Image', fontsize=16, color='black')
        ax[0].imshow(imgs[i][0], cmap='gray')
        ax[1].set_title('Reconst', fontsize=16, color='black')
        ax[1].imshow(reconsts[i][0], cmap='gray')
        ax[2].set_title('PredictHeatMap', fontsize=16, color='black')
        #ax[2].imshow(imgs[i][0], cmap='gray')
        ax[2].imshow(abs[i][0], cmap='jet', alpha=0.5)
        #ax[3].set_title('GroundTruth', fontsize=16, color='black')
        #ax[3].imshow(gts[i][0], cmap='gray')
        #ax[4].set_title('PredictMask', fontsize=16, color='black')
        #mask = abs[i][0]
        #mask[mask > threshold] = 1
        #mask[mask <= threshold] = 0
        #ax[4].imshow(mask, cmap='gray')
        fig.savefig(os.path.join(args.ad_dir, f'{i}_{labels[i]}_{scores[i]:.3f}.png'), dpi=100)
        plt.cla()
    plt.clf()
    plt.close()