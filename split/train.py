# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Script for training merge model


import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
import torchmetrics
from torch.nn import Module
from torchvision.ops import FrozenBatchNorm2d

from tqdm import tqdm
from torch import optim, nn
from torch.utils.data import DataLoader
from dataset.dataset import ImageDataset
from loss.loss import bce_loss
from modules.split_modules import SplitModel
from split.test import test


def convert_to_frozen_batchnorm(module: Module):
    """
    Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
    Args:
        module (torch.nn.Module):
    Returns:
        If module is BatchNorm/SyncBatchNorm, returns a new module.
        Otherwise, in-place convert module and return it.
    Similar to convert_sync_batchnorm in
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
    """
    bn_module = nn.modules.batchnorm
    bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
    res = module
    if isinstance(module, bn_module):
        res = FrozenBatchNorm2d(module.num_features)
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = convert_to_frozen_batchnorm(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


def train(opt, net):
    """
    Train the split model
    Args:
      opt(dic): Options
      net(torch.model): Split model instance
    """
    with open(opt.json, 'r') as f:
        labels = json.load(f)
    dir_img = opt.img_dir

    with open(opt.val_json, 'r') as f:
        val_labels = json.load(f)
    val_img_dir = opt.val_img_dir

    device = torch.device("cuda") if opt.gpu else torch.device("cpu")

    train_set = ImageDataset(dir_img, labels, opt.featureW, scale=opt.scale, device=device)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    val_set = ImageDataset(val_img_dir, val_labels, opt.featureW, scale=opt.scale, device=device)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)

    print('Data loaded!')

    loss_func = bce_loss
    optimizer = optim.Adam(net.parameters(),
                           lr=opt.lr,
                           weight_decay=0.001)
    best_accuracy = 0
    metric_col = torchmetrics.classification.BinaryF1Score().to(device)
    metric_row = torchmetrics.classification.BinaryF1Score().to(device)
    for epoch in tqdm(range(opt.epochs)):
        try:
            print('epoch:{}'.format(epoch + 1))
            net.train()
            epoch_loss = 0
            for i, b in tqdm(enumerate(train_loader)):
                img, label = b
                if opt.gpu:
                    img = img.cuda()
                    label = [x.cuda() for x in label]
                pred_label = net(img)
                loss = loss_func(pred_label, label, [0.1, 0.25, 1])
                row_pred, column_pred = pred_label
                row_label, column_label = label
                metric_col(column_pred, column_label)
                metric_row(row_pred, row_label)
                epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            acc_col = metric_col.compute()
            acc_row = metric_row.compute()
            print(
                'Epoch finished ! Loss: {0} , Row F1 Score: {1}, Col F1 Score: {2}'.format(
                    epoch_loss / (i + 1), acc_row, acc_col)
            )
            metric_col.reset()
            metric_row.reset()
            val_loss, val_acc_row, val_acc_col = test(opt, net, val_loader, device)
            mean_val_acc = val_acc_row + val_acc_col
            if mean_val_acc > best_accuracy:
                best_accuracy = mean_val_acc
                torch.save(net.module.state_dict(),
                           os.path.join(opt.saved_dir, 'model_epoch{}.pth'.format(epoch + 1)))
        except KeyboardInterrupt:
            print('Exiting gracefully...')
            torch.save(net.module.state_dict(),
                       os.path.join(opt.saved_dir, 'model_exit_in_epoch_{}.pth'.format(epoch + 1)))
            exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size of the training set')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--gpu', type=bool, default=True, help='if use gpu')
    parser.add_argument('--gpu_list', type=str, default='0',
                        help='which gpu could use')
    parser.add_argument('--lr', type=float, default=0.00075,
                        help='learning rate, default=0.00075 for Adam')
    parser.add_argument('--saved_dir', type=str, required=True,
                        help='saved directory for output models')
    parser.add_argument('--json', type=str, required=True,
                        help='labels of the data')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='image directory for input data')
    parser.add_argument('--val_json', type=str, required=True,
                        help='labels of the validation data')
    parser.add_argument('--val_img_dir', type=str, required=True,
                        help='image directory for validation data')
    parser.add_argument('--featureW', type=int, default=8, help='width of output')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scale of the image')
    parser.add_argument('--pretrained', type=str, required=False)

    opt = parser.parse_args()

    net = SplitModel(3)
    if opt.pretrained:
        IncompatibleKeys = net.load_state_dict(torch.load(opt.pretrained, map_location="cpu"))
        net = convert_to_frozen_batchnorm(net)

    if opt.gpu:
        cudnn.benchmark = True
        cudnn.deterministic = True
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_list
        net = torch.nn.DataParallel(net).cuda()

    if not os.path.exists(opt.saved_dir):
        os.mkdir(opt.saved_dir)

    train(opt, net)
