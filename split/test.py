# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Script for testing Split model.


import argparse
import json
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torchmetrics

from torch.utils.data import DataLoader
from dataset.dataset import ImageDataset
from modules.split_modules import SplitModel
from loss.loss import bce_loss


def test(opt, net, data=None, device=None):
  """
  Test script for Split model
  Args:
    opt(dic): Options
    net(torch.model): Split model instance
    data(dataloader): Dataloader or None, if load data with configuration in opt.
  Return:
    total_loss(torch.tensor): The total loss of the dataset
    accuracy(torch.tensor): the accuracy of the dataset
  """
  if not data:
    with open(opt.json_dir, 'r') as f:
      labels = json.load(f)
    dir_img = opt.img_dir

    test_set = ImageDataset(dir_img, labels, opt.featureW, scale=opt.scale)
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=True)
  else:
    test_loader = data

  loss_func = bce_loss

  metric_col = torchmetrics.classification.BinaryF1Score().to(device)
  metric_row = torchmetrics.classification.BinaryF1Score().to(device)

  for epoch in range(1):
    net.eval()
    epoch_loss = 0
    for i, b in enumerate(test_loader):
      with torch.no_grad():
        img, label = b
        if opt.gpu:
          img = img.cuda()
          label = [x.cuda() for x in label]
        pred_label = net(img)
        loss = loss_func(pred_label, label, [0.1, 0.25, 1])
        epoch_loss += loss
        row_pred, column_pred = pred_label
        row_label, column_label = label
        metric_col(column_pred, column_label)
        metric_row(row_pred, row_label)
    acc_col = metric_col.compute()
    acc_row = metric_row.compute()
    total_loss = epoch_loss / (i + 1)
    print('Validation finished ! Loss: {0} , Row F1 Score: {1}, Col F1 Score: {2}'.format(
      epoch_loss / (i + 1), acc_row, acc_col)
    )
    return total_loss, acc_row, acc_col


def model_select(opt, net):
  model_dir = opt.model_dir
  models = os.listdir(model_dir)
  losses = []
  accuracies = []
  for model in models:
    net.load_state_dict(torch.load(os.path.join(model_dir, model)))
    loss, accuracy = test(opt, net)
    losses.append(loss)
    accuracies.append(accuracy)
  min_loss_index = np.argmin(np.array(losses))
  max_accuracy_index = np.argmax(np.array(accuracies))
  print('accuracy:', max_accuracy_index, accuracies[max_accuracy_index],
        models[max_accuracy_index])
  print('losses', min_loss_index, losses[min_loss_index],
        models[min_loss_index])

  print(losses)
  print(accuracies)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=32,
                      help='batch size of the training set')
  parser.add_argument('--gpu', type=bool, default=True, help='if use gpu')
  parser.add_argument('--gpu_list', type=str, default='0',
                      help='which gpu could use')
  parser.add_argument('--model_dir', type=str, required=True,
                      help='saved directory for output models')
  parser.add_argument('--json_dir', type=str, required=True,
                      help='labels of the data')
  parser.add_argument('--img_dir', type=str, required=True,
                      help='image directory for input data')
  parser.add_argument('--featureW', type=int, default=8, help='width of output')
  parser.add_argument('--scale', type=float, default=0.5,
                      help='scale of the image')

  opt = parser.parse_args()

  net = SplitModel(3)
  if opt.gpu:
    cudnn.benchmark = True
    cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_list
    net = torch.nn.DataParallel(net).cuda()

  net.load_state_dict(torch.load('saved_models/CP53.pth'))
  print(test(opt, net))
  # model_select(opt, net)
