# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Image dataset for training and testing Split and Merge models(ICDAR 2019 :Deep Splitting and Merging for Table Structure Decomposition).

import cv2
import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, img_dir, labels_dict, output_width, scale=0.5,
                 min_width=40, mode='split', suffix='.jpg', device: torch.device = torch.device("cpu")):
        """
        Initialization of the dataset

        Args:
          img_dir(str): The directory of images
          labels_dict(dict): A dictionary stores ids of images and
              corresponding ground truth, which are two vectors with
              w and h elements, where w is the width of image and h
              is the height of image, indicates the probability of
              there is a line in that row or column.
          output_width(int): Defines the width of the output tensor,
          scale(float): The scale of resizing image.
          min_width(int): Specifies minimal width of resizing image.
          mode(str): The model should be one of 'split' and 'merge'
        """
        self.labels_dict = labels_dict
        self.image_names = list(labels_dict.keys())
        self.nSamples = len(self.image_names)
        self.img_dir = img_dir
        self.output_width = output_width
        self.scale = scale
        self.min_width = min_width
        self.mode = mode
        self.suffix = suffix
        self.device = device

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image_name = self.image_names[index]
        # TODO: find out how this is used
        if self.suffix == '.npy':
            img = np.load(os.path.join(self.img_dir, image_name + self.suffix))
        elif self.suffix == '.jpg':
            img = np.array(
                Image.open(os.path.join(self.img_dir, image_name)).convert("RGB")
            )
            mask_h = np.array(
                Image.open(os.path.join(self.img_dir, image_name.replace(".jpg", "_horizontal.jpg"))).convert("L")
            )
            mask_v = np.array(
                Image.open(os.path.join(self.img_dir, image_name.replace(".jpg", "_vertical.jpg"))).convert("L")
            )
        h, w, c = img.shape
        new_h = int(self.scale * h) if int(
            self.scale * h) > self.min_width else self.min_width
        new_w = int(self.scale * w) if int(
            self.scale * w) > self.min_width else self.min_width
        img = np.array(
            [cv2.resize(img[:, :, i], (new_w, new_h), interpolation=cv2.INTER_AREA) for i in
             range(c)])
        img_array = np.array(img) / 255.
        mask_h_1d = (mask_h[:, 0] / 255. > 0.5).astype(np.int).tolist()
        mask_v_1d = (mask_v[0] / 255. > 0.5).astype(np.int).tolist()

        if self.mode == 'merge':
            labels = self.labels_dict[image_name]
            rows = mask_h_1d
            columns = mask_v_1d
            h_matrix = labels['h_matrix']
            v_matrix = labels['v_matrix']

            img_tensor = torch.from_numpy(img_array).type(torch.FloatTensor).to(self.device)
            row_label = torch.from_numpy(np.array(h_matrix)).type(torch.FloatTensor).to(self.device)
            column_label = torch.from_numpy(np.array(v_matrix)).type(
                torch.FloatTensor).to(self.device)

            return img_tensor, (row_label, column_label), (rows, columns)
        else:
            row_label = mask_h_1d
            column_label = mask_v_1d

            # resize ground truth to proper size
            width = int(np.floor(new_w / self.output_width))
            height = int(np.floor(new_h / self.output_width))
            row_label = np.array([row_label]).T * np.ones((len(row_label), width))
            column_label = np.array(column_label) * np.ones(
                (height, len(column_label)))

            row_label = np.array(row_label, dtype=np.uint8)
            column_label = np.array(column_label, dtype=np.uint8)

            row_label = cv2.resize(row_label, (width, new_h))
            column_label = cv2.resize(column_label, (new_w, height))

            row_label = row_label[:, 0]
            column_label = column_label[0, :]

            img_tensor = torch.from_numpy(img_array).type(torch.FloatTensor).to(self.device)
            row_label = torch.from_numpy(row_label).type(torch.FloatTensor).to(self.device)
            column_label = torch.from_numpy(column_label).type(torch.FloatTensor).to(self.device)

            return img_tensor, (row_label, column_label)
