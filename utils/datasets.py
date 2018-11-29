import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize
import sys


# 这个函数在训练的时候没有用到
class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


# 自定义的数据接口输出，是继承自torch.utils.data.Dataset类的一个类
class ListDataset(Dataset):
    # 传入参数list_path是data/coco/trainvalno5k.txt，里面存储的是每张图片的路径
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        # 遍历img_files列表，一个元素是一张图片的路径。将每张图片路径转换成每张图片标签路径，存储于label_files列表中
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        # 得到传入网络的图片尺寸
        self.img_shape = (img_size, img_size)
        # 一张图片中检测目标最多有多少个，这里默认为50个
        self.max_objects = 50

    # 在dataloader.py的_DataLoaderIter类中，dataset[i]的dataset对象就是ListDataset产生的迭代对象，dataset[i]就会调用
    # __getitem__(self, index)函数 ，index就是需要读入图片的序号
    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first 将HxWxC转换成pytorch的CxHxW格式
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------
        # 得到正在处理图片的标签路径
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            # 载入图片标签，并且将它变为n行5列的列表，每行5个元素对应 类别序号,x,y,w,h. x,y为标签方框中心坐标，w,h为方框宽高，都是
            # 相对于整张图片的w,h来归一化的
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image 得到没有缩放与填充之前图片中，目标方框的左上角与右下角坐标(x1,y1,x2,y2)
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates 将方框标签转换成相对于填充后的图片上的中心坐标与宽高，x,y,w,h
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix 检测个数最多个数为max_objects，这里给每个目标都给出一个默认的标签0
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            # 如果图片有目标存在，则将标签赋值给filled_labels. range(len(labels))[:self.max_objects，如果labels中目标标签的数量超过
            # max_objects，则只取labels中max_objects个标签赋值给filed_labels。如果labels中目标标签的数量小于max_objects，则只
            # 对filed_labels中标签个数的元素赋值。
            # range(len(labels))[:self.max_objects]，当labels中标签数量小于max_objects时，只取labels中标签的索引。如果labels中标
            # 签数量大于max_objects时，取labels中max_objects个索引
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

