from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 得到所有类别名字列表
classes = load_classes(opt.class_path)

# Get data configuration  data_config_path=config/coco.data文件存储的是：类别数量、训练集图片路径脚本、验证集图片路径脚本、类别
# 名字脚本等信息，data_config是一个字典，存储了这些值
data_config = parse_data_config(opt.data_config_path)
# 得到训练数据集脚本路径 data/coco/trainvalno5k.txt
train_path = data_config["train"]

# Get hyper parameters 得到config/yolov3.cfg文件里面的超参数，parse_model_config()函数将yolov3.cfg的各个网络层用一个列表保存，
# 一个列表元素是一个字典，对应网络的一部分结构。列表第一个元素对应网络的超参数
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model，初始化网络模型，得到一个对象model，model中保存了网络的结构以及参数
model = Darknet(opt.model_config_path)
# model.load_weights(opt.weights_path)
# 初始化模型参数，只有卷积和batch normal才会被初始化。weights_init_normal是一个函数，它会将参数用正态分布来初始化
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

# 将模型设置为训练状态
model.train()

# Get dataloader
# ListDataset(train_path)是一个自定义的继承自torch.utils.data.Dataset的类，定义了数据的接口。torch.utils.data.DataLoader()
# 也是初始化DataLoader这个类，在for batch_i, (_, imgs, targets) in enumerate(dataloader)中遍历DataLoader类对应的对象
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epochs):
    # 每次遍历dataloader得到一个batch所有图片的路径：_，这个batch所有图片的数据组成的tenosr：imgs,batch size=2时，imgs维度为
    # 2x3x416x416。这个batch所有图片的标签组成的tensor：targets，维度为2x50x5，
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        # 将图片数据组成的tensor变成Variable?为什么要变成Variable????
        imgs = Variable(imgs.type(Tensor))
        # 一个batch所有图片的标签组成的tensor，每张图片标签最多给50个，每个标签是(ID,x,y.w,h),相对于填充图片的中心坐标与宽高
        targets = Variable(targets.type(Tensor), requires_grad=False)

        # 每个batch的梯度值清零
        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
