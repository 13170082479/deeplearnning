'''
    作者:LSY
    文件:07.手写数字识别
    日期:2022/4/16 12:50
    版本:
    功能:
    需求分析:
'''
import torch
import  torch.nn as nn
import  torch.nn.functional as F
import torch.optim as optim
from torchvision import  datasets,transforms
import torchvision
from  torch.autograd import  Variable
from  torch.utils.data import  DataLoader


train_dataset = datasets.MNIST(root='./num/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./num/',train=False, transform=transforms.ToTensor(),download=True)

