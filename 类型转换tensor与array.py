'''
    作者:LSY
    文件:类型转换tensor与array
    日期:2022/4/13 17:25
    版本:
    功能:
    需求分析:
'''
#torch Tensor 和 Numpy array共享底层的内存空间,改变互相影响
from __future__ import print_function
import torch

a = torch.ones(5)
print(a)


b = a.numpy()
print(b)
print(a)
a.add_(2)
print(a)
print(b)


import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
#所有CPU上的Tensors，除了CharTensor 都可以转换为Numpy array并可以反向转换
#Cuda Tensor:Tensors 可以用.to()方法将其移动到任意设备上

import torch
#判断GPU存在 CUDA存在
if torch.cuda.is_available():
    #将设备指定成GPU
    device = torch.device("cuda")
    #直接在GPU上创建张量y,CPU上创建张量x
    x = torch.randn(1)
    y = torch.ones_like(x, device=device)
    #将x转移到GPU上
    x = x.to(device)
    #操作
    z = x + y#z也在GPU
    print(z)
    print(z.to("cpu",torch.double))