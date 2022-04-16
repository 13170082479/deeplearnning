'''
    作者:LSY
    文件:基本元素操作
    日期:2022/4/13 14:38
    版本:
    功能:
    需求分析:
'''
from __future__ import print_function
import torch
#x = torch.empty(5, 3)
#print(x)
# x = torch.rand(5, 3)
#print(x)
#创建全零矩阵
#x = torch.zeros(5, 3 ,dtype=torch.long)
#print(x)
#x1 = torch.tensor([2.5,3.5])
#print(x)

#利用news_methods得到一个张量
#x = x.new_ones(5 ,3, dtype = torch.double)
#print(x)
# 通过已有张量创建相同尺寸的
#y = torch.randn_like(x, dtype=torch.float)
#print(y)
#print(x1.size())
#torch.Size([5,3])

#print(y[:1])
#print(y[:, :2])

#x = torch.randn(4, 4)
#y = x.view(16)
#z = x.view(-1, 8)
#print(x.size(), y.size(),z.size())

#

#x = torch.randn(1)
#print(x)#tensor([-1.4008])封装的tensor
#print(x.item())#用item()拿出来
x = torch.randn(2)#item()取出，每次只能取数组中一个元素
print(x)
print(x[0].item())#下标索引到唯一元素，用item()取出