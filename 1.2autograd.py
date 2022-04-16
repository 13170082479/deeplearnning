'''
    作者:LSY
    文件:1.2autograd
    日期:2022/4/13 22:25
    版本:
    功能:
    需求分析:
'''
import torch
#autograd package 提供了一个Tensors上所有的操作进行自动微分的功能
#Tensors类和Function类 tensors
x1 = torch.ones(3, 3)
print(x1)
x = torch.ones(2, 2 , requires_grad=True)#tensors类追踪x所有的梯度
print(x)

y = x + 2#
#z = x *4
print(y)
print(x.grad_fn)
print(y.grad_fn) #加法<AddBackward0 object at 0x0000022AD47D9A00>
#print(z.grad_fn) #乘法<MulBackward0 object at 0x0000022AD47D9A00>

z = y*y*3
out = z.mean()
print(z, out) # grad_fn均值tensor(27., grad_fn=<MeanBackward0>)

#requires_grad_():下划线结尾的方法都是inplace方法，原地替换
a = torch.randn(2, 2)
a = (a*3)/(a -1)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()
print(x.grad)

print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x))