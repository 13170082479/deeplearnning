'''
    作者:LSY
    文件:1
    日期:2022/4/15 17:23
    版本:
    功能:
    需求分析:
'''
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
