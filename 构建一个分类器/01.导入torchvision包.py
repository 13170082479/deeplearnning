'''
    作者:LSY
    文件:导入torchvision包
    日期:2022/4/15 20:47
    版本:
    功能:
    需求分析:
'''
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision数据集的输出是PILImage格式 转换数据域到（-1,1）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)

trainloader  = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers = 0)

testset = torchvision.datasets.CIFAR10(root='./data', train = False,download=True,transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 0)

classs = ('plane', 'car','bird','cat','deer','dog','frog','horse','ship','truck')

import  matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter =iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classs[labels[j]] for j in range(4)))