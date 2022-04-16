'''
    作者:LSY
    文件:03.定义损失函数
    日期:2022/4/15 21:31
    版本:
    功能:
    需求分析:
'''
import  torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# 选用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器，选用随机梯度下降优化器
optimizer  = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

