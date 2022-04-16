'''
    作者:LSY
    文件:反向传播
    日期:2022/4/15 18:37
    版本:
    功能:
    需求分析:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 卷积两层 输入通道维度1，输出通道维度6，卷积核大小3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 输入通道维度6，输出通道维度16，卷积核大小3*3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 三层全连接网络
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在 (2,2)的池化窗口下执行最大池化操作
        # 任意卷积层后面加激活层，池化层
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 进入全连接层之前 view调整张量维度
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 计算size， 除了第0个维度上的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

input = torch.randn(1, 1, 32, 32)
output = net(input)

target = torch.randn(10)
# 改变target的形状为二维向量 -1适配大小
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
#print(loss)

#print(loss.grad_fn) # MSELoss
#print(loss.grad_fn.next_functions[0][0]) # Linear
#print(loss.grad_fn.next_functions[0][0].next_funtions[0][0]) # relu


# 反向传播一定先执行梯度清零
net.zero_grad()
print("before:")
print(net.conv1.bias.grad)

loss.backward()
print("after:")
print(net.conv1.bias.grad)