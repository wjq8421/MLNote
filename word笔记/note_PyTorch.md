### 目录

[TOC]

____

神经网络上的局部极值问题：深层网络虽然局部极值非常多，但是通过深度学习的Batch Gradient Descent优化方法很难陷进去，而且就算陷进去，其局部极小值点与全局极小值点也是非常接近的。而浅层网络却虽然拥有较少的局部极小值点，但是却很容易陷进去，且这些局部极小值点与全局极小值点相差较大。所以更希望使用大容量的网络去训练模型，同时运用一些方法来控制网络的过拟合。



​	Degradation问题 -> Deep Residual Net

_____

​	Variable：会被放入计算图中，然后进行前向传播、反向传播、自动求导。

```python
from torch.autograd import Variable

x = torch.randn(3)
x = Variable(x, requires_grad=True) # 需求导

y = x * 2

y.backward(torch.FloatTensor([1, 0.1, 20])) # 原本的梯度需分别乘上对应的值
print(x.grad)
```

_____

​	ImageFolder类：处理图片，在root根目录下，相同标签的图像应放在同一个类标签文件夹中。

```python
from torchvision.datasets import ImageFolder
# transform参数用于图片增强
dset = ImageFolder(rootdir, transform=, target_transform=)
```

______

​	DataLoader类：数据迭代器，实现取batch、shuffle或者多线程去读取数据。

```python
from torch.utils.data import DataLoader
# collate_fn：表示如何取样本
dataIter = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=)
```

______

​	所有的层结构和损失函数都来自于`torch.nn`。所有的模型构建都是从基类`nn.Module`继承的，不需要自己编写反向传播。

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
  
model = Model()
```

_____

### 线性模型

$$
f(x)=w^Tx+b
$$

​	权重的大小直接可以表示这个属性的重要程度。

​	使用均方误差来衡量$f(x)$和$y$之间的差别：
$$
(w^*,b^*)=\underset{w,b}{arg\ min}\ Loss(w,b)=\underset{w,b}{arg\ min}\sum_{i=1}^m(f(x_i)-y_i)^2
$$
​	为了计算方便，将$w$和$b$写进同一个矩阵，将数据集表示成一个$m\times (d+1)$的矩阵$X$。$m$指数据个数，$d$指特征数。

```python
from torch import optim
from torch import nn
import numpy as np
import torch

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], 
                    [7.042], [10.791], [5.313], [7.997],
                    [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], 
                    [1.573], [3.366], [2.596], [2.53], [1.221], 
                    [2.827], [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # 输入1维，输出1维
    
    def forward(self, x):
        out = self.linear(x)
        return out
  
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

# 使用均方误差作为优化函数，使用梯度下降进行优化
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()  # 
        target = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        target = Variable(y_train)
        
    # forward
    out = model(inputs) # 得到网络前向传播的结果
    loss = criterion(out, target) # 得到损失函数
    
    # backward：归零梯度、做反向传播和更新参数
    # 每次做反向传播之前都要归零梯度，不然梯度会累加在一起，造成结果不收敛
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print('Epoch [{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.data.item()))
```

______

### Logistic回归

​	对于二分类问题，Logistic回归的目标是希望找到一个区分度足够好的决策边界，能够将两类很好的分开。再通过找到分类概率$P(Y=1)$与输入变量$x$的直接关系，然后通过比较概率值来判断类别。
$$
P(Y=0|x)=\frac{1}{1+e^{w\cdot x +b}}\\
P(Y=1|x)=Sigmoid(w\cdot x+b)=\frac{e^{w\cdot x + b}}{1+e^{w\cdot x +b}}
$$
​		其损失函数为：
$$
L(w)=-\frac{1}{n}\sum_{i=1}^n[y_ilog(f(x_i))+(1-y_i)log(1-f(x_i))]
$$

```python
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()
        
    def forward(self, x):
        x = self.lr(x)
        out = self.sm(x)
        return out
```

_____

​	Sigmoid函数的缺点：

1. 会造成梯度消失。在靠近1和0两端时，梯度几乎为0。而梯度下降法通过梯度乘上学习率来更新参数，因此若梯度接近于0，那么没有任何信息来更新参数，就会造成模型不收敛。
2. 输出不是以0为均值，导致经过Sigmoid激活函数之后的输出，作为后面一层网络的输入的时候是非0均值的。此时，如果输入进入下一层神经元的时候全是正的，就会导致梯度全是正的。但由于神经网络在训练的时候都是按batch进行训练的，可在一定程度上缓解该问题。



​	一般在同一个网络中，我们都使用同一种类型的激活函数。

____

### 优化算法

​	**泰勒级数**：对于任何一个无限可微函数$h(x)$，在一个点$x=x_0$附近，其泰勒级数为：
$$
h(x)=\sum_{k=0}^\infty \frac{h^{(k)}(x_0)}{k!}(x-x_0)^k \\
=h(x_0)+h'(x_0)(x-x_0)+\frac{h''(x_0)}{2!}(x-x_0)^2+\cdots
$$
​	当$x$足够接近$x_0$，有如下的近似式：
$$
h(x)\approx h(x_0)+h'(x_0)(x-x_0)
$$
​	对于多元泰勒级数，有如下的公式：
$$
h(x,y)=h(x_0,y_0)+\frac{\partial h(x_0,y_0)}{\partial x}(x-x_0)+\frac{\partial h(x_0,y_0)}{\partial y}(y-y_0)
$$
​	**SGD**：每次使用一个batch数据进行梯度的计算。

​	**Momentum**：动量的计算基于前面梯度。-> **Nesterov**

​	**Adagrad**：一种自适应学习率的方法。缺点为：在某些情况下 造成学习过早停止。
$$
w^{t+1}\leftarrow w^t - \frac{\eta}{\sqrt{\sum_{i=0}^t(g^i)^2}+\epsilon}g^t
$$
​	**RMSprop**：自适应学习率的改进方法。RMSprop不再会将前面所有的梯度平方求和，而是通过一个衰减率将其变小，使用了一种滑动平均的方式，越靠前面的梯度对自适应的学习率影响越小，能更加有效地避免Adagrad学习率一直递减太多的问题。
$$
cache^t=\alpha * cache^{t-1}+(1-\alpha)(g^t)^2\\
w^{t+1}\leftarrow w^t - \frac{\eta}{\sqrt{cache^t+\epsilon}}g^t
$$
​	**Adam**：可看成是RMSprop加上动量的学习方法。

______

​	如果神经网络中每个权重都被初始化成相同的值，那么每个神经元就会计算相同的结果，在反向传播的时候也会计算出相同的梯度，最后导致所有权重都会有相同的更新。即权重之间失去了不对称性。

​	

​	随机初始化策略：高斯随机化、均匀随机化。

​	W：并不是越小的随机化产生的结果越好，因为权重初始化越小，反向传播中关于权重的梯度也越小，因为梯度与参数的大小是成比例的，所以这会极大地减弱梯度流的信号，称为神经网络训练中的一个隐患。



​	<font color=red>**批标准化**</font>：

_____

防止过拟合：

1. <font color=red>**L2正则化**</font>：$\frac{1}{2}\lambda w^2$，可看成是权重更新在原来的基础上再$-\lambda w$

2. **L1正则化**：$\lambda |w|$，在优化的过程中可以让权重变得更加稀疏。

3. <font color=red>**Dropout**</font>：在训练网络的时候依概率$P$保留每个神经元，即每次训练的时候有些神经元会被设置为0。

   但是在预测的时候，会保留网络全部的权重，取代应用Dropout。因为，如果预测应用Dropout，由于随机性，每次预测出来的结果都不一样，这显然是不行的。

   Q：为什么在网络输出部分应用$P$缩放可以达到相同的效果呢？

   A：考虑一个神经元在应用Dropout之前的输出是$x$，那么应用Dropout之后它的输出期望值就是$Px$，所以在预测的时候，如果保留所有的权重，就必须调整$x\rightarrow Px$来保证其输出与期待相同。



​	把Dropout看作是集成的学习方法：每一次训练Dropout之后就可以看作是一个新的模型，然后训练了很多次之后就可以看成是这些模型的集成。

_____

### 多层全连接神经网络

```python
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BatchNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(BatchNet, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Linear(in_dim, n_hidden_1),
                        nn.BatchNorm1d(n_hidden_1),
                        nn.ReLU(True))
        self.layer2 = nn.Sequential(
                        nn.Linear(n_hidden_1, n_hidden_2),
                        nn.BatchNorm1d(n_hidden_2),
                        nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
  
batch_size = 64
learning_rate = 1e-2
num_epoches = 20

data_tf = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]) # 减去0.5再除以0.5
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BatchNet(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model.cuda()
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    train_loss = 0.0
    train_acc = 0
    for batch_x, batch_y in train_loader:
        # shape of batch_x: [64, 1, 28, 28]
        # shape of batch_y: [64]
        if torch.cuda.is_available():
            batch_x = Variable(batch_x.view(-1, 28*28)).cuda()
            batch_y = Variable(batch_y).cuda()
        else:
            batch_x = Variable(batch_x.view(-1, 28*28))
            batch_y = Variable(batch_y)
        
        out = model(batch_x) # shape of out: [64, 10]
        loss = criterion(out, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item() * batch_y.size(0)
        _, pred = torch.max(out, 1)
  			train_acc += (pred == batch_y).sum().item()
    print('Epoch [{}/{}], loss: {:.6f}, acc: {:.6f}'.format(epoch, num_epoches, train_loss / len(train_dataset), train_acc / len(train_dataset)))
  
model.eval()
eval_loss = 0.0
eval_acc = 0.0
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
    
    out = model(img)
    loss = criterion(out, label) # 返回的loss为batch个样本的平均值
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_corrent = (pred == label).sum() # 获取每个批次预判正确的个数
    eval_acc += num_corrent.item() 
    
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(test_dataset), eval_acc / len(test_dataset)))
```

_____

### 卷积神经网络

​	卷积神经网络中的主要层结构：卷积层、池化层、全连接层。

#### 卷积层

​	由于图片特征的局部性，让每个神经元只与输入数据的一个局部区域连接即可提取出相应的特征。

​	与神经元连接的空间大小叫做神经元的感受野（Receptive Field），其实就是滤波器的宽和高（空间维度）。在深度方向上，其大小总是和输入的深度相等。如输入数据的尺寸是$16\times 16\times 20$，如果感受野（滤波器尺寸）是$3\times 3$，卷积层中每个神经元和输入数据之间就有$3\times 3\times 20=180$个连接。

​	卷积层的输出深度是一个超参数，与使用的滤波器数量一致，每种滤波器所做的是在输入数据中寻找一种特征。

​	滑动步长、边界填充：控制输出数据在空间上的尺寸。

​	输出尺寸：
$$
\frac{W-F+2P}{S}+1
$$
​	其中，$W$表示输入的数据大小，$F$表示卷积层中的神经元的感受野尺寸，$S$表示步长，$P$表示边界填充0的数量。

​	当$W=10$，$P=0$，$F=3$，$S=2$，$\frac{10-3+0}{2}=4.5$，即结果不是一个整数。说明神经元不能整齐对称地滑过输入数据体，这样的超参数设置是无效的。

​	参数共享：一个滤波器检测出一个空间位置$(x_1,y_1)$处的特征，那么也能够有效检测出$(x_2,y_2)$位置的特征，所以就可以用相同的滤波器来检测相同的特征。如卷积层输入的深度为$10$，输出为$20\times 20\times 32$，滤波器大小为$3\times 3$，则一共有$32$个滤波器，总共的参数为$32\times 3\times 3\times 10=2880$，以及$32$个偏置项。



​	零填充：使卷积层的输入和输出在空间上的维度保持一致。若不使用零填充，数据体的尺寸就会略微减少，在不断进行卷积的过程中，图像的边缘信息会过快地损失掉。



#### 池化层

> 其作用是逐渐降低数据体的空间尺寸，能够减少网络中参数的数量，减少计算资源耗费，同时能够有效地控制过拟合。

​	池化层有效的原因？ 由于图片特征具有不变性，即通过下采样不会丢失图片拥有的特征，所以将图片缩小再进行卷积处理，能够大大降低卷积运算的时间。一般尺寸为$2\times 2$的窗口，滑动步长为$2$。

​	池化层的感受野大小很少超过3，因为这会使得池化过程过于激烈，造成信息的丢失，使算法的性能变差。

​	最大池化、平均池化、L2范数池化。实际证明，在卷积层之间引入最大池化的效果是最好的，而平均池化一般放在卷积神经网络的最后一层。

____

​	一般而言，几个小滤波器卷积层的组合比一个大滤波器卷积层要好。如连续使用3个$3\times 3$感受野，与单独使用一个$7\times 7$大小的卷积层相比，1）多个卷积层首先与非线性激活层交替的结构，比单一卷积层的结构更能提取出深层的特征；2）使用的参数也更少。

​	一般而言，输入层的大小应该能被2整除很多次，如32、64、224。

_____

#### GoogleNet

1. 采用Inception模块，而且没有全连接层。

   Inception模块设计了一个局部的网络拓扑结构，然后将这些模块堆叠在一起形成一个抽象层网络结构。即运用几个并行的滤波器对输入进行卷积和池化，这些滤波器有不同的感受野，最后将输出的结果按深度拼接在一起形成输出层。

![img](E:\文件管理集合\笔记\机器学习\Cpp\pic\inception模块.jpg)

​	<font color=red>**$1\times 1$卷积的作用**</font>

1. 在相同尺寸的感受野中叠加更多的卷积，能提取到更丰富的特征。对于某个像素点来说，$1\times 1$卷积等效于该像素点在所有特征上进行一次全连接的计算。（参考Network in Network结构）
2. 使用$1\times 1$卷积对输入先进行降维，减少特征数之后再做卷积计算量就会显著减少。如下图，从$192\times 256\times 3\times 3\times 32\times 32$减小为$192\times 96\times 1\times 1\times 32\times 32 + 96\times 256\times 3\times 3\times 32\times 32$。

![img](E:\文件管理集合\笔记\机器学习\Cpp\pic\增加1乘1卷积后降低了计算量.jpg)

​	<font color=red>多个尺寸上进行卷积再聚合的解释</font>

1. 在多个尺度上同时进行卷积，能提取到不同尺度的特征，特征更为丰富也意味着最后分类判断时更加准确。
2. 传统的卷积层的输入数据只和一种尺度（如$3\times 3$）的卷积核进行卷积，输出固定维度（如256）的数据，所有256个输出特征基本上是均匀分布在$3\times 3$尺度范围上，这可理解成输出了一个稀疏分布的特征集。而inception在多个尺度上提取特征，输出的256个特征就不再是均匀分布，而是相关性强的特征聚集在一起，这可理解成多个密集分布的子特征集，能加快收敛速度。（可参考稀疏矩阵分解成密集矩阵计算的原理）

```python
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class Inception(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        
        self.branch3x3db_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3db_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3db_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3db1 = self.branch3x3db_1(x)
        branch3x3db1 = self.branch3x3db_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db_3(branch3x3db1)
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch5x5, branch3x3db1, branch_pool]
        return torch.cat(outputs, 1) # 按深度拼接起来
```



#### ResNet

​	解决的问题：在不断加深神经网络的时候，会出现一个Degradation，即准确率会先上升然后达到饱和，再持续增加深度则会导致模型准确率下降。

![img](E:\文件管理集合\笔记\机器学习\Cpp\pic\resnet)

​	上图为ResNet的残差学习单元，其不再学习一个完整的输出$H(x)$，而是学习输出和输入的差别$F(x)=H(x)-x$，即残差。

```python
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        return out
```

Network in Network，Highway Network。

____

 1. 拍照时的光照条件；

 2. 物体本身的变形；

 3. 物体本身是否隐藏在一些遮蔽物中；

    由于上述等问题，希望能够对原始图片进行增强，在一定程度上解决部分问题。

_____

### 图片数据处理



参考链接

1. 在Pytorch中建立自己的图片数据集：https://oidiotlin.com/create-custom-dataset-in-pytorch/