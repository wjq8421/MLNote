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

​	DataLoader类：实现取batch、shuffle或者多线程去读取数据。

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

