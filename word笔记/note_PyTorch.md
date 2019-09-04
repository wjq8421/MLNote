​	神经网络上的局部极值问题：深层网络虽然局部极值非常多，但是通过深度学习的Batch Gradient Descent优化方法很难陷进去，而且就算陷进去，其局部极小值点与全局极小值点也是非常接近的。而浅层网络却虽然拥有较少的局部极小值点，但是却很容易陷进去，且这些局部极小值点与全局极小值点相差较大。



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



