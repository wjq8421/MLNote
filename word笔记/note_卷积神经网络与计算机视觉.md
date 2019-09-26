CNN应用案例：图像分类、目标检测、语义分割、场景理解、图像生成



无监督学习方法：层次聚类、k均值聚类、高斯混合模型（GMM）、自组织映射（SOM）、隐马尔可夫模型（HMM）。

____

## 传统特征描述符

### HOG

> Histogram of Oriented Gradient，方向梯度直方图

用于自动检测图像中的对象，HOG描述符对图像中局部部分的梯度方向的分布进行编码。

背后的想法：通过边缘方向的直方图来描述图像内的对象外观和形状。



实现步骤：

1、图像预处理：伽马校正和灰度化。

2、计算图像在水平和垂直方向上的梯度值；

![img](https://pic1.zhimg.com/80/v2-1d866ca3e02c8288b17c9b714f71f5f0_hd.jpg)

对于像素点A，水平梯度$g_x=30-20=10$，垂直梯度$g_y=64-32=32$。

计算梯度强度值$g$和梯度方向$\theta$：
$$
g=\sqrt{g_x^2+g_y^2} \\
\theta=arctan\frac{g_x}{g_y}
$$

由于梯度方向将会取绝对值，因此梯度方向的范围是0~180度。

3、按non-overlap取一个$8\times 8$的cell计算梯度直方图，则就会有$8*\times 8\times 2=128$个值。若将0~180度分成9个bins，分别是$0,20,\cdots,160$，然后统计每个像素点的梯度方向所在的bin，对该bin的投票值由梯度强制决定。统计完64个点的值，可得到一个直方图，即为大小为9的vector。

4、由4个cell组成$16\times 16$的block，block由cell按步长1迭代截取。此时，4个cell组成大小为36的vector，再进行归一化，以降低光照的影响。

5、对于$64\times 128$大小的图像而言，将会有7个水平位置和15个竖直位置的block，共有$7\times 15=105$个block。整合所有block的vector，形成大小为$36\times 105=3780$的vector，即为HOG特征向量。



一般来说，只有图像区域比较小的情况，基于统计原理的直方图对于该区域才有表达能力；如果图像区域比较大，那么两个完全不同的图像的HOG特征可能很相似；但如果区域较小，这种可能性就很小。因此，需要把图像分割成很多区块，然后对每个区块计算HOG特征。

a. 图像分割策略：overlap、non-overlap。

​		overlap：可防止对一些物体的切割，如分割的时候正好把眼睛切割并且分到了两个区块中，提取完HOG特征之后，会影响接下来的分类效果，但如果使用overlap，那么至少在一个区块里会有完整的眼睛。	缺点：计算量大，重叠区域的像素需要重复计算。





HOG特征：https://zhuanlan.zhihu.com/p/40960756

https://blog.csdn.net/zhanghenan123/article/details/80853523



### SIFT

> 尺度不变特征变换（Scale-invariant Feature Transform）



SIFT算法原理解析：https://blog.csdn.net/hit2015spring/article/details/52895367

SIFT算法总结：https://blog.csdn.net/jancis/article/details/80824793



### SURF

