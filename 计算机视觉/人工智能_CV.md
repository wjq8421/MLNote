### RANSAC

> （Random Sample Consessus，随机采样一致算法）：从一组含有Outliers的数据中正确估计数学模型参数的迭代算法。

普通最小二乘：从一个整体误差最小的角度去考虑，尽量谁也不得罪。

RANSAC：假设数据具有某种特性，为了达到目的，适当割舍一些现有的数据。

1. 假定模型（如直线方程），并随即抽取Nums个（2个）样本点，对模型进行拟合；
2. 由于不是严格线性，数据点都有一定波动，假设容差范围为$\sigma$，找出距离拟合曲线容差范围内的点，并统计点的个数；
3. 重新随机抽取Nums个点，重复步骤1~2，直到结束迭代。
4. 每一次拟合后，容差范围内都有一定的数据点数，找出数据点数最多的情况，就是最终的拟合结果。



- 每一次随机抽取样本数Nums的选取；
- 迭代次数Iter的选取；
- 容差范围$\sigma$的选取。

 

#### LDP问题

> Location Determination Problem



#### Homography 矩阵



#### 图像拼接

> Image Stitching





#### 参考链接

1. 随机抽选一致算法：https://www.cnblogs.com/xingshansi/p/6763668.html
2. RANSAC算法详解：https://zhuanlan.zhihu.com/p/62238520
3. Scipy: Cookbook/RANSAC：http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC
4. 在PythonCV3中进行特征点筛选与优化：http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC

_____

### 霍夫变换