# 第三章 线性模型

1. 许多功能更为强大的非线性模型（nonlinear model）可在线性模型的基础上通过引入层级结构或高维映射而得。
2. **对数线性回归在形式上仍是线性回归，但实质上**已是在求取输入空间到输出空间的非线性函数映射。**这种单调可微的函数都是可逆的，这样得到的模型成为“广义线性模型”**
3. **对数几率：$\frac{y}{1-y}$称为几率，反映了$x$作为正例的相对可能性**
4. **线性判别分析(Linear Discriminant Analysis,简称LDA)**是一种经典的线性学习方法，在二分类问题上因为最早由[Fisher, 1936]提出,亦称“Fisher判别分析”。
   LDA的思想非常朴素：给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离；在对新样本进行分类时，将其投影到同样的这条直线上,再根据投影点的位置来确定新样本的类别。如图是一个二维示意图：
![20230608161020](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20230608161020.png)

> **此处给出了散度的作用，以及和协方差之间的关系**。

![20230608161110](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20230608161110.png)
5. 多分类可通过多次二分类来完成
6. **类别不平衡（class-imbalance）就是指分类任务中不同类别的训练样例数目差别很大的情况**。举一个极端例子，有998个反例，2个正例，那么模型只需要学习看谁都是反例的学习器，精度就到达99.8%了
- 再缩放（rescaling）,亦称“再平衡（rebalance）”

![20230608162940](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20230608162940.png)
![20230608163000](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20230608163000.png)
![20230608163203](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20230608163203.png)

**欠采样亦称“下采样”(downsampling)，过采样亦称“上采样”(upsam-pling)**