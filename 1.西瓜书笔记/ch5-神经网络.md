# 第五章 神经网络

1. 理想的激活函数是阶跃函数，但由于其数学性质不好，故而作为妥协，常用Sigmoid函数作为激活函数。由于其将较大范围的值**挤压到(0,1)输出值范围内**，因此有时也称为“挤压函数”（squashing function）
2. **感知机（Perceptron）由两层神经元组成**
3. 权重和阈值可统一为权重，进行学习。
4. 已经证明：若两类模式是线性可分的，即存在一个线性超平面能将它们分开，则感知机的学习过程一定会收敛（converge）；**否则感知机学习过程将会发生振荡（fluctuation）。**
5. 误差逆传播（error BackPropagation，简称BP），亦称“反向传播算法”.BP算法的目标是要最小化训练集D上的累积误差。
6. **标准的BP算法**每次仅针对一个训练样例更新连接权重和阈值。
7. **累积BP算法直接针对累积误差最小化，它在读取整个数据集D一遍后才对参数进行更新。**
![20230608200745](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20230608200745.png)
> 一言以蔽之，先用累积BP算法，再用标准BP算法。
> 读取训练集一遍称为进行了“一轮”(one round,亦称one epoch)学习.
> 标准BP算法和累积BP算法的区别类似于**随机梯度下降(stochastic gradientdescent,简称SGD)与标准梯度下降**之间的区别.

8. 已经证明，只要神经元的隐层够多，多层前馈网络就能以任意精度逼近任意复杂度的连续函数。**难点在于**，隐层神经元的个数如何设计最为合理。
9. BP网络由于过于强大，经常过拟合。对此，有两种策略进行缓解。
   1. 早停（early stopping）：将数据分成训练集和验证集，训练集用来计算梯度、更新连接权和阈值，验证集用来估计误差，**若训练集误差降低但验证集误差升高**，则停止训练，同时返回具有最小验证集误差的连接权和阈值.
   2. 正则化（regularization）：其基本思想是在误差目标函数中**增加一个用于描述网络复杂度的部分**，例如连接权与阈值的平方和。顺便一提，增加连接权与阈值平方和这一项后，训练过程将会偏好比较小的连接权重和阈值，使网络输出更加“光滑”，从而对过拟合有所缓解。

10. 有些网络是具有多个局部极小值的，此时无法保证找到的解是全局最小。有以下策略来帮助我们“跳出”局部极小，从而进一步接近全局最小。**但是也会造成“跳出”全局最小。**
![20230608203537](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20230608203537.png)
11. 常见的神经网络
   - RBF（Radial Basis Function，径向基函数）网络是一种**单隐层前馈神经网络**。已经证明，具有足够多隐层神经元的RBF网络能以任意精度逼近任意函数。
   - ART（Adaptive Resonance Theory，自适应谐振理论）网络是**竞争型学习**的重要代表。
     - 竞争型学习（competitive learning）是神经网络中一种常用的无监督学习策略，在使用该策略时，网络的输出神经元相互竞争，每一时刻仅有一个竞争获胜的神经元被激活，其他神经元的状态被抑制。这种机制亦称“胜者通吃”（winner-take-all）原则。
     - ART网络具有一个很重要的优点：可进行增量学习（incremental learning）或在线学习（online learning）
   - SOM（Self-Organizing Map，自组织映射）网络，亦称“自组织特征映射”(Self-Organizing Feature Map)、Kohonen 网络。是一种**竞争学习型的无监督神经网络**，它能将高维输入数据映射到低维空间（通常为二维），同时保持输入数据在高维空间的拓扑结构，即将高维空间中相似的样本点映射到网络输出层中的邻近神经元。
   - 级联相关网络是一种**结构自适应网络**，所谓结构自适应网络就是将网络结构也当作学习的目标之一，并希望能在训练过程中找到最符合数据特点的网络结构。
     - 与一般的前馈神经网络相比，**级联相关网络无需设置网络层数、隐层神经元数目，且训练速度较快，但其在数据较小时易陷入过拟合**
   - Elman网络是最常用的**递归神经网络**之一。所谓“递归神经网络”（recurrent neural networks）允许网络中出现**环形结构**，从而可让一些神经元的输出反馈回来作为输入信号。**从而能处理与事件有关的动态变化。
   - Boltzmann机
     - 神经网络中有一类模型是为网络状态定义一个“能量”(energy),能量最小化时网络达到理想状态，而网络的训练就是在最小化这个能量函数.Boltzmann机就是一种“基于能量的模型”(energy-based model)。事实上，Boltzmann机是一种递归神经网络。

12. **多隐层神经网络难以直接用经典算法（例如标准BP算法）进行训练，因为误差在多隐层内逆传播时，往往会“发散”（diverge）而不能收敛到稳定状态。**解决方案：
    - 无监督逐层训练(unsupervisedlayer-wisetraining)是多隐层网络训练的有效手段，其基本思想是每次训练一层隐结点，训练时将上一层隐结点的输出作为输入，而本层隐结点的输出作为下一层隐结点的输入,这称为“预训练”(pre-training); 在预训练全部完成后，再对整个网络进行“微调”(fine-tuning)训练.
      - 事实上，“预训练+微调”的做法可视为将大量参数分组，对每组先找到局部看来比较好的设置，然后再基于这些局部较优的结果联合起来进行全局寻优。这样就在利用了模型大量参数所提供的自由度的同时，有效地节省了训练开销。
    - 权共享（weight sharing）：即让一组神经元使用相同的连接权。

13. 可以从另一个角度来理解**深度学习**。无论是DBN还是CNN，其多隐层堆叠、每层对上一层的输出进行处理的机制，可看作是在对输入信号进行逐层加工，从而把初始的、与输出目标之间联系不太密切的输入表示，转化成与输出目标联系更密切的表示，使得原来仅基于最后一层输出映射难以完成的任务成为可能。换言之，通过多层处理，逐渐将初始的“低层”特征表示转化为“高层”特征表示后，用“简单模型”即可完成复杂的分类等学习任务。由此可将深度学习理解为进行“特征学习”(feature learning)或“表示学习”(representation learning).
