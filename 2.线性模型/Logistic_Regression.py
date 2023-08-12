import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


# 定义逻辑回归模型类
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # 定义线性层，输入维度为 input_dim，输出维度为 1
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 使用 sigmoid 函数将线性输出转换为概率值
        return torch.sigmoid(self.linear(x))


def weight_init(m):
    # 初始化权重
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, val=0)


def train(net, train_X, train_y, test_X, test_y, num_epochs, lr, weight_decay):
    # 训练函数，用于训练神经网络模型并计算损失
    # 如果提供了测试数据，则返回训练集损失和测试集损失，否则返回训练集损失

    # 创建 SGD 优化器，用于参数更新
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

    # 在每个训练周期内进行训练和评估
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # 清零梯度
        l = loss(net(train_X), train_y)  # 计算损失
        l.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        # print('epoch %d, loss: %f' % (epoch + 1, l.item()))

    # 计算并记录训练集的误差
    train_loss = loss(net(train_X), train_y)

    # 如果提供了测试集标签，则计算并记录测试集的误差
    if test_y is not None:
        test_loss = loss(net(test_X), test_y)

        return train_loss, test_loss
    
    return train_loss


def get_data():
    # 使用 pandas 读取数据并处理数据，返回特征张量和标签张量。

    data_path = r"2.线性模型/watermelon3_0_Ch.csv"
    data = pd.read_csv(data_path)

    # 在每个样本中，第一个特征是 编号，我们将其从数据集中删除
    all_features = data.iloc[:, 1:-1]

    # 处理离散值，用one-hot编码
    all_features = pd.get_dummies(all_features, dummy_na=True)

    features = torch.tensor(all_features.values, dtype=torch.float32)
    # 将标签列的 "是" 转换为 1，"否" 转换为 0
    labels = data["好瓜"].apply(lambda x: 1 if x == "是" else 0)
    labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    
    return features, labels


def get_k_fold_data(k, i, X, y):
    """获取第 i 折交叉验证时所需要的训练和验证数据

    Args:
        k (int): 折数
        i (int): 当前折索引
        X (torch.Tensor): 特征数据
        y (torch.Tensor): 标签数据

    Returns:
        X_train (torch.Tensor): 训练集特征数据
        y_train (torch.Tensor): 训练集标签数据
        X_valid (torch.Tensor): 验证集特征数据
        y_valid (torch.Tensor): 验证集标签数据
    """
    assert k > 1  # 确保折数大于 1

    fold_size = X.shape[0] // k  # 每折的样本数
    X_train, y_train = None, None  # 初始化训练集

    # 迭代每一折，获取训练集和验证集
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 当前折的索引范围
        X_part, y_part = X[idx, :], y[idx]  # 当前折的数据

        if j == i:  # 当前折为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:  # 训练集的第一个折
            X_train, y_train = X_part, y_part
        else:  # 训练集的其他折
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    return X_train, y_train, X_valid, y_valid



def k_fold(k, features, labels, num_epochs, learning_rate, weight_decay):
    # 执行 k 折交叉验证来评估神经网络模型的性能
    train_l_sum, valid_l_sum = 0, 0  # 初始化训练集和验证集的损失和

    # 循环每一折
    for i in range(k):
        data = get_k_fold_data(k, i, features, labels)  # 获取当前折的训练集和验证集
        net = LogisticRegressionModel(features.shape[1])  # 重新创建逻辑回归模型
        # net.apply(weight_init)  # 初始化模型参数
        train_loss, valid_loss = train(net, *data, num_epochs, learning_rate, weight_decay)  # 训练并返回损失

        train_l_sum += train_loss  # 训练集损失
        valid_l_sum += valid_loss # 验证集损失
        
        print('fold %d, train loss %f, valid loss %f' % (i, train_loss, valid_loss))  # 打印每一折的损失
    
    average_train_l = train_l_sum / k  # 训练集损失均值
    average_valid_l = valid_l_sum / k  # 验证集损失均值

    return average_train_l, average_valid_l


features, labels = get_data()
loss = nn.BCELoss()  # 二元交叉熵损失函数
num_epochs, lr, weight_decay = 100, 0.07, 1e-2  # 训练周期数、学习率、权重衰减系数

avg_train_loss, avg_valid_loss = k_fold(17,features, labels, num_epochs, lr, weight_decay)
print(f"训练集平均损失：{avg_train_loss:.6f}")
print(f"验证集平均损失：{avg_valid_loss:.6f}")