import torch
import random

def data_synthetic(w, b, num_examples):
    """定义数据集生成方法"""
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape(-1, 1)

"""生成数据集"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
feature, label = data_synthetic(true_w, true_b, 1000)

def data_iter(batch_size, feature, label):
    """生成小批量数据"""
    num_examples = len(feature)
    index = list(range(num_examples))
    random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        batch_index = torch.tensor(index[i:min(num_examples, i + batch_size)])
        yield feature[batch_index], label[batch_index]

"""初始化参数、特征值和标签"""
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(x, w, b):
    """定义线性回归模型"""
    return torch.matmul(x, w) + b
    
def squared_loss(y, y_hat):
    """定义损失函数"""
    return (y.reshape(y_hat.shape) - y_hat) ** 2 / 2
    
def sgd(params, batch_size, lr):
    """定义优化算法"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

"""定义超参"""
lr = 0.03 #学习率
batch_size = 10 #小批量数据大小
num_epochs = 3 #训练次数
net = linreg #模型方法
loss = squared_loss #损失函数

"""模型训练"""
for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, feature, label):
        l = loss(y, net(x, w, b))
        l.sum().backward()
        sgd([w, b], batch_size, lr)
    with torch.no_grad():
        train_l = loss(net(feature, w, b), label)
        print("epoch %d, loss %f" % (epoch + 1, train_l.mean()))
