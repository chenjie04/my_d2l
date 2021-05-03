import torch 
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
from d2l import torch as d2l

trans = transforms.ToTensor()
mnsit_train = torchvision.datasets.FashionMNIST(root='./data',train=True,transform=trans,download=False)
mnist_test = torchvision.datasets.FashionMNIST(root='./data',train=False,transform=trans,download=False)

batch_size = 256
train_iter = data.DataLoader(mnsit_train,batch_size,shuffle=True,num_workers=4)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=4)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp / partition

# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W) + b)

# 定义损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

# 定义评价指标
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y 
    return float(cmp.type(y.dtype).sum())

# 定义累加器
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# 评估数据迭代器 data_iter 访问的数据集在任意模型 net 上的准确率
def evaluate_accuracy(net, data_iter):
    """计算模型在指定数据集上的精度"""
    metric = Accumulator(2) # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0] / metric[1]



# 训练
lr = 0.1
num_epochs = 10




for epoch in range(num_epochs):
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = cross_entropy(y_hat, y)
        l.sum().backward()
        d2l.sgd([W,b],lr,batch_size)
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    test_acc = evaluate_accuracy(net, test_iter)
    print(f'epch {epoch +1}, train loss {metric[0] / metric[2] :.5f}, train acc {metric[1] / metric[2] :.5f}, test acc {test_acc:.5f}')

