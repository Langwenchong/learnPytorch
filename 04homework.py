# import numpy as np
# import matplotlib.pyplot as plt
# import torch

# x_data = [1.0,2.0,3.0]
# y_data = [2.0,4.0,6.0]

# w = torch.Tensor([1.0])#初始权值
# w.requires_grad = True#计算梯度，默认是不计算的

# def forward(x):
#     return x * w

# def loss(x,y):#构建计算图
#     y_pred = forward(x)
#     return (y_pred-y) **2

# print('Predict (befortraining)',4,forward(4))

# for epoch in range(100):
#     l = loss(1, 2)#为了在for循环之前定义l,以便之后的输出，无实际意义
#     for x,y in zip(x_data,y_data):
#         l = loss(x, y)
#         l.backward()
#         print('\tgrad:',x,y,w.grad.item())
#         w.data = w.data - 0.01*w.grad.data #注意这里的grad是一个tensor，所以要取他的data
#         w.grad.data.zero_() #释放之前计算的梯度
#     print('Epoch:',epoch,l.item())

# print('Predict(after training)',4,forward(4).item())

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])  # 初始权值
w1.requires_grad = True  # 计算梯度，默认是不计算的
w2 = torch.Tensor([1.0])
w2.requires_grad = True
b = torch.Tensor([1.0])
b.requires_grad = True


def forward(x):
    return w1 * x**2 + w2 * x + b


def loss(x, y):  # 构建计算图
    y_pred = forward(x)
    return (y_pred-y) ** 2

# print('Predict (befortraining)',4,forward(4))

# for epoch in range(100):
#     l = loss(1, 2)#为了在for循环之前定义l,以便之后的输出，无实际意义
#     for x,y in zip(x_data,y_data):
#         l = loss(x, y)
#         l.backward()
#         print('\tgrad:',x,y,w1.grad.item(),w2.grad.item(),b.grad.item())
#         w1.data = w1.data - 0.01*w1.grad.data #注意这里的grad是一个tensor，所以要取他的data
#         w2.data = w2.data - 0.01 * w2.grad.data
#         b.data = b.data - 0.01 * b.grad.data
#         w1.grad.data.zero_() #释放之前计算的梯度
#         w2.grad.data.zero_()
#         b.grad.data.zero_()
#     print('Epoch:',epoch,l.item())


# print('Predict(after training)',4,forward(4).item())
mse_list = []
rate = 0.01
for epoch in range(100):
    cost = 0
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('grad:', x, y, 'w1:', w1.grad.item(),
              'w2:', w2.grad.item(), 'b:', b.grad.item())
        print('\tvalue:', x, y, 'w1:', w1.item(),
              'w2:', w2.item(), 'b:', b.item())
        cost += l.item()
        w1.data -= rate*w1.grad.data
        w2.data -= rate*w2.grad.data
        b.data -= rate*b.grad.data

        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()
    mse_list.append(cost/len(x_data))
    print('progress:', epoch, mse_list[epoch])

print('predict y=', forward(4).item(), 'when x=4')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.plot(range(100), mse_list)
plt.show()

