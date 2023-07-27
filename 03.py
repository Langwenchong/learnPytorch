# import matplotlib.pyplot as plt
#
# # prepare the training set
# x_data = [1.0, 2.0, 3.0]
# y_data = [2.0, 4.0, 6.0]
#
# # initial guess of weight
# w = 1.0
#
#
# # define the model linear model y = w*x
# def forward(x):
#     return x * w
#
#
# # define the cost function MSE
# def cost(xs, ys):
#     cost = 0
#     for x, y in zip(xs, ys):
#         y_pred = forward(x)
#         cost += (y_pred - y) ** 2
#     return cost / len(xs)
#
#
# # define the gradient function  gd
# def gradient(xs, ys):
#     grad = 0
#     for x, y in zip(xs, ys):
#         grad += 2 * x * (x * w - y)
#     return grad / len(xs)
#
#
# epoch_list = []
# cost_list = []
# print('predict (before training)', 4, forward(4))
# for epoch in range(100):
#     cost_val = cost(x_data, y_data)
#     grad_val = gradient(x_data, y_data)
#     w -= 0.01 * grad_val  # 0.01 learning rate
#     print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
#     epoch_list.append(epoch)
#     cost_list.append(cost_val)
#
# print('predict (after training)', 4, forward(4))
# plt.plot(epoch_list, cost_list)
# plt.ylabel('cost')
# plt.xlabel('epoch')
# plt.show()

# 随机梯度下降
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


# calculate loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# define the gradient function  sgd
def gradient(x, y):
    return 2 * x * (x * w - y)


epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad  # update weight by every grad of sample of training set
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()