import torch
# import torch.nn.functional as F
 
# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
 
#design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
 
    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()
 
# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average = False) 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
 
# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


# bceloss大致原理
# import math
# import torch
# pred = torch.tensor([[-0.2],[0.2],[0.8]])
# target = torch.tensor([[0.0],[0.0],[1.0]])
 
# sigmoid = torch.nn.Sigmoid()
# pred_s = sigmoid(pred)
# print(pred_s)
# """
# pred_s 输出tensor([[0.4502],[0.5498],[0.6900]])
# 0*math.log(0.4502)+1*math.log(1-0.4502)
# 0*math.log(0.5498)+1*math.log(1-0.5498)
# 1*math.log(0.6900) + 0*log(1-0.6900)
# """
# result = 0
# i=0
# for label in target:
#     if label.item() == 0:
#         result +=  math.log(1-pred_s[i].item())
#     else:
#         result += math.log(pred_s[i].item())
#     i+=1
# result /= 3
# print("bce：", -result)
# loss = torch.nn.BCELoss()
# print('BCELoss:',loss(pred_s,target).item())