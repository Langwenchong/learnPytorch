import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取原始数据，并划分训练集和测试集
raw_data = np.loadtxt('./diabetes.csv/diabetes.csv', delimiter=',', dtype=np.float32)
X = raw_data[:, :-1]
y = raw_data[:, [-1]]
# 切割后30%是测试数据
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)
Xtest = torch.from_numpy(Xtest)
Xtest = Xtest.to(device)
Ytest = torch.from_numpy(Ytest)
Ytest = Ytest.to(device)

# 将训练数据集进行批量处理
# prepare dataset


class DiabetesDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = DiabetesDataset(Xtrain, Ytrain)
train_loader = DataLoader(dataset=train_dataset, batch_size=64,
                          shuffle=True, num_workers=0)  # num_workers 多线程

# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = Model()
model.to(device)

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
criterio = criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# training cycle forward, backward, update
loss_list = []
epoch_list = []
acc_list = []


def train(epoch):
    train_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_pred = model(inputs)

        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        count = i

    if epoch % 2000 == 1999:
        loss_list.append(train_loss)
        epoch_list.append(epoch)
        print(epoch,"train loss:", train_loss/count)

# 如果当前是一个最新的acc峰值，那么记录参数
name_list=[]
param_list=[]
def test():
    with torch.no_grad():
        # 直接batch一次性全预测，因此行数应为30%，列特征从8维变为1维
        y_pred = model(Xtest)
        # 最终的预测值介于[0,1]因此我们需要给一个阈值，如果是>=0.5则为1，否则为0
        y_pred=y_pred.to(device)
        y_pred_label = torch.where(
            y_pred >= 0.5, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
        # 因此Ytest.size(0)也可以写为Xtest.size(0)
        acc = torch.eq(y_pred_label, Ytest).sum().item() / Ytest.size(0)
        if  len(acc_list) == 0 or acc > acc_list[len(acc_list)-1]:
            # 首先清空之前的记录
            name_list.clear()
            param_list.clear()
            # 重新加入参数
            for name,param in model.named_parameters():
                name_list.append(name)
                param_list.append(param)
        acc_list.append(acc)
        print(f"test acc:{100*acc}%")
 

if __name__ == '__main__':
    for epoch in range(50000):
        train(epoch)
        if epoch % 2000 == 1999:
            test()
    # 输出当前acc峰值时对应的参数
    for idx in range(0,len(name_list)):
        print(name_list[idx],param_list[idx])
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(epoch_list, loss_list, 'r-')
    ax2.plot(epoch_list, acc_list, 'g--')

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color='r')
    ax2.set_ylabel("accuracy(%)", color='g')
    plt.show()
