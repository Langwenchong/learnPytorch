import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from torch.cuda import set_device
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# prepare dataset


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        # 这里是加载整个数据，因此是一个矩阵形式
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # 这里列数是特征值，因此len返回样本数量应该是行数
        self.len = xy.shape[0]  # shape(多少行，多少列)
        # 这里是切割成两组矩阵，默认会按照该（x_data,y_data）返还数据
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('./diabetes.csv/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32,
                          shuffle=True, num_workers=0)  # num_workers 多线程


# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
model = model.to(device)

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
criterion = criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
lossList = []
accuracyList = []
# training cycle forward, backward, update
if __name__ == '__main__':
    for epoch in range(10000):
        total_loss = 0.0
        correct = 0.0
        # mini-batch的体现
        # train_loader 是先shuffle后mini_batch
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            total_loss += loss.item()
            # print(y_pred)
            y_tmp = torch.where(y_pred > 0.6, y_pred, torch.zeros_like(y_pred))
            y_tmp = torch.where(y_tmp <= 0.6, y_tmp, torch.ones_like(y_tmp))
            # print(y_tmp)
            correct += (y_tmp.eq(labels)).type(torch.float).sum().item()
            # print(epoch, i, loss.item())
            # print(correct)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lossList.append(total_loss)
        accuracy = 100*correct/len(dataset)
        accuracyList.append(accuracy)
        # print(torch.cuda.current_device())
        print("epoch=", epoch, "accuracy=",
              f"{accuracy}%", "totalLoss=", total_loss)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(range(10000), lossList, 'r-')
ax2.plot(range(10000), accuracyList, 'g--')

ax1.set_xlabel("epoch")
ax1.set_ylabel("loss", color='r')
ax2.set_ylabel("accuracy(%)", color='g')
plt.show()
