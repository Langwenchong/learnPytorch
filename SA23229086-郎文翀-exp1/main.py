import os
import torch
import math
import datetime

from torch import nn
from torch import optim
from torch import utils
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用来记录最小loss时的参数
min_loss = 0x3f3f3f
# tensorboard日志存储文件夹
logdir = "./logs"
# 训练epoch次数
epochs = 1000
# 对于sinx采样频率
total_size = 10000
# 训练集/验证集/测试集的划分比例
train_size, val_size, test_size = 0.6, 0.2, 0.2
# batch分批次训练
batch_size = 64
# 定义中间层的宽度与深度
layers = [8, 16, 8]
# 初始学习率设置
lr, wd = 1e-4, 5e-4
# 每隔le_period epoch后学习率×decay
lr_period, lr_decay = 250, 0.8
# mse损失
criterion = nn.MSELoss().to(device)

# 采样数据
X = torch.linspace(0, 2*math.pi, total_size, dtype=torch.float32)
Y = torch.sin(X)
X = torch.tensor(X, dtype=torch.float32).clone().detach()
Y = torch.tensor(Y, dtype=torch.float32).clone().detach()
dataSet = TensorDataset(X, Y)


train_data, val_data, test_data = torch.utils.data.random_split(dataSet, [int(train_size*total_size), int(
    val_size*total_size), int(total_size - int(train_size * total_size) - int(val_size * total_size))])

train_dataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_dataLoader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataLoader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
len_train_dataLoader = len(train_dataLoader)


class Model(nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.fc = nn.Sequential()
        self.layers = layers
        # 初始数据为[bs,1]
        for i in range(len(self.layers)):
            if i == 0:
                self.fc.add_module(f'Linear{i+1}', nn.Linear(1, layers[0]))
            else:
                self.fc.add_module(
                    f'Linear{i+1}', nn.Linear(layers[i-1], layers[i]))
            # self.fc.add_module(f'Relu{i+1}', nn.ReLU())
        self.fc.add_module(f'Linear_end', nn.Linear(layers[-1], 1))

    def forward(self, x):
        return self.fc(x)


def get_trainer(model, lr, wd):
    return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd, amsgrad=True)


def get_scheduler(trainer, lr_decay):
    return optim.lr_scheduler.StepLR(trainer, lr_period*len_train_dataLoader, lr_decay)


def train(model, train_dataLoader, val_dataLoader, epochs, criterion, trainer, scheduler, min_loss):
    # 每条线对应一个SummaryWriter实例
    writer = {
        'train_loss': SummaryWriter("./logs/train_loss"),
        'val_loss': SummaryWriter("./logs/val_loss"),
        'batch_loss': SummaryWriter("./logs/batch_loss"),
        'lr': SummaryWriter("./logs/lr")
    }
    for epoch in range(epochs):
        model.train()
        # 记录每个epoch的训练集的损失与验证集损失
        train_loss = 0.0
        val_loss = 0.0
        # 记录每30个batch后的均损失
        batch_loss = 0.0
        # 进度条显示
        loop = tqdm(enumerate(train_dataLoader), total=len(train_dataLoader))
        # 此时只是更新进度条进度,因此每一个epoch对应一个新的进度条
        for batch_idx, data in loop:
            x, y = data
            trainer.zero_grad()
            # [bs]->[bs,1]
            x = x.unsqueeze(1).to(device)
            y = y.unsqueeze(1).to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            trainer.step()
            # scheduler.step()
            train_loss += loss.item()
            batch_loss += loss.item()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(train_loss=loss.item())
            if batch_idx % 30 == 29:
                writer['batch_loss'].add_scalar(
                    "train loss/30 batch", batch_loss/30, epoch*len(train_dataLoader)+batch_idx+1)
                writer['lr'].add_scalar(
                    "lr/30 batch", trainer.param_groups[0]['lr'], epoch*len(train_dataLoader)+batch_idx+1)
                batch_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_dataLoader:
                x = x.unsqueeze(1).to(device)
                y = y.unsqueeze(1).to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        # 记录验证集表现最好时的模型参数用于测试集
        if val_loss < min_loss:
            model_path = './myModel.pth'
            min_loss = val_loss
            torch.save(model.state_dict(), model_path)
        if epoch % 100 == 99:
            writer['train_loss'].add_scalar(
                "train&val loss/100 epoch", train_loss, epoch+1)
            writer['val_loss'].add_scalar(
                "train&val loss/100 epoch", val_loss, epoch+1)

# 用最好的一次参数模型跑测试集


def test(model, test_dataLoader, criterion):
    model.eval()
    test_loss = 0.0
    for x, y in test_dataLoader:
        x = x.unsqueeze(1).to(device)
        y = y.unsqueeze(1).to(device)
        y_pred = model(x)
        test_loss += criterion(y_pred, y)
    print('test loss: %.7f' % test_loss)


if __name__ == '__main__':
    model = Model(layers).to(device)
    print(model)
    trainer = get_trainer(model, lr, wd)
    scheduler = get_scheduler(trainer, lr_decay)
    train(model, train_dataLoader, val_dataLoader,
          epochs, criterion, trainer, scheduler, min_loss)
    try:
        model.load_state_dict(torch.load('./myModel.pth'))
        test(model, test_dataLoader, criterion)
    except:
        print("Load Model Parameters Failed!")
