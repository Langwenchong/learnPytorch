import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

# 自定义数据目录
data_path = 'data/tiny-imagenet-200/'


# 定义数据类处理文件
class RawData:

    __labels_t_path = '%s%s' % (data_path, 'wnids.txt')
    __train_data_path = '%s%s' % (data_path, 'train/')
    __val_data_path = '%s%s' % (data_path, 'val/')

    # 原数据集中的train下的<类别id>数组与<<类别A所有图片>,<类别B所有图片>>数组
    __labels_t = None
    __image_names = None

    # 原数据集中的val下的<类别id>数组,<类别id>数组(?:这里貌似是为了保证一定是训练集中有的类别),图片数组
    __val_labels_t = None
    __val_labels = None
    __val_names = None

    @staticmethod
    def labels_t():
        # 加载wnids文件读取不同类的id，之后可以通过查找words.txt知道对应的类别
        if RawData.__labels_t is None:
            labels_t = []
            with open(RawData.__labels_t_path) as wnid:
                for line in wnid:
                    labels_t.append(line.strip('\n'))

            RawData.__labels_t = labels_t

        return RawData.__labels_t

    @staticmethod
    def image_names():
        if RawData.__image_names is None:
            image_names = []
            labels_t = RawData.labels_t()
            for label in labels_t:
                # 获取到某一个类别的图片集中的对应的检测框文件
                txt_path = RawData.__train_data_path + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        # 这里只做分类，所以只是提取这个类别中图片的名字
                        image_name.append(line.strip('\n').split('\t')[0])
                # iamge_names存储了许多不同类的图片数据集
                # imager_name也是一个数组存储的某一个类别的多张图片
                image_names.append(image_name)

            RawData.__image_names = image_names

        return RawData.__image_names

    @staticmethod
    def val_labels_t():
        if RawData.__val_labels_t is None:
            val_labels_t = []
            with open(RawData.__val_data_path + 'val_annotations.txt') as txt:
                for line in txt:
                    # 存储验证集但是这里视为测试集的label id
                    val_labels_t.append(line.strip('\n').split('\t')[1])

            RawData.__val_labels_t = val_labels_t

        return RawData.__val_labels_t

    @staticmethod
    def val_names():
        if RawData.__val_names is None:
            val_names = []
            with open(RawData.__val_data_path + 'val_annotations.txt') as txt:
                for line in txt:
                    # 存储验证集但是这里视为测试集的图片id
                    val_names.append(line.strip('\n').split('\t')[0])

            RawData.__val_names = val_names

        return RawData.__val_names

    @staticmethod
    def val_labels():
        if RawData.__val_labels is None:
            val_labels = []
            val_labels_t = RawData.val_labels_t()
            labels_t = RawData.labels_t()
            # 这里是为了保证验证集中的类别id与测试集中的类别id相对应
            for i in range(len(val_labels_t)):
                for i_t in range(len(labels_t)):
                    if val_labels_t[i] == labels_t[i_t]:
                        # 注意不是存储label而时存储label的索引,因为我们可以直接通过labels_t[idx]获取到类别
                        val_labels.append(i_t)
            val_labels = np.array(val_labels)

            RawData.__val_labels = val_labels

        return RawData.__val_labels


# 定义 Dataset 类
class Data(Dataset):

    def __init__(self, type_, transform):
        """
        type_: 选择训练集还是验证集
        """
        self.__train_data_path = '%s%s' % (data_path, 'train/')
        self.__val_data_path = '%s%s' % (data_path, 'val/')

        self.type = type_

        self.labels_t = RawData.labels_t()
        self.image_names = RawData.image_names()
        self.val_names = RawData.val_names()

        self.transform = transform

    def __getitem__(self, index):
        label = None
        image = None

        labels_t = self.labels_t
        image_names = self.image_names
        val_labels = RawData.val_labels()
        val_names = self.val_names

        if self.type == "train":
            # label实际上就是labels_t的索引值
            label = index // 500  # 每个类别的图片 500 张
            remain = index % 500
            # 此时image_names中的图片是顺序排列的,即一类图片放在一起,这里是按照该index去取,某一个类中的第remain张图片
            image_path = os.path.join(
                self.__train_data_path, labels_t[label], 'images', image_names[label][remain])
            image = cv2.imread(image_path)
            # 修改分辨率
            image = np.array(image).reshape(64, 64, 3)

        elif self.type == "val":
            label = val_labels[index]
            val_image_path = os.path.join(
                self.__val_data_path, 'images', val_names[index])
            image = np.array(cv2.imread(val_image_path)).reshape(64, 64, 3)

        return label, self.transform(image)

    def __len__(self):
        len_ = 0
        if self.type == "train":
            # 类别数量*每一个类中的图片个数
            len_ = len(self.image_names) * len(self.image_names[0])
        elif self.type == "val":
            len_ = len(self.val_names)

        return len_


class residual_block(nn.Module):
    """残差网络
    """

    def __init__(self, channel, dropout=False, normalize=False):
        super(residual_block, self).__init__()
        self.normalize = normalize

        # 3×3卷积,但是保证通道数不变,数据的特征维度不变
        self.cov1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.cov2 = nn.Conv2d(channel, channel, 3, 1, 1)

        if normalize:
            self.nor1 = nn.BatchNorm2d(channel)
            self.nor2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()
        self.dropout = False
        if dropout:
            # 0.2的概率不激活当前层的该神经元,防止过拟合
            self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x_ = self.cov1(x)
        if self.normalize:
            x_ = self.nor1(x_)
        x_ = self.relu(x_)

        x_ = self.cov2(x_)
        if self.normalize:
            x_ = self.nor2(x_)
        x_ = self.relu(x_)

        if self.dropout:
            x_ = self.drop(x_)
        # 直接数值相加
        x = x + x_
        return x


class nor_cov(nn.Module):
    """单层卷积网络
    """

    def __init__(self, in_channel, out_channel, dropout=False, normalize=False):
        super(nor_cov, self).__init__()

        # 3×3卷积实现,只是通道数会发生改变,数据特征维度不变
        self.cov = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

        self.normalize = normalize
        if normalize:
            self.nor = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()
        self.dropout = False
        if dropout:
            self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.cov(x)

        if self.normalize:
            x = self.nor(x)
        x = self.relu(x)

        if self.dropout:
            x = self.drop(x)

        return x


class dou_cov(nn.Module):
    """双层卷积网络
    """

    def __init__(self, channel, dropout=False, normalize=False):
        super(dou_cov, self).__init__()
        self.cov1 = nor_cov(in_channel=channel, out_channel=channel,
                            dropout=dropout, normalize=normalize)
        self.cov2 = nor_cov(in_channel=channel, out_channel=channel,
                            dropout=dropout, normalize=normalize)

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        return x


class CNN_net(nn.Module):
    """自定义卷积网络
    """

    def __init__(self, normalize=False):
        super(CNN_net, self).__init__()

        self.cov1 = nor_cov(in_channel=3, out_channel=64,
                            dropout=True, normalize=normalize)
        self.dou_cov1 = dou_cov(channel=64, dropout=True, normalize=True)
        self.dou_cov2 = dou_cov(channel=64, dropout=True, normalize=True)
        # self.res1 = residual_block(64, dropout=False)
        self.res2 = residual_block(64, dropout=True, normalize=normalize)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.cov2 = nor_cov(in_channel=64, out_channel=128,
                            dropout=True, normalize=normalize)
        self.dou_cov3 = dou_cov(channel=128, dropout=True, normalize=True)
        self.dou_cov4 = dou_cov(channel=128, dropout=True, normalize=True)
        # self.res3 = residual_block(128, dropout=False)
        self.res4 = residual_block(128, dropout=True, normalize=normalize)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.cov3 = nor_cov(in_channel=128, out_channel=256,
                            dropout=True, normalize=normalize)
        self.dou_cov5 = dou_cov(channel=256, dropout=True, normalize=True)
        self.dou_cov6 = dou_cov(channel=256, dropout=True, normalize=True)
        # self.res5 = residual_block(256, dropout=False)
        self.res6 = residual_block(256, dropout=True, normalize=normalize)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.cov4 = nor_cov(in_channel=256, out_channel=512,
                            dropout=True, normalize=normalize)
        self.dou_cov7 = dou_cov(channel=512, dropout=True, normalize=True)
        self.dou_cov8 = dou_cov(channel=512, dropout=True, normalize=True)
        # self.res7 = residual_block(512, dropout=False)
        self.res8 = residual_block(512, dropout=True, normalize=normalize)

        self.pool4 = nn.MaxPool2d(2, 2)

        self.cov5 = nor_cov(in_channel=512, out_channel=256,
                            dropout=True, normalize=normalize)
        self.dou_cov9 = dou_cov(channel=256, dropout=True, normalize=True)
        self.dou_cov10 = dou_cov(channel=256, dropout=True, normalize=True)
        # self.res9 = residual_block(256, dropout=False)
        self.res10 = residual_block(256, dropout=True, normalize=normalize)

        self.pool5 = nn.MaxPool2d(2, 2)

        self.cov6 = nor_cov(in_channel=256, out_channel=128,
                            dropout=True, normalize=normalize)
        self.dou_cov11 = dou_cov(channel=128, dropout=True, normalize=True)
        self.dou_cov12 = dou_cov(channel=128, dropout=True, normalize=True)
        # self.res11 = residual_block(128, dropout=False)
        self.res12 = residual_block(128, dropout=True, normalize=normalize)

        self.pool6 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 1 * 1, 200)

    def forward(self, x):
        x = self.cov1(x)
        # x = self.dou_cov1(x)
        # x = self.dou_cov2(x)
        # x = self.res1(x)
        x = self.res2(x)
        x = self.pool1(x)

        x = self.cov2(x)
        # x = self.dou_cov3(x)
        # x = self.dou_cov4(x)
        # x = self.res3(x)
        x = self.res4(x)
        x = self.pool2(x)

        x = self.cov3(x)
        # x = self.dou_cov5(x)
        # x = self.dou_cov6(x)
        # x = self.res5(x)
        x = self.res6(x)
        x = self.pool3(x)

        x = self.cov4(x)
        # x = self.dou_cov7(x)
        # x = self.dou_cov8(x)
        # x = self.res7(x)
        x = self.res8(x)
        x = self.pool4(x)

        x = self.cov5(x)
        # x = self.dou_cov9(x)
        # x = self.dou_cov10(x)
        # x = self.res9(x)
        x = self.res10(x)
        x = self.pool5(x)

        x = self.cov6(x)
        # x = self.dou_cov11(x)
        # x = self.dou_cov12(x)
        # x = self.res11(x)
        x = self.res12(x)
        x = self.pool6(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)

        return x
