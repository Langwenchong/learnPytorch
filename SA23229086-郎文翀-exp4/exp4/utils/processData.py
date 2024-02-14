import torch
import numpy as np
import pandas as pd
import random
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data


def getDataLoader(data_name,batch_size=32):
    if data_name in ["cora","citeseer"]:
        data,labels = get_data(data_name)
        # 每两个阶段关注的邻居数量都限制一定数量，以防节点太多关注度太低，逐layer解放数量从局部到全局
        # 这里的batch_size只是决定此次输入的初始节点，然后+neighbors节点才是这个batch图所涉及的节点和边,mask决定哪些点可以被选
        train_loader = NeighborLoader(data, num_neighbors=[10]*2, shuffle=True, input_nodes=data.train_mask, batch_size=batch_size)
        val_loader = NeighborLoader(data, num_neighbors=[10]*2, input_nodes=data.val_mask, batch_size=batch_size)
        test_loader = NeighborLoader(data, num_neighbors=[10]*2, input_nodes=data.test_mask, batch_size=batch_size)
    return data,labels,train_loader,val_loader,test_loader

def get_data(data_name,train_ratio=0.6,val_ratio=0.3,test_ratio=0.1):
    data_edge_path = f'data/{data_name}/{data_name}.cites'
    data_content_path = f'data/{data_name}/{data_name}.content'

    # 读入数据
    raw_content = pd.read_csv(data_content_path, header=None, sep='\t', low_memory=False)
    raw_edge = pd.read_csv(data_edge_path, header=None, sep='\t', low_memory=False)

    # 每一个paper的id都赋予一个唯一的自然idx索引
    paper_ids = raw_content[0]
    paper_id_map = {}
    for i, pp_id in enumerate(paper_ids):
        paper_id_map[pp_id] = i

    # 存储edge中边的信息
    edge_index = torch.from_numpy(raw_edge.apply(lambda col: col.map(paper_id_map)).dropna().values).long().t().contiguous()
    # x存储的是每一个节点的feature，即中间的二进制编码数据
    x = torch.from_numpy(raw_content.values[:, 1:-1].astype(np.float64)).float()
    # 存储的是类别，要注意这里存储的只是cora中7个类别，即label的class个数
    labels = np.unique(raw_content[raw_content.keys()[-1]]).tolist()
    # 这里存储的是每一个节点对应的类别索引
    y = torch.from_numpy(raw_content[raw_content.keys()[-1]].map(lambda x: labels.index(x)).values).long()
    
    # 使用掩码划分数据集
    def get_mask(y: torch.tensor, train_ratio, val_ratio, test_ratio):
        train_mask = torch.tensor([False] * y.shape[0])
        val_mask=torch.tensor([False] * y.shape[0])
        test_mask=torch.tensor([False] * y.shape[0])
        # 遍历每一个label类别
        for i in torch.unique(y).unbind():
            # 获取对应当前这个label的节点索引值，转换为列表，这样子划分是为了保证数据集的类别划分合理
            temp = torch.arange(0, y.shape[0])[y == i].tolist()
            length = len(temp)
            train_size = int(train_ratio * length)
            val_size = int(val_ratio * length)

            random.shuffle(temp)
            train_indices = temp[:train_size]
            val_indices = temp[train_size:train_size+val_size]
            test_indices = temp[train_size+val_size:]
            train_mask[train_indices] = True 
            val_mask[val_indices] = True
            test_mask[test_indices]=True
        
        return train_mask, val_mask, test_mask
    
    train_mask,val_mask,test_mask = get_mask(y,train_ratio,val_ratio,test_ratio)
    # 生成图数据，主要是点集，边集，lables标签类别以及划分数据的掩码
    data=Data(x=x,edge_index=edge_index,y=y,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)

    return data,labels
