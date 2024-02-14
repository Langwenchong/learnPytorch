import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils.undirected import to_undirected
import random

# 预测节点所属的分类
class GCNNodeClassifier(torch.nn.Module):
    def __init__(self,
                #  输入的节点初始特征维度
                 dim_features,
                #  类别
                 num_classes,
                #  模型深度
                 num_layers,
                #  节点是否具有回环
                 add_self_loops: bool = True,
                #  正则化
                 use_pairnorm: bool = False,
                #  丢弃边
                 drop_edge: float = 1.,
                #  激活函数类型
                 activation: str = 'relu',
                #  图是否为无向图
                 # undirected: bool = False
                 ):
        super(GCNNodeClassifier, self).__init__()
        # 默认中间隐层节点特征维度为2048
        dim_hidden = 2048

        self.gconvs = torch.nn.ModuleList(
            # 首层特殊处理，因为输入特征维度不同
            [GCNConv(in_channels=dim_features, out_channels=dim_hidden, add_self_loops=add_self_loops)]
            # 最后层特殊处理
            + [GCNConv(in_channels=dim_hidden, out_channels=dim_hidden, add_self_loops=add_self_loops) for i in
               range(num_layers - 2)]
        )
        # 最后一层需要将特征维度线性映射到类别个数
        self.final_conv = GCNConv(in_channels=dim_hidden, out_channels=num_classes, add_self_loops=add_self_loops)

        self.use_pairnorm = use_pairnorm
        if self.use_pairnorm:
            self.pairnorm = PairNorm()

        self.drop_edge = drop_edge

        activations_map = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
                           'leaky_relu': torch.nn.LeakyReLU(0.1)}
        
        self.activation_fn = activations_map[activation]

    def forward(self, x, edge_index, device):
        # 按照一定概率随机丢弃边，注意这里是保留概率，所以丢弃概率为1-keep_ratio
        def drop_edge(edge_index, keep_ratio: float = 1.):
            # 要保留的边个数
            num_keep = int(keep_ratio * edge_index.shape[1])
            temp = [True] * num_keep + [False] * (edge_index.shape[1] - num_keep)
            # 随机选择保留的边
            random.shuffle(temp)
            return edge_index[:, temp]
        
        # 遍历每一层
        for l in self.gconvs:
            edges = drop_edge(edge_index, self.drop_edge).to(device)
            # 每层的输入是节点和边
            x = l(x, edges)
            if self.use_pairnorm:
                x = self.pairnorm(x)
            x = self.activation_fn(x)
        x = self.final_conv(x, edge_index)

        return x