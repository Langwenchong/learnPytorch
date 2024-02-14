import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils.undirected import to_undirected
import random


class GCNEdgePrediction(torch.nn.Module):
    def __init__(self,
                 dim_features,
                 num_classes,
                 num_layers,
                 add_self_loops: bool = True,
                 use_pairnorm: bool = False,
                 drop_edge: float = 1.,
                 activation: str = 'relu',
                 ):
        super(GCNEdgePrediction, self).__init__()
        dim_hidden = 128

        self.gconvs = torch.nn.ModuleList(
            [GCNConv(in_channels=dim_features, out_channels=dim_hidden, add_self_loops=add_self_loops)]
            + [GCNConv(in_channels=dim_hidden, out_channels=dim_hidden, add_self_loops=add_self_loops)
               for i in range(num_layers - 2)]
        )
        self.final_conv = GCNConv(
            in_channels=dim_hidden, out_channels=num_classes, add_self_loops=add_self_loops)

        self.use_pairnorm = use_pairnorm
        if self.use_pairnorm:
            self.pairnorm = PairNorm()
        self.drop_edge = drop_edge
        activations_map = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
                           'leaky_relu': torch.nn.LeakyReLU(0.1)}
        self.activation_fn = activations_map[activation]

    def encode(self, x, edge_index, device):

        # 按照一定概率随机丢弃边，注意这里是保留概率，所以丢弃概率为1-keep_ratio
        def drop_edge(edge_index, keep_ratio: float = 1.):
            # 要保留的边个数
            num_keep = int(keep_ratio * edge_index.shape[1])
            temp = [True] * num_keep + [False] * (edge_index.shape[1] - num_keep)
            # 随机选择保留的边
            random.shuffle(temp)
            return edge_index[:, temp]
        
    
        for l in self.gconvs:
            edges = drop_edge(edge_index, self.drop_edge).to(device)
            x = l(x, edges)
            if self.use_pairnorm:
                x = self.pairnorm(x)
            x = self.activation_fn(x)
        x = self.final_conv(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # 余弦相似度，其实就是计算当前这个边时两个节点的相似度
        edge_label_index = edge_label_index.type(torch.long)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim = -1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple = False).t()