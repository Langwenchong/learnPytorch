import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
from models import GCNNodeClassifier,GCNEdgePrediction
from utils import getDataLoader
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_cases = [ 
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu'},
    # num layers
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu'},
    {'num_layers':6, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu'},
    # self loop
    {'num_layers':2, 'add_self_loops':False, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu'},
    # pair norm
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu'},
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu'},
    {'num_layers':6, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu'},
    # drop edge
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.6, 'activation':'relu'},
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.6, 'activation':'relu'},
    # activation fn
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'tanh'},
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'leaky_relu'},]

def run(data_name,epochs=100):
    run_id = int(datetime.timestamp(datetime.now()))
    log_path = f"logs/edge_predict_logs/{data_name}-{run_id}"

    batch_size=64
    data,labels,train_loader,val_loader,test_loader = getDataLoader(data_name,batch_size)

    for i,config in enumerate(test_cases):
        # 训练
        best_val_acc = -1
        writer = SummaryWriter(f"{log_path}/config{i+1}")
        print(f'Test Case {i+1:>2}')
        model=GCNEdgePrediction(data.num_features,data.num_features,**config).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.6)

        for epoch in range(1,epochs+1):
             train_loss,val_loss,val_acc = train(train_loader,val_loader,model,optimizer,exp_lr_scheduler)
             writer.add_scalar('train_loss',train_loss,epoch)
             writer.add_scalar('val_loss',val_loss,epoch)
             writer.add_scalar('val_acc',val_acc,epoch)
             print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f},Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
             if val_acc > best_val_acc:
                 torch.save(model,"./output/best_model.pth")
                 best_val_acc = val_acc

        # 测试
        best_model = torch.load('./output/best_model.pth')
        test_acc = test(test_loader,best_model)
        print(f'Best Val Acc: {best_val_acc:.4f} Test Acc: {test_acc:.4f}')

def train(train_loader,val_loader,model,optimizer,scheduler):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    total_examples = total_loss = 0
    loop = tqdm(train_loader,desc="Training: ")
    for data in loop:
        if data.edge_index.size(1) == 0:
                continue
        optimizer.zero_grad()
        data=data.to(device)
        z = model.encode(data.x,data.edge_index,device)
        # 函数的实现主要目标是生成与给定图中正边不同的负边。num_neg_samples（可选参数，int）：要返回的（近似的）负样本数。如果设置为 None，将尝试为每个正边返回一个负边。默认值为 None。
        # 注意这里的边[2，bs]，所以实际上是bs通道上拼接
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,num_nodes=data.num_nodes,num_neg_samples=None,method='sparse'
        )
        neg_edge_index = neg_edge_index.to(device)
        # 就是本身cites中表示A<-B(B引用A),额外加上一些负样例，即表示A不与B连接
        edge_label_index = torch.cat([data.edge_index,neg_edge_index],dim=-1)
        # 这里是[2,bs]，其中A<-B用1标识，A不被B引用用0标识
        edge_label = torch.cat([torch.ones(data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],dim=0)
        edge_label = edge_label.to(device)
        # 其实就是预测每一个边所连接的两个顶点的feature余弦相似度
        pred = model.decode(z,edge_label_index).view(-1).sigmoid()
        # 因此前面设置1和0就有用了，融合特征后本身相连的边需要继续尽可能接近即余弦相似度相似，而不连接的就接近0
        loss = criterion(pred,edge_label)
        loss.backward()
        optimizer.step()
        total_examples += data.batch_size
        total_loss += loss.item()*data.batch_size
    scheduler.step()
    train_loss = total_loss/total_examples

    model.eval()
    with torch.no_grad():
        scores=[]
        total_loss=0
        # 只有预测的余弦相似度高于0.95的才认为边相连
        threshold = torch.tensor([0.95]).to(device)
        loop = tqdm(val_loader,desc="Validating: ")
        for data in loop:
            # 这批取得图没有已知连接边
            if data.edge_index.size(1) == 0:
                continue
            data = data.to(device)
            z = model.encode(data.x,data.edge_index,device)
            pred = model.decode(z,data.edge_index).view(-1).sigmoid()
            loss = criterion(pred,torch.ones(data.edge_index.size(1)).to(device))
            total_loss += loss.item()*data.batch_size
            pred = (pred>threshold).float()*1
            score = accuracy_score(np.ones(data.edge_index.size(1)),pred.cpu().numpy())            
            scores.append(score)
        val_loss = total_loss/total_examples
        val_acc = np.average(scores)
    return train_loss,val_loss,val_acc

            
def test(test_loader,model):
    model.eval()
    with torch.no_grad():
        labels,preds,scores=[],[],[]
        threshold = torch.tensor([0.95]).to(device)
        loop = tqdm(test_loader,desc="Testing: ")
        for data in loop:
            # 这批取得图没有已知连接边
            if data.edge_index.size(1) == 0:
                continue
            data = data.to(device)
            z = model.encode(data.x,data.edge_index,device)
            pred = model.decode(z,data.edge_index).view(-1).sigmoid()
            pred = (pred>threshold).float()*1
            score = accuracy_score(np.ones(data.edge_index.size(1)),pred.cpu().numpy())            
            scores.append(score)
        test_acc = np.average(scores)
    return test_acc

    
if __name__ == "__main__":
    run('citeseer',10)
