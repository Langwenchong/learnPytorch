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
    log_path = f"logs/node_classify_logs/{data_name}-{run_id}"

    batch_size=64
    data,labels,train_loader,val_loader,test_loader = getDataLoader(data_name,batch_size)

    for i,config in enumerate(test_cases):
        # 训练
        best_val_acc = -1
        writer = SummaryWriter(f"{log_path}/config{i+1}")
        print(f'Test Case {i+1:>2}')
        model=GCNNodeClassifier(data.num_features,len(labels),**config).to(device)
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
    criterion = F.cross_entropy
    model.train()
    total_examples = total_loss = 0
    loop = tqdm(train_loader,desc="Training: ")
    labels,preds=[],[]
    for data in loop:
        optimizer.zero_grad()
        labels.append(data.y)
        data=data.to(device)
        pred = model(data.x,data.edge_index,device)
        loss = criterion(pred,data.y)
        preds.append((pred>0).float().cpu())
        loss.backward()
        optimizer.step()
        # 这里计算损失，应该还是使用batch_size合理，这里要×bs，是因为默认mean策略了
        total_loss += loss.item()*data.batch_size
        total_examples += data.batch_size
    scheduler.step()
    train_loss = total_loss/total_examples

    model.eval()
    with torch.no_grad():
        labels,preds=[],[]
        total_loss=0
        loop = tqdm(val_loader,desc="Validating: ")
        for data in loop:
            labels.append(data.y)
            data=data.to(device)
            pred = model(data.x,data.edge_index,device)
            preds.append((pred>0).float().cpu())
            loss = criterion(pred,data.y)
            total_loss += loss.item()*data.batch_size
        labels,preds = torch.cat(labels,dim=0).numpy(),torch.argmax(torch.cat(preds,dim=0),dim=-1).numpy()
        val_loss = total_loss/total_examples
        val_acc = accuracy_score(labels,preds)
    return train_loss,val_loss,val_acc
    
        
def test(test_loader,model):
    model.eval()
    with torch.no_grad():
        labels,preds=[],[]
        loop = tqdm(test_loader,desc="Testing: ")
        for data in loop:
            labels.append(data.y)
            data=data.to(device)
            pred = model(data.x,data.edge_index,device)
            preds.append((pred>0).float().cpu())
        labels,preds = torch.cat(labels,dim=0).numpy(),torch.argmax(torch.cat(preds,dim=0),dim=-1).numpy()
        # 只需要计算准确率
        test_acc = accuracy_score(labels,preds)
    return test_acc


if __name__ == "__main__":
    run('cora',50)


            



