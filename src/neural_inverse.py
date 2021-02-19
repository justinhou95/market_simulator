import numpy as np
import matplotlib.pyplot as plt
import torch
import signatory
import hedge

def leadlag(p):
    B = p.size()[0]
    p = torch.cat([p[:,:,None],p[:,:,None]],axis = -1)
    p = p.view([B,-1])
    p = torch.cat([p[:,:-1,None],p[:,1:,None]],axis = -1)
    return p
def leadlag_inverse(p):
    return p[0::2,0]
def generate_bmpath(batch, N):
    T = 1
    dt = T/N
    path_diff = np.random.normal(size = [batch,N,1]) * np.sqrt(dt)
    path = np.cumsum(path_diff,axis = 1)
    path = np.concatenate([np.zeros([batch,1,1]), path], axis = 1)
    return path

class Dataset(torch.utils.data.Dataset):
        def __init__(self, X):
            self.X = X
        def __len__(self):
            return len(self.X)
        def __getitem__(self, index):
            return self.X[index]
        
def data_prepare(X,y):
    batch = y.shape[0]
    if batch > 1:
        split = int(batch/2)
        X_train = X[:split]
        y_train = y[:split]

        X_test = X[split:]
        y_test = y[split:]

        ds = Dataset(torch.tensor(X_train,dtype = torch.float32))
        dl = torch.utils.data.DataLoader(ds, batch_size=64)
        return ds, dl, X_train, y_train, X_test, y_test
        
    else:
        dl = torch.tensor(X)[None,:,:]
        return dl
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dim, order, N):
        super(Net, self).__init__()
        self.N = N
        self.order = order
        self.fc1 = nn.Linear(input_dim,200)  
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,N*10)
        self.fc4 = nn.Linear(10,1)
        self.logsig1 = signatory.LogSignature(depth=order)
    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view([-1,self.N,10])
        x = self.fc4(x)
        x = torch.cumsum(x,axis = 1)
        x = torch.cat([torch.zeros(size = [B,1,1]),x], axis = 1)
        x = leadlag(x)
        sig = self.logsig1(x, basepoint = True)
        return x, sig
    
    def train_net(self, dl, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.003)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, x in enumerate(dl):
                optimizer.zero_grad()
                x = x.float()
                p, x_re = self.__call__(x)
                loss = criterion(x, x_re)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if epoch%100 == 0:    
                print('step: ',epoch,'loss: ', running_loss)
        print('Finished Training')

def reconstruct_plot(model,X,y,order = 0):
    batch = y.shape[0]
    criterion = nn.MSELoss()
    with torch.no_grad():  
        X = torch.tensor(X, dtype = torch.float)
        y_predict, X_predict = model(X)
        lo = criterion(X, X_predict)
    print('LOSS is: ', lo.numpy())
    
    if batch > 1:
        plt.figure(figsize=(16, 2))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            idx = np.random.randint(0,batch)
            y_recover = leadlag_inverse(y[idx])[0] + leadlag_inverse(y_predict[idx])
            y_true = leadlag_inverse(y[idx])
            plt.plot(y_true)
            plt.plot(y_recover)
#             K = 1
#             sigma = 1
#             time = np.linspace(0,1,y_true.shape[0])
#             C, V = hedge.delta_hedge(K, sigma, time, y_true)
#             C_recover, V_recover = hedge.delta_hedge(K, sigma, time, y_recover.numpy())
#             plt.plot(V)
#             plt.plot(V_recover)
#             if i == 3:
#                 plt.legend(['True path', 'Neural path'] + ['True hedge', 'Neural hedge'])
            
            
    else:
        logsig_recover = signatory.logsignature(y_predict, order)
#         plt.plot(leadlag_inverse(y[0]))
        y_recover = leadlag_inverse(y[0])[0] + leadlag_inverse(y_predict[0])
#         plt.plot(y_recover)
        return y_recover, logsig_recover 
    plt.show()
    
    
    