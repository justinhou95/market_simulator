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
        ds = Dataset(torch.tensor(X,dtype = torch.float32))
        dl = torch.utils.data.DataLoader(ds, batch_size=100)
        return ds, dl
#         
#         split = int(batch/2)
#         X_train = X[:split]
#         y_train = y[:split]

#         X_test = X[split:]
#         y_test = y[split:]

#         ds = Dataset(torch.tensor(X_train,dtype = torch.float32))
#         dl = torch.utils.data.DataLoader(ds, batch_size=64)
#         return ds, dl, X_train, y_train, X_test, y_test
        
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
        
        
class Net_two(nn.Module):
    def __init__(self, input_dim, order, N):
        super(Net_two, self).__init__()
        self.N = N
        self.order = order
        self.fc1 = nn.Linear(input_dim,200)  
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,N*10)
        self.fc4 = nn.Linear(10,2)
        self.logsig1 = signatory.LogSignature(depth=order)
    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view([-1,self.N,10])
        x = self.fc4(x)
        x = torch.cumsum(x,axis = 1)
        x = torch.cat([torch.zeros(size = [B,1,2]),x], axis = 1)
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
        
        
class Net_time(nn.Module):
    def __init__(self, input_dim, order, N, d):
        super(Net_time, self).__init__()
        self.level = np.array([len(w) for w in signatory.lyndon_words(d+1,order)])
        self.d = d
        self.N = N
        self.order = order
        self.fc1 = nn.Linear(input_dim,200)  
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,N*10)
        self.fc4 = nn.Linear(10,self.d)
        self.logsig1 = signatory.LogSignature(depth=order)
    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view([-1,self.N,10])
        x = self.fc4(x)
        x = torch.cumsum(x,axis = 1)
        x = torch.cat([torch.zeros(size = [B,1,self.d]),x], axis = 1)
        
        time = torch.linspace(0,1,self.N+1)
        time_torch = time.repeat([B,1])[:,:,None]
        
        x = torch.cat([time_torch,x], axis = -1)
        sig = self.logsig1(x, basepoint = True)
        return x, sig
    
    def train_net(self, dl, epochs):
#         print(self.level)
#         weight = 1.1**(self.level-1)
#         weight = torch.Tensor(weight)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.003)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, x in enumerate(dl):
                optimizer.zero_grad()
                x = x.float()
#                 print(x.shape)
                p, x_re = self.__call__(x)
                loss = criterion(x, x_re)
#                 print(x_re.shape)
#                 loss = torch.dot(torch.square(x - x_re)[0,:] , weight )
#                 print(loss.shape)
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
        y_recover =  [(leadlag_inverse(y_id)[0] + leadlag_inverse(y_predict_id)).numpy() for y_id,y_predict_id in zip(y,y_predict)]
        logsig_recover = signatory.logsignature(y_predict, order)
        
        plt.figure(figsize=(16, 2))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            idx = np.random.randint(0,batch)
            y_true = leadlag_inverse(y[idx])
            plt.plot(y_true)
            plt.plot(y_recover[idx])
        plt.show()
        return y_recover, logsig_recover
    else:
        logsig_recover = signatory.logsignature(y_predict, order)
        y_recover =  leadlag_inverse(y_predict[0]) #+ leadlag_inverse(y[0])[0]
#         plt.plot(leadlag_inverse(y[0]))
#         plt.plot(y_recover)
        return y_recover, logsig_recover 


def inverse_multiple_path_time(logsig, N, order, d):
    X0 = logsig.numpy()
    ds = Dataset(torch.tensor(X0,dtype = torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=64)
    net0 = Net_time(X0.shape[-1],order, N, d)
    net0.train_net(dl,1000)
    with torch.no_grad():  
        X = torch.tensor(X0, dtype = torch.float)
        y_predict, X_predict = net0(X)    
    return net0, y_predict, X_predict


def inverse_single_path_time(logsig, N, order, time, d, net0 = None):
    X0 = logsig.numpy()
    dl = torch.tensor(X0)[None,:,:]
    if not net0:
        net0 = Net_time(X0.shape[-1],order,N, time, d)
    net0.train_net(dl,1000)
    criterion = nn.MSELoss()
    with torch.no_grad():  
        X = torch.tensor(X0, dtype = torch.float)
        y_predict, X_predict = net0(X)    
    return net0, y_predict, X_predict

def inverse_single_path_two(logsig, N, order, net0 = None):
    X0 = logsig.numpy()
    dl = torch.tensor(X0)[None,:,:]
    if not net0:
        net0 = Net_two(X0.shape[-1],order,N)
    net0.train_net(dl,1000)
    criterion = nn.MSELoss()
    with torch.no_grad():  
        X = torch.tensor(X0, dtype = torch.float)
        y_predict, X_predict = net0(X)    
    return net0, y_predict, X_predict
    
    
def inverse_single_path(path0, order, sig = None, net0 = None, N = 28):
    if not sig:
        N = path0.shape[1]-1
        path_torch = leadlag(torch.tensor(path0)[:,:,0])
        path_leadlag = path_torch.numpy()
        logsig = signatory.logsignature(path_torch, order)
        X0 = logsig.numpy()
        y0 = path_leadlag
    else:
        X0 = path0
        y0 = path0[:,:,None]
    dl = data_prepare(X0,y0)
    if not net0:
        net0 = Net(X0.shape[-1],order,N)
    net0.train_net(dl,1000)
    y_recover, logsig_recover = reconstruct_plot(net0, X0, y0, order)
    return net0, y_recover, logsig_recover
    
def inverse_multiple_path(path, order, net = None, train = True):
    N = path.shape[1]-1
    path_torch = leadlag(torch.tensor(path)[:,:,0])
    path_leadlag = path_torch.numpy()
    logsig = signatory.logsignature(path_torch, order)
    X = logsig.numpy()
    y = path_leadlag
    if not net:
        ds, dl = data_prepare(X,y)
        net = Net(X.shape[-1],order,N)
        net.train_net(dl,1000)
    elif train:
        ds, dl = data_prepare(X,y)
        net.train_net(dl,1000)
     
    y_recover, logsig_recover = reconstruct_plot(net, X, y, order)

    return y_recover, logsig_recover, net
    