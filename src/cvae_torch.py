import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class DataSetCVAE(torch.utils.data.Dataset):
    def __init__(self, data, data_cond):
        self.data = data
        self.data_cond = data_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_cond[idx]
    
def data_pepare(data,data_cond,split_number, BATCH_SIZE):
    data = torch.tensor(data,dtype = torch.float)
    data_cond = torch.tensor(data_cond,dtype = torch.float)
    if split_number:
        data_train = data[:split_number]
        data_test = data[split_number:]
        data_cond_train = data_cond[:split_number]
        data_cond_test = data_cond[split_number:]
    else:
        data_train = data
        data_test = data
        data_cond_train = data_cond
        data_cond_test = data_cond
    train_set = DataSetCVAE(data_train,data_cond_train)
    test_set = DataSetCVAE(data_test,data_cond_test)
    train_iterator = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=BATCH_SIZE)
    return train_iterator, test_iterator


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.LeakyReLU(0.3)
    def forward(self, x):
        hidden = self.act(self.linear(x))
        mean = self.act(self.mu(hidden))
        log_var = self.act(self.var(hidden))
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, condition_dim):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.LeakyReLU(0.3)
    def forward(self, x):
        x = self.act(self.latent_to_hidden(x))
        generated_x = torch.sigmoid(self.hidden_to_out(x))
        return generated_x
    
class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, condition_dim, BETA):
        super().__init__()
        self.latent_dim = latent_dim
        self.BETA = BETA
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, condition_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, condition_dim)
    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        z_mu, z_var = self.encoder(x)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        z = torch.cat((x_sample, y), dim=1)
        generated_x = self.decoder(z)
        return generated_x, z_mu, z_var
    def calculate_loss(self, x, reconstructed_x, mean, log_var):
        RCL = torch.sum((reconstructed_x - x).pow(2))
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        LOSS = (1 - self.BETA)* RCL + self.BETA*KLD
        return LOSS
    def train_step(self):
        self.train()
        train_loss = 0
        for i, (x, y) in enumerate(self.train_iterator):
            self.optimizer.zero_grad()
            reconstructed_x, z_mu, z_var = self.__call__(x, y)
            loss = self.calculate_loss(x, reconstructed_x, z_mu, z_var)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        return train_loss
    def test_step(self):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_iterator):
                reconstructed_x, z_mu, z_var = self.__call__(x, y)
                loss = self.calculate_loss(x, reconstructed_x, z_mu, z_var)
                test_loss += loss.item()
        return test_loss
    def prepare(self, train_iterator, test_iterator):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
    def train_all(self, N_EPOCHS):
        for e in range(N_EPOCHS):
            train_loss = self.train_step()
            test_loss = self.test_step()

            train_loss /= self.train_iterator.batch_size
            test_loss /= self.test_iterator.batch_size
            print(f'Epoch {e}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')
    def generate(self,cond):
        N_generated = len(cond)
        z = torch.randn(N_generated, self.latent_dim)
        z_and_cond = torch.cat((z, cond), dim=1)
        reconstructed = self.decoder(z_and_cond).detach()
        return reconstructed


    