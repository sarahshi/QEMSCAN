# %% 

""" Created on April 3, 2023 // @author: Sarah Shi, Norbert Toth, Po-Yen Tung """

import torch
import time
from torch import nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class FeatureDataset(Dataset):
    def __init__(self, x):
        if len(x.shape)==2:
            self.x = x
        else:
            self.x = x.reshape(-1, x.shape[-1]) #dataset keeps the right shape for training

    def __len__(self):
        return self.x.shape[0] 
    
    def __getitem__(self, n): 
        return torch.Tensor(self.x[n])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Autoencoder(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encode = nn.Sequential(nn.Linear(self.input_dim,512),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(512,256),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(256,128),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(128,self.latent_dim)
                                    )

        self.decode = nn.Sequential(nn.Linear(self.latent_dim,128),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(128,256),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(256,512),
                                     nn.LeakyReLU(0.02),
                                     
                                     nn.Linear(512,self.input_dim)
                                    )
        

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de



class Autoencoder_elemental(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2, hidden_layer_sizes=(512, 256, 128)): # test shallow autoencoder w layers of 64, 32
        super(Autoencoder_elemental, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hls = hidden_layer_sizes

        def element(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.LeakyReLU(0.02),
            ]

        encoder = element(self.input_dim, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += element(self.hls[i], self.hls[i + 1])
        encoder += [nn.Linear(self.hls[-1], latent_dim)]

        decoder = element(latent_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += element(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.input_dim)] 

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de




class Tanh_Autoencoder(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2):
        super(Tanh_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encode = nn.Sequential(nn.Linear(self.input_dim,64),
                                     nn.Tanh(),
                                     nn.Linear(64,32),
                                     nn.Tanh(),
                                     nn.Linear(32,self.latent_dim)
                                    )

        self.decode = nn.Sequential(nn.Linear(self.latent_dim,32),
                                     nn.Tanh(),
                                     nn.Linear(32,64),
                                     nn.Tanh(),
                                     nn.Linear(64,self.input_dim)
                                    )

    def encoded(self, x):
        #encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de



class Tanh_Autoencoder_elemental(nn.Module):
    def __init__(self,input_dim = 7, latent_dim = 2, hidden_layer_sizes=(64, 32)):
        super(Tanh_Autoencoder_elemental, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hls = hidden_layer_sizes

        def element(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.Tanh(),
            ]

        encoder = element(self.input_dim, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += element(self.hls[i], self.hls[i + 1])
        encoder += [nn.Linear(self.hls[-1], latent_dim)]

        decoder = element(latent_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += element(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.input_dim)] 

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de



def train(model, optimizer, train_loader, test_loader, n_epoch, criterion):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    avg_train_loss = []
    avg_test_loss = []

    for epoch in range(n_epoch):
        # Training
        model.train()
        t = time.time()
        train_loss = []
        for i, data in enumerate(train_loader):
            x = data.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().item())
        
        # Testing
        model.eval()
        test_loss = []
        for i, test in enumerate(test_loader):
            x = test.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)
            test_loss.append(loss.detach().item())
        
        # Logging
        avg_loss = sum(train_loss) / len(train_loss)
        avg_test = sum(test_loss) / len(test_loss)
        avg_train_loss.append(avg_loss)
        avg_test_loss.append(avg_test)
        
        training_time = time.time() - t
        
        print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_loss:.6f}, test_loss: {avg_test:.6f}, time: {training_time:.2f} s')

    return avg_train_loss, avg_test_loss



def save_model(model, optimizer, path):
    check_point = {'params': model.state_dict(),                            
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)

def load_model(model, optimizer=None, path=''):
    check_point = torch.load(path)
    model.load_state_dict(check_point['params'])
    if optimizer is not None:
        optimizer.load_state_dict(check_point['potimizer'])

def getLatent(model, dataset:np):
    #transform real data to latent space using the trained model
    latents=[]
    model.to(device)

    dataset_ = FeatureDataset(dataset)
    loader = DataLoader(dataset_,batch_size=20,shuffle=False)
    
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(loader):
            x = data.to(device)
            z = model.encoded(x)
            latents.append(z.detach().cpu().numpy())
    
    return np.concatenate(latents, axis=0)
