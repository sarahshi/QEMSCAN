# %% 

""" Created on December 16, 2021 // @author: Sarah Shi, Norbert Toth, Po-Yen Tung """


import os 
import sys
import math
import time
import random

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

import skimage 
import skimage as ski
from sklearn import decomposition
from sklearn.mixture import BayesianGaussianMixture
from scipy.special import softmax
from skimage import segmentation, morphology, measure, transform, io
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max

#own code:
import QEMSCAN_functions as qf
import PyTorch_AE as pt
from PyTorch_AE import Autoencoder, Shallow_Autoencoder, Tanh_Autoencoder

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

import mc3
import scipy.interpolate as interpolate


# %% 

path_parent = os.path.dirname(os.getcwd())
path_grandparent = os.path.dirname(path_parent)

output_dir = ["/OutputData/030122Run_NoNa"] 

for ii in range(len(output_dir)):
    if not os.path.exists(output_dir[ii]):
       os.makedirs(path_grandparent + output_dir[ii], exist_ok=True)

DF_NEW = pd.read_csv('./CleanedProfilesDF.csv', index_col= ['Comment', 'DataSet/Point'],)
samplename = DF_NEW.index.levels[0]

# %% 

def build_conc_map(data, shape=None,):
    """
    Build 3D numpy conc_map from pandas dataframe.
    
    Input
    -----------
    data (pd dataframe)     
        pandas dataframe to be transformed to numpy data matrix.
    shape (list) (optional) 
        desired shape of the resulting array; if not given
        one will be generated using the data and shape of dataframe.
                        
    Return
    -----------
    conc_map (3D numpy array) 
        the resulting data matrix of shape either given or
        calculated.
    data_mask (2D numpy array) 
        mask showing where data exists in conc_map (binary mask)
    """

    if isinstance(data, pd.DataFrame):
        pass
    else:
        raise ValueError("Data is not pandas dataframe.")
    
    if shape == None:
        x_max = data['X'].max() + 1
        y_max = data['Y'].max() + 1
        n_features = len(data.columns) - 2 #need to substract x,y
        shape = [x_max, y_max, n_features]
    else:
        pass
    
    conc_map = np.zeros(shape)
    conc_map.fill(np.nan)
    data_mask = np.zeros((shape[0], shape[1]))
    k = 1
    length = len(data)
    
    print("Starting to build data-cube.")
    for i in range(length):
        x = data.loc[i, 'X']
        y = data.loc[i, 'Y']
        conc_map[x,y] = data.iloc[i,2:].to_numpy()
        data_mask[x,y] += 1

        if int((i/length)*10) == k:
            print("Progress: " + str(k*10) + "%")
            k += 1
        else:
            pass
    print("Progress: " + str(100)+ "%")

    ## save from CSV to npz to use this data cube 
    return conc_map, data_mask



def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def autoencode(name):
    conc_data = np.load(path_grandparent + "/InputData/NPZ/" + name + ".npz", allow_pickle = True)
    conc_map = conc_data["conc_map"]
    mask = conc_data["data_mask"]
    elements = conc_data["elements"]
    conc_map = np.delete(conc_map, np.s_[0], axis = 2)
    elements = np.delete(elements, np.s_[0], axis=0)
    #perform z-score normalisation
    array_norm, factors = qf.feature_normalisation(conc_map[mask.astype("bool")], return_params = True)
    array_norm = softmax(array_norm, axis = 1)
    #split the dataset into train and test sets
    train_data, test_data = train_test_split(array_norm, test_size=0.1, random_state=42)

    #define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = pt.FeatureDataset(train_data)
    test_dataset = pt.FeatureDataset(test_data)   

    #autoencoder params:
    lr = 1e-3
    wd = 0
    batch_size = 128
    #use half the data available
    epochs = 50
    input_size = feature_dataset.__getitem__(0).size(0)

    #define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    #define model
    model = Tanh_Autoencoder(input_dim=input_size).to(device)

    #use ADAM optimizer with mean squared error loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd) 
    criterion = nn.MSELoss()

    #train model using pre-defined function
    train_loss, test_loss = pt.train(model, optimizer, feature_loader, test_loader, epochs, criterion)
    np.savez(path_grandparent + output_dir[0] + "/" + name + '_tanh_loss.npz', train_loss = train_loss, test_loss = test_loss)

    #transform entire dataset to latent space
    z = pt.getLatent(model, array_norm)

    #plot latent representation
    fig1, ax1 = plt.subplots(1,1,figsize = (12,12))
    ax1.scatter(z[:,0], z[:,1], s = 0.0001)
    ax1.set_xlabel("Latent Variable 1")
    ax1.set_ylabel("Latent Variable 2")
    ax1.set_title(name + " Tanh Latent Space Representation")
    fig1.savefig(path_grandparent + output_dir[0] + "/" + name + "_tanh.png", dpi = 300)

    #plot train and test losses over time
    # fig2, ax2 = plt.subplots(1,1, figsize = (12,12))
    # ax2.plot(np.linspace(1,epochs,epochs), train_loss, 'k', label = "train")
    # ax2.plot(np.linspace(1,epochs,epochs), test_loss, 'r', label = "test")
    # ax2.set_title(name + " Losses")
    # ax2.legend(loc = "upper right")
    # fig2.savefig(path_grandparent + output_dir[0] + name + "_tanh_loss.png", dpi = 300)

    #save main model params
    model_path = path_grandparent + output_dir[0] + "/" + name + "_tanh_params.pt"
    pt.save_model(model, optimizer, model_path)

    #save all other params
    conc_file = name + "_tanh.npz"
    np.savez(path_grandparent + output_dir[0] + "/" + name + "_tanh.npz", batch_size = batch_size, epochs = epochs, input_size = input_size, 
                elements = elements, conc_file = conc_file, factors = factors, z = z)

    # labs, centers = qf.cluster(z, n_clusters = 12, data_mask = mask, method = 'b-gmm')
    # np.savez(path_grandparent + output_dir[0] + "/" + name + '_tanh_labscenters.npz', labs = labs, centers = centers)

#start execute here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
same_seeds(42)

names = ["K7_4mu_allphases", "K8_4mu_allphases", "T7_4mu_allphases", "T8_4mu_allphases"]

i = int(sys.argv[1]) - 1
start_time = time.time()
print("starting " + str(names[i]))
autoencode(names[i])
print(names[i] + " done! Time: " + str(time.time() - start_time) + "s")

# %%
