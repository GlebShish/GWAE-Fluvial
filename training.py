#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from data import *
import wandb
wandb.init(project="rvae", sync_tensorboard=True)

import torch.utils.data
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

from torch_geometric.data import DataLoader

import visual_tools
import classGridDataset
import useful_custom_functions
from grid import load_grid, show_grid, Graph_to_Grid 
from curve import connecting_geodesic, linear_interpolation

from smoothness import smoothness_factor_simple, smoothness_factor_hard, plot_smoothness

from GVWAE import *


# In[2]:


torch.manual_seed(187549026630100)

path_prefix = '../DATA/3_types_of_channels_with_stats/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensor_path = '../DATA/processed_dir/Channels_1500_tensor.pt'


# #### Functions for loading the dataset from Excel

# In[3]:


#dataset_list = get_dataset(path_prefix, tensor_path, device) 


# In[4]:


import pickle 
with open('dataset.pickle', 'rb') as f:
    dataset_list = pickle.load(f)


# In[5]:


poro = []
perm = []

for d in dataset_list:
    poro.append(d.x[:,0].cpu())
    perm.append(d.x[:,1].cpu())
    
poro = np.concatenate((poro), axis=0)
perm = np.concatenate((perm), axis=0)


# Normalizing perm, cause it's in log-norm

# In[6]:


transformer, perm_normalized = useful_custom_functions.from_lognorm_to_norm(perm) #


# Showing property distributions

# In[7]:


condition = (perm>0)
data_in = np.extract(condition, perm)
ccc = transformer.inverse_transform(perm_normalized)


plt.subplots(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.hist(data_in, 100)

plt.subplot(1, 3, 2)
plt.hist(perm_normalized, 100)
#plt.colorbar()

plt.subplot(1, 3, 3)
plt.hist(ccc, 100)
#plt.colorbar()

plt.show()


# In[ ]:





# In[8]:


#get min's and max's
perm_normalized = transformer.transform(perm.reshape(-1, 1))
max0 = np.amax(poro, axis=0)
max1 = np.amax(perm_normalized, axis=0).item(0)
min1 = np.amin(perm_normalized, axis=0).item(0)

#convert to tensors
max0 = torch.tensor(max0)
max1 = torch.tensor(max1)
min1 = torch.tensor(min1)
print(max0, max1, min1)
for i in range(len(dataset_list)):        
    dataset_list[i].x[:,0] /= max0
    tmp = transformer.transform(dataset_list[i].x[:,1].reshape(-1, 1).cpu()) #normalization of every graph
    dataset_list[i].x[:,1] =  torch.tensor(tmp[:,0])#-min1)/(max1-min1)


# In[9]:


#!/usr/bin/env python3
import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from enum import Enum
from torch_geometric.nn.conv import MessagePassing
from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
import nnj
from nnj import JacType
from torch import Tensor
from torch_sparse import SparseTensor, matmul


# In[11]:


mu_epochs = 30 #Epochs for mu training
sigma_epochs = 20 #Epochs for sigma training
batch_size = 1
nv=1919 #Number of vertices in the graph
nz=30 # Latent code size
writer = SummaryWriter()


# In[12]:


trainer = TrainerRBF(batch_size=batch_size, nv=nv, nz=nz, model_loss='mmd',                      gamma=0, channels=2, num_centers=1000, edge_index=dataset_list[0].edge_index, 
                     writer=writer) 


# In[13]:


train_loader = DataLoader(dataset_list, batch_size=batch_size, shuffle=True)


# In[14]:


trainer.train(mu_epochs,sigma_epochs, train_loader, path_prefix)


# In[15]:


torch.save(trainer.model, 'rvae.pt')


# In[ ]:


z1 = trainer.model.encode(dataset_list[112], 1)
z2 = trainer.model.encode(dataset_list[1389], 1)


# In[ ]:


z1.shape


# In[ ]:


x1, _ = trainer.model.decode(z1, False) 


# In[ ]:


c, _ = connecting_geodesic(trainer, z1, z2, n_nodes=32, eval_grid=64, max_iter=500, l_rate=1e-2)    
c_pts = c(torch.arange(start=0, end=1, step=0.05).to('cuda')).detach().cpu()


# In[ ]:


lin_pts = linear_interpolation(c_pts[0], c_pts[-1], n_points=20)


# In[ ]:


a = smoothness_factor_simple(c_pts, trainer)
plot_smoothness(a)

b = smoothness_factor_simple(lin_pts, trainer)
plot_smoothness(b)


# In[ ]:


#second_derivatives = smoothness_factor_hard(c_pts, trainer, num_nodes=10)
#plot_smoothness(second_derivatives)


# In[ ]:


#second_derivatives = smoothness_factor_hard(lin_pts, trainer, num_nodes=10)
#plot_smoothness(second_derivatives)


# In[ ]:


from torch.autograd import grad
def MF_measure(z, trainer):
    x, _ = trainer.model.decode(z, trainer.edge_index, 1) 
    x = x[0]
    J1 = torch.stack([torch.squeeze(grad(x[_, 1], z, retain_graph=True)[0]) for _ in range(trainer.nv)])
    J2 = torch.stack([torch.squeeze(grad(x[_, 0], z, retain_graph=True)[0]) for _ in range(trainer.nv)])
    J = torch.cat((J1,J2))
    J = torch.unsqueeze(J, axis=0)
    G = torch.einsum("bji,bjk->bik", J, J)
    det = (G/10).det().abs()
    det = det.detach_().cpu().numpy()
    measure = np.sqrt(det)
    return measure

trainer.edge_index = dataset_list[0].edge_index


# In[ ]:


MF_euclid = []
for NUM in tqdm(range(20)):    
    MF_euclid.append(MF_measure(torch.reshape(lin_pts[NUM],(1,nz,1)).clone().detach().requires_grad_(True).to('cuda'), trainer))    


# In[ ]:


MF_geodesic = []
for NUM in tqdm(range(20)):    
    MF_geodesic.append(MF_measure(torch.reshape(c_pts[NUM],(1,nz,1)).clone().detach().requires_grad_(True).to('cuda'), trainer))    


# In[ ]:


plt.plot(MF_geodesic)
plt.plot(MF_euclid)
plt.show()


# In[ ]:


lin_pts = lin_pts.detach().cpu().numpy()


# In[ ]:


for NUM in range(len(c_pts)): 
    pred_geodesic, _ = trainer.model.decode(torch.reshape(c_pts[NUM],(1,nz,1)).clone().cuda(), dataset_list[0].edge_index, 1)
    pred_euclid, _ = trainer.model.decode(torch.reshape(torch.tensor(lin_pts[NUM]),(1,nz,1)).clone().cuda(), dataset_list[0].edge_index, 1)
   
    pred_graph_geodesic = Data(x=pred_geodesic[0], label=dataset_list[0].label, edge_index=dataset_list[0].edge_index)
    pred_graph_euclid = Data(x=pred_euclid[0], label=dataset_list[0].label, edge_index=dataset_list[0].edge_index)
    
    Graph_to_Grid(path_prefix+'Facies.csv', pred_graph_geodesic, 0, 'poro', path_prefix+ 'Generated_grids\\',str(NUM)+'_Geodesic')
    Graph_to_Grid(path_prefix+'Facies.csv', pred_graph_euclid, 0, 'poro', path_prefix+'Generated_grids\\',str(NUM)+'_Euclid')


# In[ ]:


for test in range(20):
    a = load_grid(path_prefix+'Generated_grids\\'+'poro'+str(test)+'_Geodesic.GRDECL')
    z = np.ones((16,12,10))
    print(a)
    for i in range(16):
        for j in range(12):
            for k in range(10):
                z[i][j][k] = a[i*16+j*12+k]
    plt.matshow(np.mean(z,axis=2))
    plt.savefig('grid_interpolations_results\\geodesic\\grid'+str(test)+'.png')
    
    
    a = load_grid(path_prefix+'Generated_grids\\'+'poro'+str(test)+'_Euclid.GRDECL')
    z1 = np.ones((16,12,10))
    for i in range(16):
        for j in range(12):
            for k in range(10):
                z1[i][j][k] = a[i*16+j*12+k]
    plt.matshow(np.mean(z,axis=2))
    
    print(np.mean(np.abs(z-z1)))
    plt.savefig('grid_interpolations_results\\euclid\\grid'+str(test)+'.png')


# In[ ]:




