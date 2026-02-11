import SPACEL
from SPACEL import Scube
import scanpy as sc
import pandas as pd
import ot
import torch
import warnings
import time
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

#read h5ad slices
def load_slices(data_dir, slice_names):
    slices = []
    for slice_name in slice_names:
        slice_i = sc.read_h5ad(data_dir+slice_name+'.h5ad')
        slices.append(slice_i)
    return slices

data_dir = '/SABench/Data/'
slice_names = ["slice1", "slice2", "slice3", "slice4"]
slices_list = load_slices(data_dir,slice_names)

def SPACEL_Align(group,cluster_key):
     
    adata_list=[i.copy() for i in group]
    for i in range(len(adata_list)):
        adata_list[i].obs[cluster_key] = adata_list[i].obs[cluster_key].astype('category')
        
    Scube.align(adata_list,
      cluster_key=cluster_key, 
      n_neighbors = 4, 
      p=1,
      
    )
    for adata in adata_list:
        adata.obsm['spatial'] = np.array(adata.obsm['spatial_aligned'])    
       
    return adata_list

new_slices=SPACEL_Align(slices_list,'Region')

for i,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/SPACEL/SPACEL_new_{}.h5ad'.format(slice_names[i]))