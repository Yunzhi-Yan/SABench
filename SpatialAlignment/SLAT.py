import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import ot
import torch
import time
import numpy as np
import scSLAT
from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
from scSLAT.viz import match_3D_multi, hist, Sankey
from scSLAT.metrics import region_statistics
torch.cuda.is_available()

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

def find_rigid_transform(A, B):###from SANTO
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def SLAT_Align(group):

    runtime=[None for i in range(len(group)-1)] 
    memory=[None for i in range(len(group)-1)] 

    new_slices=group.copy()
    
    for i in range(len(new_slices)-1):
     
        import tracemalloc
        import time
        tracemalloc.start()
        start_time=time.time()
    
        #
        adata1 = new_slices[i]
        adata2 = new_slices[i+1]
        Cal_Spatial_Net(adata1, k_cutoff=10, model='KNN')
        Cal_Spatial_Net(adata2, k_cutoff=10, model='KNN')
        edges, features = load_anndatas([adata1, adata2], feature='DPCA',check_order=False)
        embd0, embd1, slat_time = run_SLAT(features, edges)
        best, index, distance = spatial_match(features, adatas=[adata1,adata2], reorder=False)
        matching = np.array([range(index.shape[0]), best])

        R,T = find_rigid_transform(adata2.obsm['spatial'],adata1.obsm['spatial'][matching[1,:]])
        adata2.obsm['spatial']=np.dot(adata2.obsm['spatial'],R.T) + T
  
        end_time=time.time()
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
        runtime[i]=end_time-start_time
        memory[i]=peak /1024/1024
        
        print('memory blocks peak:{:>10.4f} MB'.format(memory[i]))
        print('time: {:.4f} s'.format(runtime[i]))
           
    return new_slices

#gpu
new_slices=SLAT_Align(slices_list)

for i,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/SLAT/SLAT_new_{}.h5ad'.format(slice_names[i]))