import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import ot
import torch
import time
import numpy as np
import plotly
import requests
from STalign import STalign

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

def STalign_Align(group,device):    
   
    runtime=[None for i in range(len(group)-1)] 
    memory=[None for i in range(len(group)-1)] 
    
    new_slices=group.copy()
    # set device for building tensors
    if device=='cuda':
        torch.set_default_device('cuda:0')
    else:
        torch.set_default_device('cpu')
        
    for i in range(len(new_slices)-1):
    
        import tracemalloc
        import time
        tracemalloc.start()
        start_time=time.time()
    
        #
        adata1 = new_slices[i]
        adata2 = new_slices[i+1]
        
        xI=np.array(adata2.obsm['spatial'][:,0], dtype=np.float64)
        yI=np.array(adata2.obsm['spatial'][:,1], dtype=np.float64)
        xJ=np.array(adata1.obsm['spatial'][:,0], dtype=np.float64)
        yJ=np.array(adata1.obsm['spatial'][:,1], dtype=np.float64)
        # rasterize at 30um resolution (assuming positions are in um units) and plot
        XI,YI,I,figI = STalign.rasterize(xI,yI)
        # plot
        ax = figI.axes[0]
        ax.invert_yaxis()
        # rasterize and plot
        XJ,YJ,J,figJ = STalign.rasterize(xJ,yJ)
        ax = figJ.axes[0]
        ax.invert_yaxis()
        
        # run LDDMM
        # specify device (default device for STalign.LDDMM is cpu)
        # keep all other parameters default
        params = {
            'niter': 10000,
            'device':device,
            'epV': 50
          }
        out = STalign.LDDMM([YI,XI],I,[YJ,XJ],J,**params)

        # get necessary output variables
        A = out['A']
        v = out['v']
        xv = out['xv']
    
        # apply transform to original points
        tpointsI= STalign.transform_points_source_to_target(xv,v,A, np.stack([yI, xI], 1))

        #switch tensor from cuda to cpu for plotting with numpy
        if tpointsI.is_cuda:
            tpointsI = tpointsI.cpu()

        #switch from row column coordinates (y,x) to (x,y)
        xI_LDDMM = tpointsI[:,1]
        yI_LDDMM = tpointsI[:,0]
        
        adata2.obsm['spatial'][:,0]=xI_LDDMM
        adata2.obsm['spatial'][:,1]=yI_LDDMM      
     
        end_time=time.time()
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
        runtime[i]=end_time-start_time
        memory[i]=peak /1024/1024
        
        print('memory blocks peak:{:>10.4f} MB'.format(memory[i]))
        print('time: {:.4f} s'.format(runtime[i]))
    
    return new_slices

new_slices=STalign_Align(slices_list,'cuda')
for i,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/STalign/STalign_new_{}.h5ad'.format(slice_names[i]))