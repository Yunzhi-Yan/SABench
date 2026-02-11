import sys
import os
#
sys.path.append(os.path.abspath('/SABench/packages/SANTO'))
import easydict
from santo.utils import santo, simulate_stitching, evaluation
from santo.data import intersect
import numpy as np
from tqdm import tqdm
import scanpy as sc
import math
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
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

def SANTO_Align(group,device):
   
    runtime=[None for i in range(len(group)-1)] 
    memory=[None for i in range(len(group)-1)] 
    
    new_slices=[data.copy() for data in group]
    for i in range(len(group)-1):
        import tracemalloc
        import time
        tracemalloc.start()
        start_time=time.time()
                    
        slice1 = new_slices[i+1]
        slice2 = new_slices[i]

        #slice1.X = np.array(slice1.X.todense())
        #slice2.X = np.array(slice2.X.todense())
        
        #
        args = easydict.EasyDict({})
        args.epochs = 20
        args.lr = 0.001
        args.k = 10
        args.alpha = 0.9 # weight of transcriptional loss
        args.dimension = 2  # choose the dimension of coordinates (2 or 3)
        args.diff_omics = False # whether to use different omics data
        args.mode = 'stitch' # Choose the mode among 'align', 'stitch' and None
        args.device = device # choose the device

        align_source_cor, trans_dict = santo(slice1, slice2, args)

        slice1.obsm['spatial'] = align_source_cor

        #
        end_time=time.time()
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
        runtime[i]=end_time-start_time
        memory[i]=peak /1024/1024
        
        print('memory blocks peak:{:>10.4f} MB'.format(memory[i]))
        print('time: {:.4f} s'.format(runtime[i]))
             
     return new_slices

#gpu
new_slices=SANTO_Align(slices_list,'cuda:0')
#cpu
#new_slices=SANTO_Align(slices_list,'cpu')
for i,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/SANTO/SANTO_new_{}.h5ad'.format(slice_names[i]))