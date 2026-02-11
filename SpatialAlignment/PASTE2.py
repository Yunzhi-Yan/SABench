import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import ot
import torch
from paste2 import PASTE2, projection

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

def pst2Align(group):
    s=0.8
    alpha = 0.1
    pis = [None for i in range(len(group)-1)] 
    
    runtime=[None for i in range(len(group)-1)] 
    memory=[None for i in range(len(group)-1)] 
    
    for i in range(len(group)-1):
        
        import tracemalloc
        import time
        tracemalloc.start()
        start_time=time.time()
           
        # PASTE2
        pis[i] = PASTE2.partial_pairwise_align(group[i], group[i+1],s=s,alpha=alpha)
        
        end_time=time.time()
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
        runtime[i]=end_time-start_time
        memory[i]=peak /1024/1024
        
        print('memory blocks peak:{:>10.4f} MB'.format(memory[i]))
        print('time: {:.4f} s'.format(runtime[i]))

    new_slices = projection.partial_stack_slices_pairwise(group, pis)
    return new_slices

new_slices=pst2Align(slices_list)

for i,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/PASTE2/PASTE2_new_{}.h5ad'.format(slice_names[i]))