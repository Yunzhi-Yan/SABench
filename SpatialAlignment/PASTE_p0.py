import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import ot
import torch
import paste as pst
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

def pstp0Align(group,Use_gpu=False,Backend=ot.backend.NumpyBackend()):
    alpha = 0.1
    pis = [None for i in range(len(group)-1)] 
    runtime=[None for i in range(len(group)-1)] 
    memory=[None for i in range(len(group)-1)] 

    for i in range(len(group)-1):
        import tracemalloc
        import time
        tracemalloc.start()
        start_time=time.time()
        
        #PASTE_p0
        pi0 = pst.match_spots_using_spatial_heuristic(group[i].obsm['spatial'],group[i+1].obsm['spatial'],use_ot=True)
        pis[i] = pst.pairwise_align(group[i], group[i+1],alpha=alpha,G_init=pi0,norm=True,verbose=False,
                                    use_gpu = Use_gpu,backend =Backend)        
              
        #
        end_time=time.time()
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
        runtime[i]=end_time-start_time
        memory[i]=peak /1024/1024
        
        print('memory blocks peak:{:>10.4f} MB'.format(memory[i]))
        print('time: {:.4f} s'.format(runtime[i]))
             
    new_slices = pst.stack_slices_pairwise(group, pis)
    
    return new_slices

#gpu
new_slices=pstp0Align(slices_list,Use_gpu=True, Backend = ot.backend.TorchBackend())
#cpu
#new_slices=pstp0Align(slices_list,Use_gpu=False)

for i,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/PASTE_p0/PASTE_p0_new_{}.h5ad'.format(slice_names[i]))