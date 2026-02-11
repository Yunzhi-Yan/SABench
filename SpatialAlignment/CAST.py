import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import ot
import torch
import time
import CAST
import os
import numpy as np
from os.path import join as pj
import warnings
warnings.filterwarnings("ignore")
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

def CAST_Align(group,device):

    work_dir = '/SABench/SpatialAlignment/CAST/' # input the demo path
    # Output directory for the results
    output_path = f'{work_dir}'
    os.makedirs(output_path,exist_ok=True)
   
    if device=='cuda':
        Device=0
    else:
        Device=-1
     
    runtime=[None for i in range(len(group)-1)] 
    memory=[None for i in range(len(group)-1)] 

    new_slices=group.copy()
    
    for i in range(len(new_slices)-1):
        
        import tracemalloc
        import time
        tracemalloc.start()
        start_time=time.time()

        #
        adata1 = new_slices[i+1]
        adata2 = new_slices[i]
        adata1.obs['x']=adata1.obsm['spatial'][:,0]
        adata1.obs['y']=adata1.obsm['spatial'][:,1]
        adata2.obs['x']=adata2.obsm['spatial'][:,0]
        adata2.obs['y']=adata2.obsm['spatial'][:,1]
        
        # combine the two datasets
        sample_list= ['adata1','adata2'] # [Query, Reference]
        sdata = adata1.concatenate(adata2)
        # rename the dataset labels 
        batch_key = 'batch'
        batch_rename = {'0' : sample_list[0],'1' : sample_list[1]}
        sdata.obs.replace({batch_key:batch_rename},inplace=True)
        
        ## Extract and subset the coordinate and expression data for each sample
        from CAST.utils import extract_coords_exp
        sdata.layers['norm1e4'] = sc.pp.normalize_total(sdata, target_sum=1e4, inplace=False)['X'].toarray() # we use normalized counts for each cell as input gene expression

        coords_raw,exps = extract_coords_exp(sdata, batch_key = 'batch', cols = ['x', 'y'], count_layer = '.X', data_format = 'norm1e4')
        
        ## Run CAST Mark — capture common spatial features
        from CAST.models.model_GCNII import Args
        from CAST import CAST_MARK
        from CAST.visualize import kmeans_plot_multiple 
   
        # run CAST Mark
        if device=='cuda':
            embed_dict = CAST_MARK(coords_raw,exps,output_path,graph_strategy='delaunay')
        else:
            embed_dict = CAST_MARK(coords_raw,exps,output_path,gpu_t = -1,graph_strategy='delaunay')
            
        # plot the results
        kmeans_plot_multiple(embed_dict,sample_list,coords_raw,'demo1',output_path,k=20,dot_size = 10,minibatch=True)

        ## Run CAST Stack — align the two samples
        from CAST.CAST_Stack import reg_params
        from CAST import CAST_STACK
        # set the parameters for CAST Stack
        query_sample = sample_list[0]
        params_dist = reg_params(dataname = query_sample,
                            gpu = Device,
                            #### Affine parameters
                            iterations=150,
                            dist_penalty1=0,
                            bleeding=500,
                            d_list = [3,2,1,1/2,1/3],
                            attention_params = [None,3,1,0],
                            #### FFD parameters
                            dist_penalty2 = [0],
                            alpha_basis_bs = [0],
                            meshsize = [8],
                            iterations_bs = [1],
                            attention_params_bs = [[None,3,1,0]],
                            mesh_weight = [None])
                     
        # set the alpha basis for the affine transformation
        params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
        # run CAST Stack
        coord_final = CAST_STACK(coords_raw,embed_dict,output_path,sample_list,params_dist,sub_node_idxs = None)
       
        adata1.obsm['spatial']=coord_final[sample_list[0]].numpy()
        adata2.obsm['spatial']=coord_final[sample_list[1]]
      
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
new_slices=CAST_Align(slices_list,'cuda')
#cpu
#new_slices=CAST_Align(slices_list,'cpu')

for i,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/CAST/CAST_new_{}.h5ad'.format(slice_names[i]))