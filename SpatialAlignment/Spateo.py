import os
import torch
import spateo as st
print("Last run with spateo version:", st.__version__)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import pandas as pd
import ot
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

def Spateo_Align(group,whether_rigid,device):
   
    runtime=[None for i in range(len(group)-1)] 
    memory=[None for i in range(len(group)-1)] 

    new_slices=[data.copy() for data in group]
    for i in range(len(group)-1):
        import tracemalloc
        import time
        tracemalloc.start()
        start_time=time.time()
        
        slice1 = new_slices[i]
        slice2 = new_slices[i+1]

        #sc.pp.filter_cells(slice1, min_genes=10)  # 
        #sc.pp.filter_genes(slice1, min_cells=3)
        slice1.layers["counts"] = slice1.X.copy()
        sc.pp.normalize_total(slice1)
        sc.pp.log1p(slice1)
        sc.pp.highly_variable_genes(slice1, n_top_genes=2000)

        #sc.pp.filter_cells(slice2, min_genes=10)
        #sc.pp.filter_genes(slice2, min_cells=3)
        slice2.layers["counts"] = slice2.X.copy()
        sc.pp.normalize_total(slice2)
        sc.pp.log1p(slice2)
        sc.pp.highly_variable_genes(slice2, n_top_genes=2000)        

        #st.align.group_pca([slice1,slice2], pca_key='X_pca')
        
        spatial_key = 'spatial'
        key_added = 'align_spatial'
        # spateo return aligned slices as well as the mapping matrix
        aligned_slices, pis = st.align.morpho_align(
            models=[slice1, slice2],
            ## Uncomment this if use highly variable genes
            # models=[slice1[:, slice1.var.highly_variable], slice2[:, slice2.var.highly_variable]],
    
            ## Uncomment the following if use pca embeddings
            # rep_layer='X_pca',
            # rep_field='obsm',
            # dissimilarity='cos',
    
            verbose=False,
            spatial_key=spatial_key,
            key_added=key_added,
            device=device,
    
            ##nonrigid related parameters
            #beta=1,
            #lambdaVF=1,
            #max_iter=500,
            #K=100,
    
            ##partial alignment
            #partial_robust_level=50

            ##Improve Efficiency and Scalibity
            #SVI_mode=True,
            #n_sampling=10000,
            #sparse_calculation_mode=True,
            #use_chunk=True,
            #chunk_capacity=4,
        )

        new_slices[i]=aligned_slices[0]
        new_slices[i+1]=aligned_slices[1]
        
        new_slices[i].obsm['spatial'] = new_slices[i].obsm['align_spatial'+ whether_rigid]  
        for key in list(new_slices[i].uns.keys()):
            del new_slices[i].uns[key]
        new_slices[i+1].obsm['spatial'] = new_slices[i+1].obsm['align_spatial'+ whether_rigid]  
        for key in list(new_slices[i+1].uns.keys()):
            del new_slices[i+1].uns[key]
 
        #
        end_time=time.time()
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
        runtime[i]=end_time-start_time
        memory[i]=peak /1024/1024
        
        print('memory blocks peak:{:>10.4f} MB'.format(memory[i]))
        print('time: {:.4f} s'.format(runtime[i]))
                 
    return new_slices

new_slices_rigid=Spateo_Align(slices_list,'_rigid','cuda')
new_slices_nonrigid=Spateo_Align(slices_list,'_nonrigid','cuda')

for i, data in enumerate(new_slices_nonrigid):  
    data.write('/SABench/AlignmentResults/Spateo_nonrigid/Spateo_nonrigid_new_{}.h5ad'.format(slice_names[i]))

for i, data in enumerate(new_slices_rigid):  
    data.write('/SABench/AlignmentResults/Spateo_rigid/Spateo_rigid_new_{}.h5ad'.format(slice_names[i]))