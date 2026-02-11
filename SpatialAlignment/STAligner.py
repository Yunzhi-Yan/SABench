import STAligner
# the location of R (used for the mclust clustering)
import os
os.environ['R_HOME'] = "/anaconda3/envs/STAligner/lib/R"
os.environ['R_USER'] = "/anaconda3/envs/STAligner/lib/python3.8/site-packages/rpy2"
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix
import torch
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

def PrepocessingData(group,slice_names):
    Batch_list = []
    adj_list = []
    for i in range(len(slice_names)):
        
        print(slice_names[i])
        adata = group[i]
        adata.X = csr_matrix(adata.X)
        adata.var_names_make_unique(join="++")

        # make spot name unique
        adata.obs_names = [x+'_'+slice_names[i] for x in adata.obs_names]    
    
        # Constructing the spatial network
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
        # ST_utils.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
       
        # Normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        #adata = adata[:, adata.var['highly_variable']] #
        adj_list.append(adata.uns['adj'])
        Batch_list.append(adata)
    return Batch_list

def STAlignerAlign(Batch_list,slice_names,used_device):
            
    #STAligner
    adata_concat = ad.concat(Batch_list, label="slice_name", keys=slice_names)
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
    print('adata_concat.shape: ', adata_concat.shape)

    # iter_comb is used to specify the order of integration. For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    iter_comb = [(1, 0), (2, 1), (3, 2)]

    # Here, to reduce GPU memory usage, each slice is considered as a subgraph for training.
    adata_concat = STAligner.train_STAligner_subgraph(adata_concat, verbose=True, knn_neigh = 100, n_epochs = 600, iter_comb = iter_comb, 
                                                    Batch_list=Batch_list, device=used_device)

    sc.pp.neighbors(adata_concat, use_rep='STAligner', random_state=666)
    sc.tl.louvain(adata_concat, random_state=666, key_added="louvain", resolution=0.2)

    for it in range(len(slice_names)):
        Batch_list[it].obs['louvain'] = adata_concat[adata_concat.obs['batch_name'] == slice_names[it]].obs['louvain'].values

    #ICP
    landmark_domains = adata_concat.obs['louvain'].unique().categories.tolist()
    landmark_domains = [[element] for element in landmark_domains]#
    for landmark_domain in landmark_domains:
        Batch_list_copy=Batch_list.copy()
        for comb in iter_comb:
            print(comb)
            i, j = comb[0], comb[1]
            adata_target = Batch_list_copy[i]
            adata_ref = Batch_list_copy[j]
            slice_target = slice_names[i]
            slice_ref = slice_names[j]
    
            aligned_coor = STAligner.ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, landmark_domain)
            adata_target.obsm["spatial"] = aligned_coor
            
        for index,data in enumerate(Batch_list_copy):
            data.write('/SABench/AlignmentResults/STAligner/STAligner_new_slices_landmark'+landmark_domain[0]+'_'+slice_names[index]+'.h5ad')
       
    print("All results have been saved")
    
Batch_list = PrepocessingData(slices_list,slice_names)
used_device = torch.device('cuda:0')
STAlignerAlign(Batch_list,slice_names,used_device)