import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import ot
import torch
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import anndata as ad
from STAIR.emb_alignment import Emb_Align
from STAIR.utils import *
import os
os.environ['R_HOME'] = "/anaconda3/envs/STAIR/lib/R"
os.environ['R_USER'] = "/anaconda3/envs/STAIR/lib/python3.8/site-packages/rpy2"
from rpy2 import robjects

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

def STAIR_Align(group,slice_names,temp_path,device):
    new_slices=group.copy()
    for i in range(len(slice_names)):
        slice = new_slices[i]
        slice.obs_names_make_unique()
        slice.obs_names = [x+'_'+slice_names[i] for x in slice.obs_names]  
    adata = ad.concat(new_slices, label="slice_name", keys=slice_names)
    adata.obs["batch"] = adata.obs["slice_name"].astype('category')
    print('adata.shape: ', adata.shape)

    result_path = construct_folder(temp_path)
    keys_use = slice_names

    # Construct the model
    emb_align = Emb_Align(adata, batch_key='batch', result_path=result_path,device=device)
    emb_align.prepare()
    # Preprocessing
    emb_align.preprocess()
    emb_align.latent()

    # Construct the heterogeneous graph
    emb_align.prepare_hgat( spatial_key = 'spatial',slice_order = keys_use)
    # learning & integrating
    emb_align.train_hgat()
    adata, atte = emb_align.predict_hgat()
    #atte.to_csv(f'{result_path}/embedding/attention.csv')
    # clustering of spatial embedding
    adata = cluster_func(adata, clustering='mclust', use_rep='STAIR', cluster_num=13, key_add='STAIR')

    from STAIR.loc_prediction import sort_slices
    dists = sort_slices(atte, start=keys_use[0])

    adata.obs['z_rec'] = adata.obs['batch'].replace(dists).astype(float)
    adata.obs['z_rec'] = (adata.obs['z_rec']- adata.obs['z_rec'].min()) / (adata.obs['z_rec'].max() - adata.obs['z_rec'].min())

    adata.obs[['batch', 'z_rec']].drop_duplicates().sort_values('z_rec', ascending=False)['batch'].tolist()

    from STAIR.loc_alignment import Loc_Align

    keys_order = adata.obs[['batch', 'z_rec']].drop_duplicates().sort_values('z_rec', ascending=False)['batch'].tolist()
    loc_align = Loc_Align(adata, batch_key='batch', batch_order=keys_order, result_path=result_path)

    loc_align.init_align(   emb_key = 'STAIR',
                        spatial_key = 'spatial',
                        num_mnn = 1  )

    loc_align.detect_fine_points(  domain_key = 'STAIR',
                               slice_boundary = True,
                               domain_boundary = True,
                               num_domains = 1,
                               alpha = 500,     #
                               return_result = False)

    loc_align.plot_edge(spatial_key = 'transform_init',figsize = (6,6),s=2)
    adata = loc_align.fine_align()

    for i in range(len(slice_names)):
        new_slices[i] = adata[adata.obs['batch'] == slice_names[i]]
        new_slices[i].obsm['spatial'] = adata[adata.obs['batch'] == slice_names[i]].obsm['transform_fine'].toarray()
    
    return new_slices

new_slices=STAIR_Align(slices_list,slice_names,'/SABench/SpatialAlignment/STAIR/STAIR',device = 'cuda:0')

for i,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/STAIR/STAIR_new_{}.h5ad'.format(slice_names[i]))