import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import scanpy as sc
import pandas as pd
from gpsa import VariationalGPSA
from gpsa import matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod
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

def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    #adata.var["mt"] = adata.var_names.str.startswith("MT-")
    #sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=False)

    #sc.pp.filter_cells(adata, min_counts=4000)
    #sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    #sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=False
    )
    return adata

def GPSAAlign(group,slice_names,device):
    runtime=[None for i in range(len(group)-1)] 
    memory=[None for i in range(len(group)-1)] 
    
    new_slices=group.copy()
    for i in range(len(group)-1):
        
        import tracemalloc
        import time
        tracemalloc.start()
        start_time=time.time()

        
        #GPSA
        data_slice1=group[i].copy()
        data_slice2=group[i+1].copy()
        
        #
        data_slice1=process_data(data_slice1, n_top_genes=3000)
        data_slice2=process_data(data_slice2, n_top_genes=3000)
        
        data = data_slice1.concatenate(data_slice2)

        n_samples_list = [data_slice1.shape[0], data_slice2.shape[0]]
        view_idx = [
            np.arange(data_slice1.shape[0]),
            np.arange(data_slice1.shape[0], data_slice1.shape[0] + data_slice2.shape[0]),]

        X1 = data[data.obs.batch == "0"].obsm["spatial"]
        X2 = data[data.obs.batch == "1"].obsm["spatial"]
        Y1 = np.array(data[data.obs.batch == "0"].X.todense())
        Y2 = np.array(data[data.obs.batch == "1"].X.todense())
        def scale_spatial_coords(X, max_val=10.0):
            X = X - X.min(0)
            X = X / X.max(0)
            return X * max_val
        X1 = scale_spatial_coords(X1)
        X2 = scale_spatial_coords(X2)

        Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
        Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

        X = np.concatenate([X1, X2])
        Y = np.concatenate([Y1, Y2])

        n_outputs = Y.shape[1]

        x = torch.from_numpy(X).float().clone().to(device)
        y = torch.from_numpy(Y).float().clone().to(device)

        data_dict = {
            "expression": {
                "spatial_coords": x,
                "outputs": y,
                "n_samples_list": n_samples_list,
            }
        }
        
        #
        N_GENES = 10
        N_SAMPLES = None
        
        N_SPATIAL_DIMS = 2
        N_VIEWS = 2
        M_G = 200
        M_X_PER_VIEW =200
        
        FIXED_VIEW_IDX =0
        N_LATENT_GPS = {"expression": None}
        
        N_EPOCHS = 5000
        PRINT_EVERY = 1000
        #
        model = VariationalGPSA(
            data_dict,
            n_spatial_dims=N_SPATIAL_DIMS,
            m_X_per_view=M_X_PER_VIEW,
            m_G=M_G,
            data_init=True,
            minmax_init=False,
            grid_init=False,
            n_latent_gps=N_LATENT_GPS,
            mean_function="identity_fixed",
            kernel_func_warp=rbf_kernel,
            kernel_func_data=rbf_kernel,
            fixed_view_idx=FIXED_VIEW_IDX,#
        ).to(device)

        view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        def train(model, loss_fn, optimizer):
            model.train()

            # Forward pass
            G_means, G_samples, F_latent_samples, F_samples = model.forward(
                X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
            )

            # Compute loss
            loss = loss_fn(data_dict, F_samples)
            
            # Compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss.item(), G_means

        # 
        for t in range(N_EPOCHS):
            loss, G_means = train(model, model.loss_fn, optimizer)
            
            if t % PRINT_EVERY == 0:
                print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
                aligned_coords = G_means["expression"].cpu().detach().numpy()
        
        data_aligned = data.copy()
        data_aligned.obsm["spatial"] = aligned_coords

        newslice1= data_aligned[data_aligned.obs['batch'] == '0'].copy()
        newslice2= data_aligned[data_aligned.obs['batch'] == '1'].copy()
        new_slices[i]=newslice1
        new_slices[i+1]=newslice2
        
        #
        end_time=time.time()
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
        runtime[i]=end_time-start_time
        memory[i]=peak /1024/1024
        
        print('memory blocks peak:{:>10.4f} MB'.format(memory[i]))
        print('time: {:.4f} s'.format(runtime[i]))
        
    return new_slices    

new_slices=GPSAAlign(slices_list,slice_names,"cuda" )
for index,data in enumerate(new_slices):
    data.write('/SABench/AlignmentResults/GPSA/GPSA_DLPFCS_new_{}.h5ad'.format(slice_names[index]))
     