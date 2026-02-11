import numpy as np
import scanpy as sc
import matplotlib.patches as patches
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from alpha_shapes import Alpha_Shaper, plot_alpha_shape

import anndata as ad
from skimage import transform as trans

from skimage.transform import estimate_transform


#gene-based
def overlap_bygrids(slice1, slice2, spatial_key1, spatial_key2, m, n):
    adata1=slice1.copy()
    adata2=slice2.copy()
    coords1 = adata1.obsm[spatial_key1]
    coords2 = adata2.obsm[spatial_key2]

    x_min = max(np.min(coords1[:, 0]), np.min(coords2[:, 0]))
    x_max = min(np.max(coords1[:, 0]), np.max(coords2[:, 0]))
    y_min = max(np.min(coords1[:, 1]), np.min(coords2[:, 1]))
    y_max = min(np.max(coords1[:, 1]), np.max(coords2[:, 1]))

    def points_per_grid(coords, x_min, x_max, y_min, y_max, m, n):#ponits:spots/cells
        grid_points_counts = []
        for i in range(n):
            for j in range(m):
                x_start = x_min + i * (x_max - x_min) / n
                y_start = y_min + j * (y_max - y_min) / m
                x_end = x_start + (x_max - x_min) / n
                y_end = y_start + (y_max - y_min) / m
                ponits_count = np.sum((coords[:, 0] >= x_start) & (coords[:, 0] < x_end) &
                                    (coords[:, 1] >= y_start) & (coords[:, 1] < y_end))
                grid_points_counts.append(ponits_count)
        return np.mean(grid_points_counts)

    average_cells = min(
        points_per_grid(coords1, x_min, x_max, y_min, y_max, m, n),
        points_per_grid(coords2, x_min, x_max, y_min, y_max, m, n)
    )
    threshold_points = average_cells / 2

    def check_grid_area(adata_coords, x_start, y_start, x_end, y_end, threshold_points):
        points_in_grid = adata_coords[(adata_coords[:, 0] >= x_start) & (adata_coords[:, 0] < x_end) &
                                      (adata_coords[:, 1] >= y_start) & (adata_coords[:, 1] < y_end)]
        return len(points_in_grid) >= threshold_points

    # Prepare the 'grid_label' column in .obs with None
    adata1.obs['grid_label'] = None
    adata2.obs['grid_label'] = None

    grid_counter = 1
    for i in range(n):#x
        for j in range(m):#y
            x_start = x_min + i * (x_max - x_min) / n
            y_start = y_min + j * (y_max - y_min) / m
            x_end = x_start + (x_max - x_min) / n
            y_end = y_start + (y_max - y_min) / m

            in_adata1 = check_grid_area(coords1, x_start, y_start, x_end, y_end, threshold_points)
            in_adata2 = check_grid_area(coords2, x_start, y_start, x_end, y_end, threshold_points)

            label = f"Grid_{grid_counter}" if in_adata1 and in_adata2 else "NonOverlap"

            adata1.obs.loc[(coords1[:, 0] >= x_start) & (coords1[:, 0] < x_end) &
                           (coords1[:, 1] >= y_start) & (coords1[:, 1] < y_end), 'grid_label'] = label

            adata2.obs.loc[(coords2[:, 0] >= x_start) & (coords2[:, 0] < x_end) &
                           (coords2[:, 1] >= y_start) & (coords2[:, 1] < y_end), 'grid_label'] = label

            if in_adata1 and in_adata2:
                grid_counter += 1

    return adata1,adata2

def plot_with_grids_and_labels(adata1, adata2, spatial_key1, spatial_key2, n, m):
    coords1 = adata1.obsm[spatial_key1]
    coords2 = adata2.obsm[spatial_key2]

    plt.figure(figsize=(10, 10))

    plt.scatter(coords1[:, 0], coords1[:, 1], s=1, label='adata1')
    plt.scatter(coords2[:, 0], coords2[:, 1], s=1, label='adata2')

    x_min = max(np.min(coords1[:, 0]), np.min(coords2[:, 0]))
    x_max = min(np.max(coords1[:, 0]), np.max(coords2[:, 0]))
    y_min = max(np.min(coords1[:, 1]), np.min(coords2[:, 1]))
    y_max = min(np.max(coords1[:, 1]), np.max(coords2[:, 1]))

    # 
    for i in range(n):  # x
        for j in range(m):  # y
            x_start = x_min + i * (x_max - x_min) / n
            y_start = y_min + j * (y_max - y_min) / m
            x_end = x_start + (x_max - x_min) / n
            y_end = y_start + (y_max - y_min) / m

            labels1 = adata1.obs.loc[
                (adata1.obsm[spatial_key1][:, 0] >= x_start) & (adata1.obsm[spatial_key1][:, 0] < x_end) &
                (adata1.obsm[spatial_key1][:, 1] >= y_start) & (adata1.obsm[spatial_key1][:, 1] < y_end), 'grid_label']
            labels2 = adata2.obs.loc[
                (adata2.obsm[spatial_key2][:, 0] >= x_start) & (adata2.obsm[spatial_key2][:, 0] < x_end) &
                (adata2.obsm[spatial_key2][:, 1] >= y_start) & (adata2.obsm[spatial_key2][:, 1] < y_end), 'grid_label']

            if len(labels1) == 0 or labels1.values[0] == 'NonOverlap' or labels1.isnull().values.any():
                label = None
            else:
                label = labels1.values[0]

            #if (label is None) and (
            #        len(labels2) > 0 and labels2.values[0] != 'NonOverlap' and not labels2.isnull().values.any()):
            #    label = labels2.values[0]

            if label is not None:
                # 
                rect = patches.Rectangle((x_start, y_start), (x_end - x_start), (y_end - y_start),
                                         linewidth=1, edgecolor='black', facecolor='none')
                plt.gca().add_patch(rect)
                # label
                number = label.split('_')[-1]
                plt.text((x_start + x_end) / 2, (y_start + y_end) / 2, f'{number}',
                         horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')

    plt.legend()
    plt.show()

def PCC_bygrid(slice1, slice2, gene_name):
    adata1, adata2 = overlap_bygrids(slice1, slice2, 'spatial', 'spatial', m=10, n=10)

    if gene_name not in adata1.var_names or gene_name not in adata2.var_names:
        raise ValueError(f"Gene {gene_name} not found in both of the slices.")

    gene_expression1 = adata1.X[:, adata1.var_names == gene_name].toarray().squeeze()
    gene_expression2 = adata2.X[:, adata2.var_names == gene_name].toarray().squeeze()

    df1 = pd.DataFrame({'expression': gene_expression1, 'grid_label': adata1.obs['grid_label']})
    df2 = pd.DataFrame({'expression': gene_expression2, 'grid_label': adata2.obs['grid_label']})

    df1 = df1[(df1['grid_label'] != 'NonOverlap') & (df1['grid_label'].notnull())]
    df2 = df2[(df2['grid_label'] != 'NonOverlap') & (df2['grid_label'].notnull())]

    mean_expression1 = df1.groupby('grid_label').mean().sort_index()
    mean_expression2 = df2.groupby('grid_label').mean().sort_index()

    if mean_expression1.size<2 or mean_expression2.size<2:
        PCC=0
    else:
        PCC,pvalue=pearsonr(mean_expression1['expression'], mean_expression2['expression'])
    
    return PCC

def cos_sim_bygrid(slice1, slice2, gene_name):
    adata1, adata2 = overlap_bygrids(slice1, slice2, 'spatial', 'spatial', m=10, n=10)

    if gene_name not in adata1.var_names or gene_name not in adata2.var_names:
        raise ValueError(f"Gene {gene_name} not found in both of the slices.")

    gene_expression1 = adata1.X[:, adata1.var_names == gene_name].toarray().squeeze()
    gene_expression2 = adata2.X[:, adata2.var_names == gene_name].toarray().squeeze()

    df1 = pd.DataFrame({'expression': gene_expression1, 'grid_label': adata1.obs['grid_label']})
    df2 = pd.DataFrame({'expression': gene_expression2, 'grid_label': adata2.obs['grid_label']})

    df1 = df1[(df1['grid_label'] != 'NonOverlap') & (df1['grid_label'].notnull())]
    df2 = df2[(df2['grid_label'] != 'NonOverlap') & (df2['grid_label'].notnull())]

    mean_expression1 = np.array(df1.groupby('grid_label').mean().sort_index()).squeeze()
    mean_expression2 =  np.array(df2.groupby('grid_label').mean().sort_index()).squeeze()

    if mean_expression1.size<2 or mean_expression2.size<2:
        cos_sim=0
    else:
        cos_sim = mean_expression1.dot(mean_expression2) / (np.linalg.norm(mean_expression1) * np.linalg.norm(mean_expression2))
        
    return cos_sim


def ssim_bygrid(slice1, slice2, gene_name, L=1):
    adata1, adata2 = overlap_bygrids(slice1, slice2, 'spatial', 'spatial', m=10, n=10)

    if gene_name not in adata1.var_names or gene_name not in adata2.var_names:
        raise ValueError(f"Gene {gene_name} not found in both of the slices.")

    gene_expression1 = adata1.X[:, adata1.var_names == gene_name].toarray().squeeze()
    gene_expression2 = adata2.X[:, adata2.var_names == gene_name].toarray().squeeze()

    df1 = pd.DataFrame({'expression': gene_expression1, 'grid_label': adata1.obs['grid_label']})
    df2 = pd.DataFrame({'expression': gene_expression2, 'grid_label': adata2.obs['grid_label']})

    df1 = df1[(df1['grid_label'] != 'NonOverlap') & (df1['grid_label'].notnull())]
    df2 = df2[(df2['grid_label'] != 'NonOverlap') & (df2['grid_label'].notnull())]

    mean_expression1 = np.array(df1.groupby('grid_label').mean().sort_index()).squeeze()
    mean_expression2 = np.array(df2.groupby('grid_label').mean().sort_index()).squeeze()

    if mean_expression1.size<2 or mean_expression2.size<2:
        ssim=0
    else:
        mean_expression1, mean_expression2 = mean_expression1/mean_expression1.max(), mean_expression2/mean_expression2.max()
        mu1 = mean_expression1.mean()
        mu2 = mean_expression2.mean()
        sigma1 = np.sqrt(((mean_expression1 - mu1) ** 2).mean())
        sigma2 = np.sqrt(((mean_expression2 - mu2) ** 2).mean())
        sigma12 = ((mean_expression1 - mu1) * (mean_expression2 - mu2)).mean()
        k1, k2= 0.01, 0.03
        C1 = (k1*L) ** 2
        C2 = (k2*L) ** 2
        C3 = C2/2
        l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
        c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
        s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
        ssim = l12 * c12 * s12
    return ssim

def MI_bygrid(slice1, slice2, gene_name):
    adata1, adata2 = overlap_bygrids(slice1, slice2, 'spatial', 'spatial', m=10, n=10)

    if gene_name not in adata1.var_names or gene_name not in adata2.var_names:
        raise ValueError(f"Gene {gene_name} not found in both of the slices.")

    gene_expression1 = adata1.X[:, adata1.var_names == gene_name].toarray().squeeze()
    gene_expression2 = adata2.X[:, adata2.var_names == gene_name].toarray().squeeze()

    df1 = pd.DataFrame({'expression': gene_expression1, 'grid_label': adata1.obs['grid_label']})
    df2 = pd.DataFrame({'expression': gene_expression2, 'grid_label': adata2.obs['grid_label']})

    df1 = df1[(df1['grid_label'] != 'NonOverlap') & (df1['grid_label'].notnull())]
    df2 = df2[(df2['grid_label'] != 'NonOverlap') & (df2['grid_label'].notnull())]

    mean_expression1 = np.array(df1.groupby('grid_label').mean().sort_index())
    mean_expression2 = np.array(df2.groupby('grid_label').mean().sort_index()).squeeze()

    if mean_expression1.size<2 or mean_expression2.size<2:
        mi=[0]
    else:
        mi = mutual_info_regression(mean_expression1, mean_expression2)
    return mi[0]


def ASSD_bygrid(slice1, slice2, gene_name):
    adata1, adata2 = overlap_bygrids(slice1, slice2, 'spatial', 'spatial', m=10, n=10)

    if gene_name not in adata1.var_names or gene_name not in adata2.var_names:
        raise ValueError(f"Gene {gene_name} not found in both of the slices.")

    gene_expression1 = adata1.X[:, adata1.var_names == gene_name].toarray().squeeze()
    gene_expression2 = adata2.X[:, adata2.var_names == gene_name].toarray().squeeze()

    df1 = pd.DataFrame({'expression': gene_expression1, 'grid_label': adata1.obs['grid_label']})
    df2 = pd.DataFrame({'expression': gene_expression2, 'grid_label': adata2.obs['grid_label']})

    df1 = df1[(df1['grid_label'] != 'NonOverlap') & (df1['grid_label'].notnull())]
    df2 = df2[(df2['grid_label'] != 'NonOverlap') & (df2['grid_label'].notnull())]

    mean_expression1 = np.array(df1.groupby('grid_label').mean().sort_index()).squeeze()
    mean_expression2 = np.array(df2.groupby('grid_label').mean().sort_index()).squeeze()

    if mean_expression1.size<2 or mean_expression2.size<2:
        ASSD=100  #
    else:
        ASSD= np.mean(np.square(mean_expression1 - mean_expression2))
    return ASSD

def Avg_pcc_values_for_method(method_results, gene_list):
    pcc_values = []
    for i in range(len(method_results)-1):
        slice1 = method_results[i]
        slice2 = method_results[i+1]
        pcc_values_pair = []
        for gene_name in gene_list:
            pcc = PCC_bygrid(slice1, slice2, gene_name)
            pcc_values_pair.append(pcc)
        pcc_values.append(pcc_values_pair)
    average_pcc = np.mean(np.array(pcc_values), axis=0).tolist()
    return average_pcc

def Avg_cos_sim_values_for_method(method_results, gene_list):
    cos_sim_values = []
    for i in range(len(method_results)-1):
        slice1 = method_results[i]
        slice2 = method_results[i+1]
        cos_sim_values_pair = []
        for gene_name in gene_list:
            cos_sim = cos_sim_bygrid(slice1, slice2, gene_name)
            cos_sim_values_pair.append(cos_sim)
        cos_sim_values.append(cos_sim_values_pair)
    average_cos_sim = np.mean(np.array(cos_sim_values), axis=0).tolist()
    return average_cos_sim

def Avg_ssim_values_for_method(method_results, gene_list):
    ssim_values = []
    for i in range(len(method_results)-1):
        slice1 = method_results[i]
        slice2 = method_results[i+1]
        ssim_values_pair = []
        for gene_name in gene_list:
            ssim = ssim_bygrid(slice1, slice2, gene_name)
            ssim_values_pair.append(ssim)
        ssim_values.append(ssim_values_pair)
    average_ssim = np.mean(np.array(ssim_values), axis=0).tolist()
    return average_ssim

def Avg_MI_values_for_method(method_results, gene_list):
    MI_values = []
    for i in range(len(method_results)-1):
        slice1 = method_results[i]
        slice2 = method_results[i+1]
        MI_values_pair = []
        for gene_name in gene_list:
            MI = MI_bygrid(slice1, slice2, gene_name)
            MI_values_pair.append(MI)
        MI_values.append(MI_values_pair)
    average_MI = np.mean(np.array(MI_values), axis=0).tolist()
    return average_MI

def Avg_ASSD_values_for_method(method_results, gene_list):
    ASSD_values = []
    for i in range(len(method_results)-1):
        slice1 = method_results[i]
        slice2 = method_results[i+1]
        ASSD_values_pair = []
        for gene_name in gene_list:
            ASSD = ASSD_bygrid(slice1, slice2, gene_name)
            ASSD_values_pair.append(ASSD)
        ASSD_values.append(ASSD_values_pair)
    average_ASSD = np.mean(np.array(ASSD_values), axis=0).tolist()
    return average_ASSD

#landmark-based
def Overlap_accuracy(adata1, adata2, spatial_key):
    coords1 = adata1.obsm[spatial_key]
    coords2 = adata2.obsm[spatial_key]

    x_min = max(np.min(coords1[:, 0]), np.min(coords2[:, 0]))
    x_max = min(np.max(coords1[:, 0]), np.max(coords2[:, 0]))
    y_min = max(np.min(coords1[:, 1]), np.min(coords2[:, 1]))
    y_max = min(np.max(coords1[:, 1]), np.max(coords2[:, 1]))

    overlap_indices1 = (coords1[:, 0] >= x_min) & (coords1[:, 0] <= x_max) & \
                       (coords1[:, 1] >= y_min) & (coords1[:, 1] <= y_max)
    overlap_indices2 = (coords2[:, 0] >= x_min) & (coords2[:, 0] <= x_max) & \
                       (coords2[:, 1] >= y_min) & (coords2[:, 1] <= y_max)

    overlapping_adata1 = adata1[overlap_indices1, :]
    overlapping_adata2 = adata2[overlap_indices2, :]

    overlap_coords1 = overlapping_adata1.obsm[spatial_key]
    overlap_coords2 = overlapping_adata2.obsm[spatial_key]

    tree = cKDTree(overlap_coords1)

    distances, indices = tree.query(overlap_coords2)

    if len(indices)==0:
        accuracy=0
    else:
        # check Region
        matching_labels_count = 0
        for idx, target_idx in enumerate(indices):
            if target_idx < len(overlapping_adata1.obs) and idx < len(overlapping_adata2.obs):
                label1 = overlapping_adata1.obs['Region'][target_idx]
                label2 = overlapping_adata2.obs['Region'][idx]

                if label1 == label2:
                    matching_labels_count += 1

        accuracy = matching_labels_count / len(indices)
    return accuracy

def Average_Accuracy(method_results):
    accuracy_values=[]
    for i in range(len(method_results)-1):
        slice1 = method_results[i]
        slice2 = method_results[i+1]
        accuracy = Overlap_accuracy(slice1,slice2,'spatial')
        accuracy_values.append(accuracy)
        
      
    average_accuracy = np.average(accuracy_values)
    return average_accuracy

def ROI_overlap(adata1, adata2, spatial_key, label_key, label):

    coords1 = adata1.obsm[spatial_key]
    coords2 = adata2.obsm[spatial_key]

    x_min = max(np.min(coords1[:, 0]), np.min(coords2[:, 0]))
    x_max = min(np.max(coords1[:, 0]), np.max(coords2[:, 0]))
    y_min = max(np.min(coords1[:, 1]), np.min(coords2[:, 1]))
    y_max = min(np.max(coords1[:, 1]), np.max(coords2[:, 1]))

    overlap_indices1 = (coords1[:, 0] >= x_min) & (coords1[:, 0] <= x_max) & \
                       (coords1[:, 1] >= y_min) & (coords1[:, 1] <= y_max)
    overlap_indices2 = (coords2[:, 0] >= x_min) & (coords2[:, 0] <= x_max) & \
                       (coords2[:, 1] >= y_min) & (coords2[:, 1] <= y_max)

    adata1 = adata1[overlap_indices1, :]
    adata2 = adata2[overlap_indices2, :]

    mask1 = np.where(adata1.obs[label_key] == label)
    mask2 = np.where(adata2.obs[label_key] == label)

    if len(mask1[0])==0 or len(mask2[0])==0:
        overlap_ratio=0
    else:
        alpha = 12.0
        points1=adata1.obsm['spatial'][mask1, ].squeeze()
        shaper1 = Alpha_Shaper(points1)
        alpha_shape1 = shaper1.get_shape(alpha=alpha)
        points2 = adata2.obsm['spatial'][mask2,].squeeze()
        shaper2 = Alpha_Shaper(points2)
        alpha_shape2 = shaper2.get_shape(alpha=alpha)

    
        intersection = alpha_shape1.intersection(alpha_shape2)
        union = alpha_shape1.union(alpha_shape2)

        intersection_area = intersection.area
        union_area = union.area

        overlap_ratio = intersection_area / union_area

    return overlap_ratio

def Average_ROI_Overlap(method_results,label):
    overlap_ratio_values=[]
    for i in range(len(method_results)-1):
        slice1 = method_results[i]
        slice2 = method_results[i+1]
        overlap_ratio = ROI_overlap(slice1,slice2,'spatial', 'Region', label)
        overlap_ratio_values.append(overlap_ratio)
        
    average_ROI_overlap = np.average(overlap_ratio_values)
    return average_ROI_overlap




def Change_slices_resolution(adata, nx, ny):
    """
    Change the resolution of spatial omics data in an AnnData object to a new AnnData object, where each rectangular region is treated as an observation (obs). The center of the rectangle serves as the coordinate, and the X matrix is constructed from the sum of feature values of all cells within the region. If a rectangle contains no cells, it is not created.

    Parameters:
      adata: anndata.AnnData object containing spatial omics data.
      nx: int, number of rectangles along the x-axis.
      ny: int, number of rectangles along the y-axis.

    Returns:
    A new AnnData object, where each obs represents a rectangular region containing at least one cell.
    """
    spatial_coords = adata.obsm['spatial']
    x_min, x_max = np.min(spatial_coords[:, 0]), np.max(spatial_coords[:, 0])
    y_min, y_max = np.min(spatial_coords[:, 1]), np.max(spatial_coords[:, 1])
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    
    new_spatial_coords = []
    aggregated_X = []
    
    for i in range(nx):
        for j in range(ny):
            x_center = x_min + (i + 0.5) * dx
            y_center = y_min + (j + 0.5) * dy

            x_start, x_end = x_min + i*dx, x_min + (i+1)*dx
            y_start, y_end = y_min + j*dy, y_min + (j+1)*dy
            
            rect_mask = (spatial_coords[:, 0] >= x_start) & (spatial_coords[:, 0] < x_end) & \
                        (spatial_coords[:, 1] >= y_start) & (spatial_coords[:, 1] < y_end)
            
            if np.any(rect_mask):
                # X
                agg_X = np.sum(adata.X[rect_mask], axis=0)
              
                new_spatial_coords.append([x_center, y_center])
                aggregated_X.append(agg_X)
    
    new_obs = pd.DataFrame(index=range(len(new_spatial_coords)), columns=['x_center', 'y_center'])
    new_obs[['x_center', 'y_center']] = np.array(new_spatial_coords)
    
    new_adata = ad.AnnData(X=np.squeeze(aggregated_X), obs=new_obs, var=adata.var)
    new_adata.obsm['spatial'] = np.array(new_spatial_coords)
    
    return new_adata


def restore_initial_slices(group_raw,group_DR,group_results):
    '''
    group_raw: Original resolution
    group_DR: Original data with downsampled resolution
    group_results: alignment results of the downsampled resolution data
    This function returns the alignment results at the original resolution.
    '''
    s1=np.array(group_DR[0].obsm['spatial'],dtype=np.float32)
    d1=np.array(group_results[0].obsm['spatial'],dtype=np.float32)
 
    tform1 = trans.SimilarityTransform()
    tform1.estimate(s1,d1)
    A1=tform1.params

    s2=np.array(group_DR[1].obsm['spatial'])
    d2=np.array(group_results[1].obsm['spatial'])
 
    tform2 = trans.SimilarityTransform()
    tform2.estimate(s2,d2)
    A2=tform2.params

    newcoods1=np.dot(A1,np.transpose(np.hstack((group_raw[0].obsm['spatial'], np.ones((group_raw[0].obsm['spatial'].shape[0], 1))))))
    newcoods2=np.dot(A2,np.transpose(np.hstack((group_raw[1].obsm['spatial'], np.ones((group_raw[1].obsm['spatial'].shape[0], 1))))))
    new1=group_raw[0].copy()
    new2=group_raw[1].copy()
    new1.obsm['spatial']=np.transpose(np.delete(newcoods1, -1, axis=0))
    new2.obsm['spatial']=np.transpose(np.delete(newcoods2, -1, axis=0))
    new_group_raw=[new1,new2]
    
    return new_group_raw

def plot_spatial(adata, color_by=None, title="Spatial", spot_size=60, ax=None):
    """
    color_by = None           use total counts for coloring
    color_by = 'region'       use obs['region'] for categorical coloring
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    spatial = adata.obsm['spatial']
    
    if color_by is None or color_by not in adata.obs.columns:
        # use total counts
        counts = np.ravel(adata.X.sum(axis=1))
        if hasattr(counts, "toarray"):
            counts = counts.toarray().ravel()
        sc = ax.scatter(spatial[:, 0], spatial[:, 1], c=counts, cmap='viridis',
                        s=spot_size, edgecolor='none', alpha=0.9)
        plt.colorbar(sc, ax=ax, shrink=0.7, label='Total counts')
    else:
        # categorical coloring
        cats = adata.obs[color_by].astype('category')
        codes = cats.cat.codes
        n_cats = len(cats.cat.categories)
        cmap = plt.get_cmap('tab20' if n_cats <= 20 else 'tab20b')
        sc = ax.scatter(spatial[:, 0], spatial[:, 1], c=codes, cmap=cmap,
                        s=spot_size, edgecolor='none', alpha=0.9)

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i/(n_cats-1 if n_cats>1 else 1)),
                              markersize=10, label=l) for i, l in enumerate(cats.cat.categories)]
        ax.legend(handles=handles, title=color_by, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    if ax is None:
        plt.tight_layout()
        plt.show()
    return ax

def interactive_coarse_align(adata_fixed, adata_moving, PLOT_key="region"):
    """
    Return: adata_fixed, adata_moving_aligned
    During execution it will open two windows for point clicking
    """

    if PLOT_key not in adata_fixed.obs.columns or PLOT_key not in adata_moving.obs.columns:
        print(f"Warning: column '{PLOT_key}' not found in obs, switching to total counts coloring.")
        PLOT_key = None

    print(f"\nUsing '{PLOT_key or 'total counts'}' for coloring")

    # plotting
    fig1, ax1 = plt.subplots(figsize=(8,8))
    fig2, ax2 = plt.subplots(figsize=(8,8))
    plot_spatial(adata_fixed, color_by=PLOT_key, title=f"Fixed : Click feature points", ax=ax1)
    plot_spatial(adata_moving, color_by=PLOT_key, title=f"Moving : Click corresponding points", ax=ax2)

    plt.figure(fig1.number); plt.show(block=False)
    plt.figure(fig2.number); plt.show(block=False)

    try:
        fig1.canvas.manager.window.move(50, 50)
        fig2.canvas.manager.window.move(900, 50)
    except:
        pass

    fixed_pts, moving_pts = [], []

    print("\n" + "="*70)
    print("Start marking points (at least 3 pairs):")
    print("1. Click a landmark in the Fugure 1 (Fixed)")
    print("2. Click the corresponding landmark in the Fugure 2 (Moving)")
    print("Close the windows to finish")
    print("="*70)

    while True:
        plt.figure(fig1.number)
        plt.suptitle(f"Fixed: {len(fixed_pts)} points selected, click next", color='red', fontsize=14)
        fig1.canvas.draw()
        pt = plt.ginput(1, timeout=0)
        if not pt: break
        x, y = pt[0]
        fixed_pts.append((x, y))
        ax1.scatter(x, y, c='red', s=400, marker='+', linewidth=5)
        fig1.canvas.draw()

        plt.figure(fig2.number)
        plt.suptitle(f"Moving: {len(moving_pts)} points selected, click corresponding point", color='red', fontsize=14)
        fig2.canvas.draw()
        pt = plt.ginput(1, timeout=0)
        if not pt:
            fixed_pts.pop()
            break
        x, y = pt[0]
        moving_pts.append((x, y))
        ax2.scatter(x, y, c='lime', s=400, marker='+', linewidth=5)
        fig2.canvas.draw()

        print(f"  Pair {len(moving_pts)} completed")

    plt.close(fig1)
    plt.close(fig2)

    if len(fixed_pts) < 3:
        print("Less than 3 pairs of points. Aborting.")
        return None, None

    # compute affine transform
    src = np.array(moving_pts)[:, ::-1]
    dst = np.array(fixed_pts)[:, ::-1]
    tform = estimate_transform('affine', src, dst)
    print(f"\nSuccess! {len(fixed_pts)} pairs used.")
    print("Affine transform matrix:")
    print(tform.params.round(4))

    # apply transform
    coords = adata_moving.obsm['spatial']
    homog = np.column_stack([coords, np.ones(len(coords))])
    aligned_coords = (tform.params @ homog.T).T[:, :2]

    adata_moving_aligned = adata_moving.copy()
    adata_moving_aligned.obsm['spatial'] = aligned_coords.astype(coords.dtype)

    return adata_fixed, adata_moving_aligned
