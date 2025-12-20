import os
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy
import scipy.sparse as sparse
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import gc  # Added: Garbage collection
import stat  # Added: File permission handling

# Disable HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


def remove_readonly(func, path, excinfo):
    """
    Helper function to handle 'Access Denied' errors during directory deletion.
    This is common on Windows if files are marked read-only or temporarily locked.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cal_spatial_adjacency_matrix(adata, n_neighbors, radius):
    """
    Calculates the spatial adjacency matrix based on Euclidean distance.
    """
    spatial_coor = pd.DataFrame(adata.obsm['spatial'])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(spatial_coor)
    distances, indices = nbrs.kneighbors(spatial_coor)

    n = spatial_coor.shape[0]
    adj_matrix = lil_matrix((n, n))

    for i in range(n):
        for idx, j in enumerate(indices[i, 1:]):
            if distances[i, idx + 1] < radius:
                adj_matrix[i, j] = 1

    return adj_matrix.tocsr()


def compute_cross_slice_mnn(adata, n_neighbors=30):
    """
    Computes the Mutual Nearest Neighbors (MNN) adjacency matrix across different slices.
    """
    slice_labels = adata.obs['slice_labels'].unique()
    n_cells = adata.n_obs
    mnn_adj_matrix = sparse.lil_matrix((n_cells, n_cells))

    # Build Annoy index for each slice
    annoy_indices = {}
    for label in slice_labels:
        indices = adata.obs['slice_labels'] == label
        X = adata.X[indices]

        if sparse.issparse(X):
            X = X.toarray()

        f = X.shape[1]
        t = AnnoyIndex(f, 'euclidean')
        for i in range(X.shape[0]):
            t.add_item(i, X[i])
        t.build(50)
        annoy_indices[label] = (t, indices)

    # Compute MNNs
    for i, label1 in enumerate(slice_labels):
        for label2 in slice_labels[i + 1:]:
            t1, indices1 = annoy_indices[label1]
            t2, indices2 = annoy_indices[label2]

            X1 = adata.X[indices1]
            X2 = adata.X[indices2]
            if sparse.issparse(X1): X1 = X1.toarray()
            if sparse.issparse(X2): X2 = X2.toarray()

            nn_indices1 = []
            for idx in range(X1.shape[0]):
                nn_indices1.append(t2.get_nns_by_vector(X1[idx], n_neighbors))

            nn_indices2 = []
            for idx in range(X2.shape[0]):
                nn_indices2.append(t1.get_nns_by_vector(X2[idx], n_neighbors))

            nn_indices1 = np.array(nn_indices1)
            nn_indices2 = np.array(nn_indices2)

            global_indices1 = np.where(indices1)[0]
            global_indices2 = np.where(indices2)[0]

            for r in range(X1.shape[0]):
                for c in nn_indices1[r]:
                    if r in nn_indices2[c]:
                        mnn_adj_matrix[global_indices1[r], global_indices2[c]] = 1
                        mnn_adj_matrix[global_indices2[c], global_indices1[r]] = 1

    return mnn_adj_matrix.tocsr()


def compute_knn_adjacency_matrix(adata, n_neighbors=30):
    """
    Computes the K-Nearest Neighbors (KNN) adjacency matrix.
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine')
    knn.fit(X)
    distances, indices = knn.kneighbors(X)

    n_cells = X.shape[0]
    adjacency_matrix = lil_matrix((n_cells, n_cells))

    for i in range(n_cells):
        for j in indices[i][1:]:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

    return adjacency_matrix.tocsr()


def average_degree(adj_matrix):
    """
    Calculates the average degree of a sparse adjacency matrix.
    """
    degrees = adj_matrix.sum(axis=1).A1 if scipy.sparse.issparse(adj_matrix) else adj_matrix.sum(axis=1)
    return np.mean(degrees)


def construct_graph(banksy_dir):
    """
    Main pipeline to read processed BANKSY files, construct multi-view graphs,
    merge data, and clean up temporary files.
    """
    print(f"Loading files from {banksy_dir}...")

    if not os.path.exists(banksy_dir):
        raise FileNotFoundError(f"Directory {banksy_dir} not found.")

    all_files = os.listdir(banksy_dir)
    h5_files = [f for f in all_files if f.endswith('.h5ad')]
    h5_files.sort()

    adata_list = []
    X_adata_list = []
    X_expr_adata_list = []
    X_gra_adata_list = []
    exp_adj_list = []
    spatial_adj_list = []

    for h5_file in h5_files:
        h5_file_path = os.path.join(banksy_dir, h5_file)
        print(f"Processing {h5_file}...")
        adata = sc.read_h5ad(h5_file_path)

        if 'slice_labels' not in adata.obs:
            adata.obs['slice_labels'] = h5_file

        adata_list.append(adata)
        n_features = adata.n_vars
        third = n_features // 3

        # Split features
        X_adata = adata[:, :third].copy()
        X_adata = ad.AnnData(X_adata.X, obs=adata.obs.copy(), var=adata.var.iloc[:third].copy(), obsm=adata.obsm.copy())
        X_adata_list.append(X_adata)

        X_expr_adata = adata[:, third:2 * third].copy()
        X_expr_adata = ad.AnnData(X_expr_adata.X, obs=adata.obs.copy(), var=adata.var.iloc[third:2 * third].copy(),
                                  obsm=adata.obsm.copy())
        X_expr_adata_list.append(X_expr_adata)

        X_gra_adata = adata[:, 2 * third:].copy()
        X_gra_adata = ad.AnnData(X_gra_adata.X, obs=adata.obs.copy(), var=adata.var.iloc[2 * third:].copy(),
                                 obsm=adata.obsm.copy())
        X_gra_adata_list.append(X_gra_adata)

        # Adjacency
        X_adata_adj = compute_knn_adjacency_matrix(X_adata)
        X_expr_adata_adj = compute_knn_adjacency_matrix(X_expr_adata)
        X_gra_adata_adj = compute_knn_adjacency_matrix(X_gra_adata)

        combined_adj_matrix = X_adata_adj + X_expr_adata_adj + X_gra_adata_adj
        final_adj_matrix = combined_adj_matrix >= 2
        final_adj_matrix = final_adj_matrix.astype(int)

        exp_adj_matrix = final_adj_matrix.tocsr()
        exp_average_degree = average_degree(exp_adj_matrix)
        print(f"  --> Average degree: {exp_average_degree:.4f}")
        exp_adj_list.append(exp_adj_matrix)

        spatial_knn_adj = cal_spatial_adjacency_matrix(adata, n_neighbors=6, radius=150)
        spatial_adj_list.append(spatial_knn_adj)

    print("Concatenating data...")
    adata_merged_all = ad.concat(adata_list, label="batch_id_unique")

    X_adata_concat = ad.concat(X_adata_list, join='outer')
    spatial_coords_list = [a.obsm['spatial'] for a in X_adata_list]
    spatial_coords_concat = np.vstack(spatial_coords_list)
    X_adata_concat.obsm['spatial'] = spatial_coords_concat

    X_expr_concat = ad.concat(X_expr_adata_list, join='outer')
    X_gra_concat = ad.concat(X_gra_adata_list, join='outer')

    print("Constructing block diagonal matrices...")
    exp_adj_concat = np.asarray(exp_adj_list[0].todense())
    for batch_id in range(1, len(exp_adj_list)):
        exp_adj_concat = scipy.linalg.block_diag(exp_adj_concat, np.asarray(exp_adj_list[batch_id].todense()))

    exp_adj_concat = sparse.csr_matrix(exp_adj_concat)

    print("Computing Cross-Slice MNN...")
    exp_mnn = compute_cross_slice_mnn(adata_merged_all, n_neighbors=50)

    average_mnn = average_degree(exp_mnn)
    print(f"MNN Average Degree: {average_mnn:.4f}")

    adj_concat = exp_mnn + exp_adj_concat

    spatial_adj_concat = np.asarray(spatial_adj_list[0].todense())
    for batch_id in range(1, len(spatial_adj_list)):
        spatial_adj_concat = scipy.linalg.block_diag(spatial_adj_concat,
                                                     np.asarray(spatial_adj_list[batch_id].todense()))
    spatial_adj_concat = sparse.csr_matrix(spatial_adj_concat)

    # -------------------------------------------------------------
    # Robust Directory Cleanup Logic
    # -------------------------------------------------------------
    print(f"Cleaning up intermediate directory: {banksy_dir}")

    # 1. Clear simple list references to help GC
    del adata_list
    del all_files
    del h5_files

    # 2. Force Garbage Collection to release file handles
    gc.collect()

    # 3. Try deleting with error handler
    try:
        shutil.rmtree(banksy_dir, onerror=remove_readonly)
        print("Cleanup successful.")
    except Exception as e:
        print(f"Warning: Failed to delete {banksy_dir}. Please delete manually. Reason: {e}")

    return X_adata_concat, X_expr_concat, X_gra_concat, adj_concat, spatial_adj_concat, average_mnn

