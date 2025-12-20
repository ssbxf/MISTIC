import os
import re
import gc
import warnings
import scanpy as sc
import numpy as np
import scipy.sparse
import anndata as ad
from anndata import AnnData

# Relative imports (Using . to access sibling banksy package)
from .banksy.main import median_dist_to_nearest_neighbour, concatenate_all
from .banksy.initialize_banksy import initialize_banksy
from .banksy.embed_banksy import generate_banksy_matrix

# Disable HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings("ignore")


def natural_sort_key(s):
    """
    Helper function: Natural sort key.
    Ensures '2A' comes before '10A' in sorting order.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def load_and_preprocess(input_dir, n_top_genes=5000, target_sum=1e4):
    """
    Step 1: Load .h5ad files from the directory, sort them naturally,
    and perform basic preprocessing (Normalization, Log1p, HVG).
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    all_files = os.listdir(input_dir)
    # Filter for .h5ad files
    h5ad_files = [f for f in all_files if f.endswith('.h5ad')]
    # Apply natural sort
    h5ad_files.sort(key=natural_sort_key)

    batch_list = []
    file_names = []

    print(f"Found {len(h5ad_files)} files. Loading...")

    for h5ad_file in h5ad_files:
        file_path = os.path.join(input_dir, h5ad_file)
        try:
            adata = sc.read_h5ad(file_path)

            # Preprocessing
            # flavor='seurat_v3' expects raw counts
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)

            # Keep only highly variable genes
            adata = adata[:, adata.var['highly_variable']].copy()

            # Extract filename without extension
            fname_no_ext = os.path.splitext(h5ad_file)[0]

            batch_list.append(adata)
            file_names.append(fname_no_ext)

            print(f"Loaded: {fname_no_ext}")

        except Exception as e:
            print(f"Error loading {h5ad_file}: {e}")

    return batch_list, file_names


def align_common_genes(batch_list):
    """
    Step 2: Intersect genes across all batches to ensure they share the same feature space.
    """
    if not batch_list:
        return []

    # Find common genes across all adata objects
    common_genes = set(batch_list[0].var_names)
    for adata in batch_list[1:]:
        common_genes &= set(adata.var_names)

    common_genes = list(common_genes)
    common_genes.sort()  # Ensure consistent order

    print(f"Number of common genes: {len(common_genes)}")

    processed_list = []
    for i, adata in enumerate(batch_list):
        # Subset to common genes
        adata_subset = adata[:, common_genes].copy()

        # Rename index to avoid duplicates during concatenation
        # Format: {list_index}_{original_index}
        adata_subset.obs.index = [f"{i}_{idx}" for idx in adata_subset.obs.index]

        processed_list.append(adata_subset)

    return processed_list


def _compute_banksy(adata, coord_keys, k_geom, max_m, nbr_weight_decay):
    """Internal helper function to run the core BANKSY workflow on a single adata."""
    spatial_key = coord_keys[2]

    # Calculate median distance to nearest neighbours
    nbrs = median_dist_to_nearest_neighbour(adata, key=spatial_key)

    # Initialize BANKSY parameters
    banksy_dict = initialize_banksy(adata,
                                    coord_keys,
                                    k_geom,
                                    nbr_weight_decay=nbr_weight_decay,
                                    max_m=max_m,
                                    plt_edge_hist=False,
                                    plt_nbr_weights=False,
                                    plt_agf_angles=False)

    lambda_list = [0.2]

    # Generate the BANKSY matrix
    banksy_dict, banksy_matrix = generate_banksy_matrix(adata,
                                                        banksy_dict,
                                                        lambda_list,
                                                        max_m=max_m)
    return banksy_matrix


def get_banksy_results(batch_list, file_names, output_dir, coord_keys=('xcoord', 'ycoord', 'spatial')):
    """
    Step 3: Compute BANKSY matrices for each slice.

    Changes:
    - Instead of returning a list of BANKSY objects, it writes them to 'output_dir' immediately.
    - Aggressively clears memory after each write.
    - Returns ONLY the concatenated original data (adata_concat).

    Args:
        batch_list: List of AnnData objects.
        file_names: List of strings (filenames).
        output_dir: Path to save the processed BANKSY .h5ad files.
        coord_keys: Tuple of spatial coordinate keys.

    Returns:
        adata_concat (AnnData): The concatenated original data.
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Starting BANKSY computation. Results will be saved to {output_dir}...")

    # Iterate through the list to process BANKSY
    for i, (adata, fname) in enumerate(zip(batch_list, file_names)):
        try:
            # 1. Add 'slice_labels' column (Filename) to the original data
            # Note: This modifies the object in batch_list in-place, which is needed for later concatenation
            adata.obs['slice_labels'] = fname

            # 2. Compute BANKSY matrix
            banksy_matrix = _compute_banksy(adata, coord_keys, k_geom=15, max_m=1, nbr_weight_decay="scaled_gaussian")

            # 3. Memory Optimization: Convert to CSR sparse matrix if dense
            if not scipy.sparse.issparse(banksy_matrix.X):
                banksy_matrix.X = scipy.sparse.csr_matrix(banksy_matrix.X)

            # 4. Construct new AnnData for BANKSY result
            banksy_adata = AnnData(banksy_matrix.X, obs=adata.obs, var=banksy_matrix.var, obsm=adata.obsm)
            banksy_adata.obs['slice_labels'] = f"{fname}_bk"

            # 5. Type Optimization: Ensure float32 to save memory
            if banksy_adata.X.dtype != 'float32':
                banksy_adata.X = banksy_adata.X.astype('float32')

            # 6. WRITE TO DISK IMMEDIATELY
            save_path = os.path.join(output_dir, f"{fname}_banksy.h5ad")
            banksy_adata.write_h5ad(save_path)
            print(f"Saved BANKSY: {save_path}")

            # 7. AGGRESSIVE MEMORY CLEANUP
            del banksy_adata
            del banksy_matrix
            gc.collect()

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # --- Concatenation Phase ---
    print("Concatenating original data...")

    # Use ad.concat to merge the batch_list.
    adata_concat = ad.concat(batch_list, label="slice_name", keys=file_names, index_unique=None)

    print(f"Concatenation complete. Shape: {adata_concat.shape}")

    # --- Final Memory Cleanup ---
    print("Cleaning up original batch list memory...")
    del batch_list
    gc.collect()

    return adata_concat