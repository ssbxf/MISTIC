import random
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_sparse import SparseTensor


_rng = np.random.RandomState(42)


def setup_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    global _rng
    _rng = np.random.RandomState(seed)


def extract_similar_pairs(adj_matrix, num_pairs=10000):
    """
    Extract positive pairs (connected nodes) from the adjacency matrix.
    """
    # Convert to coordinate format (rows, cols)
    rows, cols = adj_matrix.nonzero()

    # Filter out self-loops if any
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]

    total_edges = len(rows)
    indices = np.arange(total_edges)

    # Sample if we have more edges than requested
    if total_edges > num_pairs:
        sampled_indices = np.random.choice(indices, num_pairs, replace=False)
        rows = rows[sampled_indices]
        cols = cols[sampled_indices]

    pairs = np.vstack((rows, cols)).T
    labels = np.ones(len(pairs))

    return pairs, labels


def generate_negative_pairs(adj_matrix, num_pairs=10000):
    """
    Generate negative pairs (disconnected nodes) via random sampling.
    """
    n = adj_matrix.shape[0]
    pairs = []
    labels = []

    while len(pairs) < num_pairs:
        i = _rng.randint(0, n)
        j = _rng.randint(0, n)

        if i != j and adj_matrix[i, j] == 0:
            pairs.append((i, j))
            labels.append(0)

    return np.array(pairs), np.array(labels)


def scipy_to_sparse_tensor(scipy_adj, device):
    """
    Convert Scipy sparse matrix to PyTorch Geometric SparseTensor.
    """
    coo = scipy_adj.tocoo()

    row = torch.from_numpy(coo.row).long()
    col = torch.from_numpy(coo.col).long()

    num_nodes = scipy_adj.shape[0]

    adj_t = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    adj_t = adj_t.to_symmetric()

    return adj_t.to(device)


def _to_dense_tensor(data_obj):
    """
    Internal Helper: Safely convert Anndata or Matrix (Sparse or Dense) to Float Tensor.
    """
    # 1. Extract raw data if it's an AnnData object
    mat = data_obj.X if hasattr(data_obj, 'X') else data_obj

    # 2. Check if it is a scipy sparse matrix and convert to dense
    if sp.issparse(mat):
        mat = mat.toarray()

    # 3. Convert to Torch Tensor
    return torch.tensor(mat, dtype=torch.float)


def prepare_data(X_adata, X_expr, X_gra, adj_concat, spatial_adj_concat, device):
    """
    Prepare all data inputs for the MISTIC model.
    """
    # 1. Feature Matrices (Safe conversion using helper)
    X1 = _to_dense_tensor(X_adata)
    X2 = _to_dense_tensor(X_expr)
    X3 = _to_dense_tensor(X_gra)

    # 2. Graph Structures (SparseTensors)
    exp_edge_matrix = scipy_to_sparse_tensor(adj_concat, device)
    spatial_edge_matrix = scipy_to_sparse_tensor(spatial_adj_concat, device)

    # 3. Encapsulate in Data object
    data = Data(
        x1=X1.to(device),
        x2=X2.to(device),
        x3=X3.to(device),
        exp_edge_index=exp_edge_matrix,
        spatial_edge_index=spatial_edge_matrix
    ).to(device)

    return data