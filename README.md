MISTIC: Multi-view Integrated Spatial Transcriptomics with Inter-slice Connectivity
📖 Overview
MISTIC is a deep learning framework designed for the robust integration of multiple spatial transcriptomics slices. Unlike conventional approaches that rely solely on cell-autonomous gene expression or rigid spatial alignment, MISTIC explicitly incorporates microenvironmental metrics—including neighborhood composition and signaling gradients—as primary integration features.

This strategy is empowered by a specialized multi-view Graph Neural Network (GNN) architecture, which utilizes attention mechanisms to dynamically decouple biological signals from technical artifacts. MISTIC effectively resolves the inherent trade-off between batch correction and biological conservation, ensuring the precise alignment of datasets while preventing the over-smoothing of fine-grained spatial domains and preserving rare, sample-specific niches.

🌟 Key Features
Dual-Dimensional Integration: Fuses intrinsic gene expression with extrinsic microenvironmental metrics.

Scalable Graph Construction: Optimized pipeline for processing large-scale multi-slice datasets (e.g., human DLPFC, mouse embryos).

Superior Batch Correction: Validated to outperform state-of-the-art methods (STAligner, GraphST, etc.) in batch mixing metrics (iLISI, kBET).

Structure Preservation: High fidelity in reconstructing continuous tissue architectures and identifying rare cell types.

🛠️ Installation (Crucial)
To ensure the Graph Neural Networks (GNNs) run correctly, PyTorch Geometric (PyG) and its dependencies must be installed strictly matching your PyTorch and CUDA versions.

1. Create Environment
We recommend using Anaconda to manage the environment:


conda create -n mistic_env python=3.8
conda activate mistic_env
2. Install PyTorch
Install a version of PyTorch compatible with your CUDA driver (e.g., CUDA 12.1):


# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
3. Install PyTorch Geometric (PyG) & Dependencies
Note: This is the most critical step. Do not simply use pip install torch-geometric. You must install the sparse libraries (torch_scatter, torch_sparse) that match your PyTorch version to enable GNN acceleration.

Run the following commands (replace cu121 and 2.1.0 with your actual CUDA/Torch versions if different):

Bash

# Step 3.1: Install PyG Core
pip install torch_geometric

# Step 3.2: Install Optimized Scatter/Sparse Kernels
# IMPORTANT: The URL must match your PyTorch version (e.g., torch-2.1.0) and CUDA version (e.g., cu121)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
4. Install Other Requirements
Install the remaining bioinformatics and utility libraries:



pip install scanpy anndata pandas numpy scipy scikit-learn annoy matplotlib


📊 Benchmarks
MISTIC has been validated across diverse biological contexts, demonstrating superior performance over six state-of-the-art benchmarks (including STAligner, Harmony, and Seurat).

Human DLPFC: Achieved ARI > 0.6, successfully distinguishing cortical layers while mixing technical batches.

Axolotl Regeneration: Preserved continuous developmental trajectories in dynamic regeneration datasets.

Mouse IVH Model: Identified rare immune niches in a heterogeneous pathological microenvironment.

(Refer to docs/figures/figure2.pdf for detailed comparison plots)


# MISTIC

**M**ulti-slice **I**ntegration **S**patially for **T**ranscriptomics **I**ntegration and **C**lustering.

MISTIC is a deep learning framework for integrating multiple spatial transcriptomics slices, effectively handling batch effects while preserving spatial domains.

## 🛠 Installation

To ensure GPU acceleration works correctly, MISTIC requires manual installation of PyTorch and PyTorch Geometric dependencies.

### Prerequisites
* Python >= 3.8
* NVIDIA GPU (Recommended)

### Step 1: Install PyTorch & PyG Dependencies

We strongly recommend installing **PyTorch 2.1.0** with **CUDA 12.1** support for the best compatibility.

**Option A: For CUDA 12.1 (Recommended)**

Run the following commands in your terminal:

```bash
# 1. Install PyTorch (GPU version)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 2. Install PyG Dependencies (Must match torch version)
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.1.0+cu121.html](https://data.pyg.org/whl/torch-2.1.0+cu121.html)

# 3. Install PyTorch Geometric
pip install torch_geometric
