# **MISTIC: Microenvironment-Guided Integration of Spatial Transcriptomic Information Across Slices**

![](https://raw.githubusercontent.com/ssbxf/images/main/figure1-01.png)

## Overview

**a** Workflow of microenvironment information extraction and multi-type data integration scope. MISTIC accommodates integration of multi-slice omics data from the same or distinct tissues.For each spatial transcriptomic slice, three core matrices are derived via the BANKSY algorithm: gene expression matrix, microenvironment expression matrix and microenvironment gradient matrix. **b** Construction of intra-slice adjacency matrices. Two distinct matrices are built for each slice: (i) Spatial distance matrix; (ii) Expression similarity matrix. Global intra-slice matrices are formed by block-diagonal concatenation of slice-specific matrices across all slices. **c** Construction of inter-slice adjacency matrix. Inter-slice mutual nearest neighbor (MNN) pairs are detected in global feature matrix (M). The final inter-slice adjacency matrix retains only high-confidence MNN pairs. **d** Hybrid graph neural network integration. Inputs include gene expression data and microenvironment data combined with adjacency matrices, outputting batch-corrected latent features. **e** Key biological applications of MISTIC. The integrated latent features support three core applications: (1) Batch effect removal (e.g., cross-platform, cross-developmental stage); (2) Condition-specific niche identification; (3) Microenvironmental interaction analysis.

## Installation

To ensure the Graph Neural Networks (GNNs) run correctly, PyTorch Geometric (PyG) and its dependencies must be installed strictly matching your PyTorch and CUDA versions.

* Python >= 3.8
* NVIDIA GPU (Recommended)

###### Create Environment

We recommend using Anaconda to manage the environment:

```bash
conda create -n mistic_env python==3.8
conda activate mistic_env
```

###### Install MISTIC

```
pip install MISTIC
```

###### Install PyTorch

Install a version of PyTorch compatible with your CUDA driver (**e.g., CUDA 12.1**)：

```
# Install PyTorch (GPU version)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

###### Install PyTorch Geometric (PyG) & Dependencies

Note: This is the most critical step. Do not simply use pip install torch-geometric. You must install the sparse libraries (torch_scatter, torch_sparse) that match your PyTorch version to enable GNN acceleration(**e.g., torch-2.1.0 and cu121**).

```Bash
# Install PyG Dependencies (Must match torch version)
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install PyTorch Geometric
pip install torch_geometric
```













