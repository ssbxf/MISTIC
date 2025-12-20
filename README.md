**MISTIC: Microenvironment-Guided Integration of Spatial Transcriptomic Information Across Slices**

![](https://raw.githubusercontent.com/ssbxf/images/main/figure1.png)📖 Overview
MISTIC is a deep learning framework designed for the robust integration of multiple spatial transcriptomics slices. Unlike conventional approaches that rely solely on cell-autonomous gene expression or rigid spatial alignment, MISTIC explicitly incorporates microenvironmental metrics—including neighborhood composition and signaling gradients—as primary integration features.

This strategy is empowered by a specialized multi-view Graph Neural Network (GNN) architecture, which utilizes attention mechanisms to dynamically decouple biological signals from technical artifacts. MISTIC effectively resolves the inherent trade-off between batch correction and biological conservation, ensuring the precise alignment of datasets while preventing the over-smoothing of fine-grained spatial domains and preserving rare, sample-specific niches.



🌟 Key Features
Dual-Dimensional Integration: Fuses intrinsic gene expression with extrinsic microenvironmental metrics.

Scalable Graph Construction: Optimized pipeline for processing large-scale multi-slice datasets (e.g., human DLPFC, mouse embryos).

Superior Batch Correction: Validated to outperform state-of-the-art methods (STAligner, GraphST, etc.) in batch mixing metrics (iLISI, kBET).

Structure Preservation: High fidelity in reconstructing continuous tissue architectures and identifying rare cell types.

## 🛠 Installation

To ensure the Graph Neural Networks (GNNs) run correctly, PyTorch Geometric (PyG) and its dependencies must be installed strictly matching your PyTorch and CUDA versions.

### Prerequisites

* Python >= 3.8
* NVIDIA GPU (Recommended)

1. Create Environment

We recommend using Anaconda to manage the environment:

```bash
conda create -n mistic_env python=3.8
conda activate mistic_env
```

2. Install MISTIC

```
pip install MISTIC
```

3. Install a version of PyTorch compatible with your CUDA driver (e.g., CUDA 12.1):


#### Example for CUDA 12.1
```
# 1. Install PyTorch (GPU version)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```



4. Install PyTorch Geometric (PyG) & Dependencies(e.g., torch-2.1.0 and cu121)
   Note: This is the most critical step. Do not simply use pip install torch-geometric. You must install the sparse libraries (torch_scatter, torch_sparse) that match your PyTorch version to enable GNN acceleration.

```Bash
# 2. Install PyG Dependencies (Must match torch version)
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.1.0+cu121.html](https://data.pyg.org/whl/torch-2.1.0+cu121.html)

# 3. Install PyTorch Geometric
pip install torch_geometric
```













