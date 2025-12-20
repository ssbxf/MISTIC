import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class AttentionLayer(nn.Module):
    """
    Attention Layer to fuse two embeddings (e.g., Expression and Spatial) adaptively.
    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Learnable parameters for attention mechanism
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for stability
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        """
        Args:
            emb1: Tensor of shape (N, in_feat) - e.g., Expression embedding
            emb2: Tensor of shape (N, in_feat) - e.g., Spatial embedding
        Returns:
            emb_combined: Fused embedding (N, out_feat)
            alpha: Attention weights
        """
        emb = []
        # Ensure input shapes are (N, 1, F) before concatenation
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))

        # Stack embeddings: (N, 2, F)
        self.emb = torch.cat(emb, dim=1)

        # Compute attention scores
        # V = tanh(XW)
        self.v = torch.tanh(torch.matmul(self.emb, self.w_omega))
        # VU = V * U
        self.vu = torch.matmul(self.v, self.u_omega)

        # Softmax to get attention weights alpha: (N, 2)
        # Using squeeze to remove the last dimension (1) before softmax
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6, dim=1)

        # Weighted sum: (N, F, 2) * (N, 2, 1) -> (N, F, 1)
        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha


class MLPDecoder(nn.Module):
    """
    Simple Multi-Layer Perceptron Decoder to reconstruct original features.
    """

    def __init__(self, in_channels, out_channels):
        super(MLPDecoder, self).__init__()
        self.fc1 = Linear(in_channels, 128)
        self.fc2 = Linear(128, out_channels)

    def forward(self, z):
        z = self.fc1(z)
        z = F.relu(z)
        z = self.fc2(z)
        return z


class AutoEncoder(nn.Module):
    """
    MISTIC AutoEncoder Model.
    Integrates GAT (for intra-slice), SAGE (for features), and Attention Fusion.
    Adaptive output layer based on graph heterogeneity and MNN density.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, is_heterogeneous=True, mnn_avg_degree=None):
        super(AutoEncoder, self).__init__()

        # --- Encoders for View 1 (e.g., High Variable Genes) ---
        # Branch A: Expression Graph (GAT)
        self.gat_exp_1 = GATConv(in_channels, hidden_channels)
        self.gat_exp_2 = GATConv(hidden_channels, hidden_channels)

        # Branch B: Spatial Graph (GAT)
        self.gat_spatial_1 = GATConv(in_channels, hidden_channels)
        self.gat_spatial_2 = GATConv(hidden_channels, hidden_channels)

        # Fusion: Weighted Attention
        self.attention_fusion1 = AttentionLayer(hidden_channels, hidden_channels)
        self.linear_fusion = Linear(hidden_channels, out_channels * 2)  # Renamed from self.Linear to follow PEP8

        # --- Encoders for View 2 & 3 (e.g., Other gene sets) ---
        # Using SAGEConv (mean aggregation) as efficient feature extractors
        self.fc_x2_1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.fc_x2_2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')

        self.fc_x3_1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.fc_x3_2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')

        # Dimensionality reduction for concatenated views 2 and 3
        self.reduce_dim_x23 = Linear(hidden_channels * 2, out_channels)

        # --- Adaptive Bottleneck / Integration Layer ---
        self.is_heterogeneous = is_heterogeneous
        self.mnn_avg_degree = mnn_avg_degree

        # Option 1: Fully Connected (for Homogeneous or Sparse MNN graphs)
        self.fc_z = Linear(out_channels * 3, out_channels)
        # Option 2: GCN (for Heterogeneous and Dense MNN graphs)
        # Input dim is out_channels * 3 because we concat [X_z(2*out) + x23(1*out)]
        self.gcn_exp = GCNConv(out_channels * 3, out_channels)

        # --- Decoders ---
        self.decoder_x1 = MLPDecoder(out_channels, in_channels)
        self.decoder_x2 = MLPDecoder(out_channels, in_channels)
        self.decoder_x3 = MLPDecoder(out_channels, in_channels)

    def forward(self, x1, exp_edge_index, spatial_edge_index, x2, x3):
        """
        Args:
            x1: Features for View 1 (e.g., Top HVGs)
            exp_edge_index: Adjacency matrix for Expression Graph
            spatial_edge_index: Adjacency matrix for Spatial Graph
            x2: Features for View 2
            x3: Features for View 3
        """

        # --- 1. Encode View 1 (Dual Graph + Attention) ---
        # Expression Graph Path
        exp_z = self.gat_exp_1(x1, exp_edge_index)
        exp_z = self.gat_exp_2(exp_z, exp_edge_index)

        # Spatial Graph Path
        spatial_z = self.gat_spatial_1(x1, spatial_edge_index)
        spatial_z = self.gat_spatial_2(spatial_z, spatial_edge_index)

        # Fusion
        X_z, _ = self.attention_fusion1(exp_z, spatial_z)
        X_z = self.linear_fusion(X_z)  # Result size: (N, out_channels * 2)

        # --- 2. Encode View 2 & 3 (GraphSAGE) ---
        # Use exp_edge_index for SAGE aggregation (assuming expression neighbors define local context)

        # View 2
        x2_z = self.fc_x2_1(x2, exp_edge_index)
        x2_z = self.fc_x2_2(x2_z, exp_edge_index)

        # View 3
        x3_z = self.fc_x3_1(x3, exp_edge_index)
        x3_z = self.fc_x3_2(x3_z, exp_edge_index)

        # Combine View 2 and 3
        x23_z = torch.cat([x2_z, x3_z], dim=1)
        x23_z_reduced = self.reduce_dim_x23(x23_z)  # Result size: (N, out_channels)

        # --- 3. Global Integration ---
        # Concatenate fused X1 with reduced X2/X3
        # Shape: (N, out_channels*2 + out_channels) -> (N, out_channels*3)
        z_concat = torch.cat([X_z, x23_z_reduced], dim=1)

        # Adaptive Processing based on Graph Topology
        # Logic: If the graph is heterogeneous and has sufficient cross-slice connections (MNN degree > 3),
        # use GCN to smooth embedding across slices. Otherwise, use simple Linear projection.
        if self.is_heterogeneous and self.mnn_avg_degree is not None and self.mnn_avg_degree > 3:
            # Use Spatial Edges for GCN smoothing (assuming spatial continuity is key here)
            # You might want to check if this should be exp_edge_index or spatial_edge_index
            z = self.gcn_exp(z_concat, spatial_edge_index)
        else:
            z = self.fc_z(z_concat)

        # --- 4. Decode ---
        x1_reconstructed = self.decoder_x1(z)
        x2_reconstructed = self.decoder_x2(z)
        x3_reconstructed = self.decoder_x3(z)

        return z, x1_reconstructed, x2_reconstructed, x3_reconstructed