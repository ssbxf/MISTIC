import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .networks import AutoEncoder
from .contrastive_loss import ContrastiveLoss
from .utils import setup_seed, prepare_data, extract_similar_pairs, generate_negative_pairs


def run_training(
        adata_concat,
        X_adata,
        X_expr,
        X_gra,
        adj_concat,
        spatial_adj_concat,
        mnn_avg_degree,
        is_heterogeneous=True,
        num_epochs=500,
        lr=0.001,
        contrastive_weight=0.1,
        device=None,
        seed=42
):
    """
    Main function to train the MISTIC model.
    """

    # 1. Setup Environment
    setup_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Running training on: {device}")
    print(
        f"Hyperparameters -> Heterogeneous: {is_heterogeneous}, Epochs: {num_epochs}, Contrastive Weight: {contrastive_weight}")

    # 2. Prepare Data
    print("Preparing tensors and graphs...")
    data = prepare_data(X_adata, X_expr, X_gra, adj_concat, spatial_adj_concat, device)

    # 3. Prepare Contrastive Pairs (Static sampling)
    print("Generating contrastive learning pairs...")
    similar_pairs, similar_labels = extract_similar_pairs(adj_concat, num_pairs=15000)
    negative_pairs, negative_labels = generate_negative_pairs(adj_concat, num_pairs=15000)

    all_pairs = np.vstack((similar_pairs, negative_pairs))
    all_labels = np.concatenate((similar_labels, negative_labels))

    pairs_tensor = torch.tensor(all_pairs, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float).to(device)

    # 4. Initialize Model
    in_channels = data.x1.shape[1]
    hidden_channels = 500
    out_channels = 30

    print(f"Initializing AutoEncoder (Heterogeneous={is_heterogeneous}, MNN_Avg_Deg={mnn_avg_degree:.2f})...")
    model = AutoEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        is_heterogeneous=is_heterogeneous,
        mnn_avg_degree=mnn_avg_degree
    ).to(device)

    # 5. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_criterion = torch.nn.MSELoss()
    contrastive_loss_func = ContrastiveLoss(margin=1.0)

    # 6. Training Loop (With Progress Bar)
    print(f"Starting training for {num_epochs} epochs...")
    model.train()

    # 使用 tqdm 包装 range
    loop = tqdm(range(num_epochs), desc="Training MISTIC", unit="epoch")

    for epoch in loop:
        optimizer.zero_grad()

        # Forward pass
        z, x1_rec, x2_rec, x3_rec = model(
            data.x1,
            data.exp_edge_index,
            data.spatial_edge_index,
            data.x2,
            data.x3
        )

        # Reconstruction Loss
        loss_x1 = mse_criterion(x1_rec, data.x1)
        loss_x2 = mse_criterion(x2_rec, data.x2)
        loss_x3 = mse_criterion(x3_rec, data.x3)
        loss_rec = loss_x1 + loss_x2 + loss_x3

        # Contrastive Loss
        z1 = z[pairs_tensor[:, 0]]
        z2 = z[pairs_tensor[:, 1]]
        loss_con = contrastive_loss_func(z1, z2, labels_tensor)

        # Total Loss
        loss = loss_rec + contrastive_weight * loss_con

        loss.backward()
        optimizer.step()

        # 更新进度条右侧的显示信息 (实时显示 Loss)
        loop.set_postfix(
            Total_Loss=f"{loss.item():.4f}",
            Rec=f"{loss_rec.item():.4f}",
            Con=f"{loss_con.item():.4f}"
        )

    # 7. Inference & Save
    print("Training finished. Extracting embeddings...")
    model.eval()
    with torch.no_grad():
        z, _, _, _ = model(
            data.x1,
            data.exp_edge_index,
            data.spatial_edge_index,
            data.x2,
            data.x3
        )

    z_cpu = z.cpu().numpy()

    if adata_concat is not None:
        adata_concat.obsm['MISTIC'] = z_cpu
        print("Embedding saved to adata_concat.obsm['MISTIC']")

    return adata_concat, model
