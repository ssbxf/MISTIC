import os
import sys

# ✅ 兼容写法：使用当前工作目录
# 假设你的 Notebook 就在项目根目录下 (即与 src 文件夹同级)
src_path = os.path.join(os.getcwd(), 'src')

if src_path not in sys.path:
    sys.path.append(src_path)

import MISTIC
INPUT_DIR = os.path.join('data', 'IVH')
OUTPUT_BANKSY_DIR = "./output/banksy_processed"

OUTPUT_FILE = os.path.join('output', 'mistic_result.h5ad')
# Preprocessing
N_TOP_GENES = 5000
# Graph Construction Parameters
SPATIAL_NEIGHBORS = 6  # k for spatial graph
SPATIAL_RADIUS = 150.0  # Radius for spatial graph
# Model Training Parameters
IS_HETEROGENEOUS = True  # Use GCN for output layer if True & dense MNN
NUM_EPOCHS = 500
CONTRASTIVE_WEIGHT = 0.05  # Coefficient for contrastive loss
LEARNING_RATE = 0.001
SEED = 42
print("==========================================")
print("       MISTIC Pipeline Started")
print("==========================================")
# ==========================================
# 2. Load and Preprocess
# ==========================================
print("\n[Step 1/5] Loading and Preprocessing Data...")
# Load .h5ad files and perform normalization
raw_adatas, filenames = MISTIC.load_and_preprocess(INPUT_DIR,n_top_genes=N_TOP_GENES)
# Align genes across all slices (intersection)
aligned_adatas = MISTIC.align_common_genes(raw_adatas)

# ==========================================
# 3. Generate BANKSY Features
# ==========================================
print("\n[Step 2/5] Running BANKSY and Preparing Matrices...")

# Compute Banksy matrices and concatenate original data
# Note: raw_adatas memory is cleared inside this function
adata_concat = MISTIC.get_banksy_results(aligned_adatas,filenames,output_dir = OUTPUT_BANKSY_DIR )
print("\n[Step 3/5] Constructing Multi-View Graphs...")
X_adata, X_expr, X_gra, adj_concat, spatial_adj_concat, avg_mnn = MISTIC.construct_graph(banksy_dir=OUTPUT_BANKSY_DIR)

print("\n[Step 4/5] Training MISTIC AutoEncoder...")

# Run the training loop
adata_result, model = MISTIC.run_training(
    adata_concat=adata_concat,
    X_adata=X_adata,
    X_expr=X_expr,
    X_gra=X_gra,
    adj_concat=adj_concat,
    spatial_adj_concat=spatial_adj_concat,
    mnn_avg_degree=avg_mnn,
    # Parameters
    is_heterogeneous=IS_HETEROGENEOUS,
    num_epochs=NUM_EPOCHS,
    contrastive_weight=CONTRASTIVE_WEIGHT,
    lr=LEARNING_RATE,
    seed=SEED
)

    # ==========================================
    # 6. Save Results
    # ==========================================
print(f"\n[Step 5/5] Saving results to {OUTPUT_FILE}...")

adata_result.write_h5ad(OUTPUT_FILE)