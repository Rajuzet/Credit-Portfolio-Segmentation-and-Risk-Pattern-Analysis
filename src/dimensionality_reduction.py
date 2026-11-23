import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'

PROCESSED_DIR = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\data\processed"
FIGURES_DIR = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\reports\figures"
MODELS_DIR = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\models"

def run_dimensionality_reduction():
    print("Step 5.1: Loading scaled features and sample data...")
    X_scaled = np.load(os.path.join(PROCESSED_DIR, "X_sample_scaled.npy"))
    df_sample = pd.read_parquet(os.path.join(PROCESSED_DIR, "cleaned_loans_sample.parquet"))
    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)
    feature_cols = meta["clustering_feature_cols"]
    print(f"Loaded scaled matrix shape: {X_scaled.shape}")

    # 1. Principal Component Analysis (PCA)
    print("Step 5.2: Fitting PCA across all components...")
    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # Save fitted PCA model
    pca_path = os.path.join(MODELS_DIR, "pca_model.joblib")
    joblib.dump(pca, pca_path)
    print(f"Saved PCA model to {pca_path}")

    # Save PCA coordinates (top 5 components)
    pca_coords_path = os.path.join(PROCESSED_DIR, "pca_coordinates.npy")
    np.save(pca_coords_path, X_pca[:, :5])

    # Scree Plot & Cumulative Variance
    print("Step 5.3: Generating PCA Scree Plot...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    components = np.arange(1, len(exp_var) + 1)
    ax1.bar(components, exp_var * 100, color='#3498db', alpha=0.7, label='Individual Variance Explained (%)')
    ax1.set_xlabel('Principal Component Number')
    ax1.set_ylabel('Individual Variance Explained (%)', color='#2980b9')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    ax1.set_xticks(components)

    ax2 = ax1.twinx()
    ax2.plot(components, cum_var * 100, color='#e74c3c', marker='o', linewidth=2, label='Cumulative Variance (%)')
    ax2.axhline(80, color='gray', linestyle='--', alpha=0.7, label='80% Variance Threshold')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    ax2.set_ylim(0, 105)

    plt.title('PCA Explained Variance & Scree Plot', fontsize=14, fontweight='bold')
    fig.tight_layout()
    scree_path = os.path.join(FIGURES_DIR, "06_pca_scree_plot.png")
    plt.savefig(scree_path, dpi=300)
    plt.close()
    print(f"Saved: {scree_path}")

    # Top PCA Loadings Interpretation
    loadings_df = pd.DataFrame(
        pca.components_[:3, :].T,
        index=feature_cols,
        columns=['PC1', 'PC2', 'PC3']
    )
    loadings_path = os.path.join(PROCESSED_DIR, "pca_loadings.csv")
    loadings_df.to_csv(loadings_path)
    print(f"Top Loadings for PC1 & PC2:\n{loadings_df.head(8)}")

    # 2. t-SNE (on 10,000 sample for speed & visual sharpness)
    print("Step 5.4: Computing t-SNE 2D embedding (10,000 points)...")
    tsne_sample_size = 10000
    np.random.seed(42)
    indices_tsne = np.random.choice(len(X_scaled), size=tsne_sample_size, replace=False)
    X_tsne_input = X_scaled[indices_tsne]
    y_tsne_default = df_sample.iloc[indices_tsne]['Default'].values

    tsne = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1000, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_tsne_input)
    np.save(os.path.join(PROCESSED_DIR, "tsne_coordinates.npy"), X_tsne)
    np.save(os.path.join(PROCESSED_DIR, "tsne_sample_indices.npy"), indices_tsne)
    print("t-SNE completed!")

    # 3. UMAP (Uniform Manifold Approximation & Projection)
    print("Step 5.5: Computing UMAP 2D embedding (15,000 points)...")
    umap_sample_size = 15000
    indices_umap = np.random.choice(len(X_scaled), size=umap_sample_size, replace=False)
    X_umap_input = X_scaled[indices_umap]
    y_umap_default = df_sample.iloc[indices_umap]['Default'].values

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
    X_umap = reducer.fit_transform(X_umap_input)
    np.save(os.path.join(PROCESSED_DIR, "umap_coordinates.npy"), X_umap)
    np.save(os.path.join(PROCESSED_DIR, "umap_sample_indices.npy"), indices_umap)
    print("UMAP completed!")

    # 4. Generate Comparative Projection Figure (PCA vs t-SNE vs UMAP)
    print("Step 5.6: Plotting Dimensionality Reduction Projections...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # PCA Projection
    scatter1 = axes[0].scatter(
        X_pca[indices_tsne, 0], X_pca[indices_tsne, 1],
        c=y_tsne_default, cmap='coolwarm', alpha=0.4, s=8
    )
    axes[0].set_title(f'PCA (PC1 vs PC2) - {exp_var[0]*100:.1f}% & {exp_var[1]*100:.1f}% Var', fontweight='bold')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    # t-SNE Projection
    scatter2 = axes[1].scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=y_tsne_default, cmap='coolwarm', alpha=0.4, s=8
    )
    axes[1].set_title('t-SNE 2D Manifold Projection', fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    # UMAP Projection
    scatter3 = axes[2].scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=y_umap_default, cmap='coolwarm', alpha=0.4, s=8
    )
    axes[2].set_title('UMAP 2D Manifold Projection', fontweight='bold')
    axes[2].set_xlabel('UMAP Dimension 1')
    axes[2].set_ylabel('UMAP Dimension 2')

    # Colorbar
    cbar = fig.colorbar(scatter3, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.03, pad=0.15)
    cbar.set_label('Borrower Outcome: 0 = Fully Paid (Blue), 1 = Defaulted (Red)', fontsize=12)

    proj_path = os.path.join(FIGURES_DIR, "07_dim_reduction_projections.png")
    plt.savefig(proj_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {proj_path}")

    # Summary JSON
    dim_summary = {
        "total_features": len(feature_cols),
        "pc1_variance": float(exp_var[0]),
        "pc2_variance": float(exp_var[1]),
        "pc3_variance": float(exp_var[2]),
        "cum_variance_5_components": float(cum_var[4]),
        "components_for_80_pct": int(np.argmax(cum_var >= 0.80) + 1),
        "tsne_samples": len(X_tsne),
        "umap_samples": len(X_umap)
    }
    with open(os.path.join(PROCESSED_DIR, "dimensionality_reduction_summary.json"), 'w') as f:
        json.dump(dim_summary, f, indent=4)

    print("Step 5: Dimensionality Reduction Completed Successfully!")

if __name__ == "__main__":
    run_dimensionality_reduction()
