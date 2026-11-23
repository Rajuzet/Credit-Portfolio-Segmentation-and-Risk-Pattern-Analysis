import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'

PROCESSED_DIR = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\data\processed"
FIGURES_DIR = r"C:\Users\packi\.gemini\antigravity\scratch\credit_portfolio_segmentation\reports\figures"

# High-contrast distinguishable cluster colors
# Cluster 0: Vivid Rose Crimson
# Cluster 1: Bright Amber Gold
# Cluster 2: Electric Purple
# Cluster 3: Deep Emerald Green
CLUSTER_COLORS = ['#E11D48', '#F59E0B', '#8B5CF6', '#059669']
CLUSTER_LABELS = [
    'Cluster 0: Stretched Renters (Elevated Risk)',
    'Cluster 1: Homeowners (Moderate Risk)',
    'Cluster 2: Edge Segment (Highest Risk)',
    'Cluster 3: Prime Mortgage (Lowest Risk)'
]
cmap_clusters = ListedColormap(CLUSTER_COLORS)

def update_figures():
    print("Regenerating figures with distinguishable cluster color palette...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "segmented_loans_sample.parquet"))
    stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "cluster_summary_stats.csv"))

    # 1. Update 11_cluster_default_risk.png
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        stats_df['kmeans_cluster'].astype(str),
        stats_df['default_rate_pct'],
        color=CLUSTER_COLORS,
        edgecolor='#0f172a',
        linewidth=1.2,
        alpha=0.9
    )
    plt.axhline(y=df['Default'].mean() * 100, color='#0f172a', linestyle='--', linewidth=2, label=f'Portfolio Baseline ({df["Default"].mean()*100:.1f}%)')
    plt.title('Historical Default Rate by Discovered Borrower Segment', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Discovered Cluster ID', fontweight='bold')
    plt.ylabel('Historical Default Rate (%)', fontweight='bold')
    plt.ylim(0, max(stats_df['default_rate_pct']) * 1.25)
    plt.legend()
    for bar in bars:
        h = bar.get_height()
        plt.annotate(f'{h:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.tight_layout()
    risk_plot_path = os.path.join(FIGURES_DIR, "11_cluster_default_risk.png")
    plt.savefig(risk_plot_path, dpi=300)
    plt.close()
    print(f"Updated: {risk_plot_path}")

    # 2. Update 12_cluster_profiles.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.barplot(data=stats_df, x='kmeans_cluster', y='mean_income', ax=axes[0, 0], palette=CLUSTER_COLORS, hue='kmeans_cluster', legend=False, edgecolor='#0f172a')
    axes[0, 0].set_title('Mean Annual Income by Segment ($)', fontweight='bold')
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Annual Income ($)')

    sns.barplot(data=stats_df, x='kmeans_cluster', y='mean_fico', ax=axes[0, 1], palette=CLUSTER_COLORS, hue='kmeans_cluster', legend=False, edgecolor='#0f172a')
    axes[0, 1].set_title('Mean FICO Credit Score by Segment', fontweight='bold')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('FICO Score')
    axes[0, 1].set_ylim(640, 750)

    sns.barplot(data=stats_df, x='kmeans_cluster', y='mean_loan', ax=axes[1, 0], palette=CLUSTER_COLORS, hue='kmeans_cluster', legend=False, edgecolor='#0f172a')
    axes[1, 0].set_title('Mean Loan Amount by Segment ($)', fontweight='bold')
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Loan Amount ($)')

    sns.barplot(data=stats_df, x='kmeans_cluster', y='mean_dti', ax=axes[1, 1], palette=CLUSTER_COLORS, hue='kmeans_cluster', legend=False, edgecolor='#0f172a')
    axes[1, 1].set_title('Mean Debt-to-Income (DTI) by Segment (%)', fontweight='bold')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('DTI (%)')

    plt.tight_layout()
    profile_plot_path = os.path.join(FIGURES_DIR, "12_cluster_profiles.png")
    plt.savefig(profile_plot_path, dpi=300)
    plt.close()
    print(f"Updated: {profile_plot_path}")

    # 3. New 14_clusters_manifold_projections.png (PCA, t-SNE, UMAP colored by clusters)
    pca_coords = np.load(os.path.join(PROCESSED_DIR, "pca_coordinates.npy"))
    tsne_coords = np.load(os.path.join(PROCESSED_DIR, "tsne_coordinates.npy"))
    tsne_idx = np.load(os.path.join(PROCESSED_DIR, "tsne_sample_indices.npy"))
    umap_coords = np.load(os.path.join(PROCESSED_DIR, "umap_coordinates.npy"))
    umap_idx = np.load(os.path.join(PROCESSED_DIR, "umap_sample_indices.npy"))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # PCA
    pca_clusters = df.iloc[:len(pca_coords)]['kmeans_cluster'].values
    axes[0].scatter(pca_coords[tsne_idx, 0], pca_coords[tsne_idx, 1], c=pca_clusters[tsne_idx], cmap=cmap_clusters, alpha=0.6, s=10)
    axes[0].set_title('PCA Projection (Colored by Cluster)', fontweight='bold', fontsize=13)
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    # t-SNE
    tsne_clusters = df.iloc[tsne_idx]['kmeans_cluster'].values
    axes[1].scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=tsne_clusters, cmap=cmap_clusters, alpha=0.6, s=10)
    axes[1].set_title('t-SNE 2D Manifold (Colored by Cluster)', fontweight='bold', fontsize=13)
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    # UMAP
    umap_clusters = df.iloc[umap_idx]['kmeans_cluster'].values
    sc = axes[2].scatter(umap_coords[:, 0], umap_coords[:, 1], c=umap_clusters, cmap=cmap_clusters, alpha=0.6, s=10)
    axes[2].set_title('UMAP 2D Manifold (Colored by Cluster)', fontweight='bold', fontsize=13)
    axes[2].set_xlabel('UMAP Dimension 1')
    axes[2].set_ylabel('UMAP Dimension 2')

    # Custom Legend for all clusters
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CLUSTER_COLORS[i], markersize=10, label=f"Cluster {i}") for i in range(4)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12, frameon=True)

    plt.tight_layout()
    manifold_clusters_path = os.path.join(FIGURES_DIR, "14_clusters_manifold_projections.png")
    plt.savefig(manifold_clusters_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {manifold_clusters_path}")
    print("All distinguishable figures generated successfully!")

if __name__ == "__main__":
    update_figures()
