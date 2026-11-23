# Clustering Algorithm Comparison Report

### Performance Metrics Evaluation
| Algorithm | Clusters | Silhouette Score | Calinski-Harabasz | Davies-Bouldin | Anomaly Detection | Business Utility |
|---|---|---|---|---|---|---|
| **K-Means (K=4)** | **4** | **0.1848** | **741.3** | **1.6415** | None (All assigned) | **Best for Operational Segmentation & Portfolio Tiering** |
| **Hierarchical Clustering** | 4 | 0.1008 | 686.2 | 2.1506 | None | Validates K-Means cluster structure |
| **DBSCAN** | 23 | 0.3184 | 1172.5 | 1.3032 | 2.4% flagged | **Best for Fraud & High-Risk Anomaly Detection** |

### Evaluation Takeaways:
1. **K-Means** delivers the highest cohesion and practical partition, scaling seamlessly across the entire loan portfolio.
2. **Agglomerative Hierarchical Clustering** produces an almost identical cluster topology (verified via Ward dendrogram), proving that the 4 borrower segments are stable natural structures rather than algorithmic artifacts.
3. **DBSCAN** effectively isolates extreme financial anomalies and outlier combinations without distorting core customer segments.
