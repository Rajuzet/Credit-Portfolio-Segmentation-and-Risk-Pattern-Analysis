# 💳 Credit Portfolio Segmentation and Risk Pattern Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-red.svg)
![Plotly](https://img.shields.io/badge/Plotly-7.1-purple.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

An end-to-end unsupervised machine learning and loan analytics project developed for **Banking & Credit Risk Management**. This project groups borrowers with similar financial, credit, and employment characteristics, projects high-dimensional customer spaces via **PCA, t-SNE, and UMAP**, compares **K-Means, Hierarchical Clustering, and DBSCAN**, uncovers historical default patterns without target leakage, flags suspicious credit anomalies, and delivers an interactive **Streamlit Dashboard**.

---

## 📌 Problem Statement
Financial institutions manage credit portfolios with diverse borrower profiles across income levels, credit scores, debt-to-income (DTI) burdens, and borrowing intents. Traditional credit risk systems often focus solely on supervised scoring. This project applies **unsupervised machine learning** to:
1. Discover distinct borrower archetypes based on financial, credit, and employment characteristics.
2. Evaluate and compare clustering algorithms (**K-Means, Agglomerative Hierarchical, and DBSCAN**).
3. Visualize customer topologies using **PCA, t-SNE, and UMAP**.
4. **Compare historical default rates across clusters** without using the `Default` status during cluster formation.
5. Identify **financial anomalies and irregular loan requests** using density-based outlier detection.
6. Provide an **interactive Streamlit application** for loan portfolio exploration and real-time applicant segment prediction.

---

## 🏛️ Dataset Overview
- **Source:** Lending Club Loan Dataset for Granting Models ([Zenodo: 10.5281/zenodo.11295916](https://doi.org/10.5281/zenodo.11295916))
- **Records:** **1,347,681** loans
- **Features:** Annual Income (`revenue`), Debt-to-Income (`dti_n`), Loan Amount (`loan_amnt`), FICO Score (`fico_n`), Employment Length (`emp_length`), Employment Experience (`experience_c`), Loan Purpose (`purpose`), Home Ownership (`home_ownership_n`), Issue Date (`issue_d`), and Default Status (`Default`).
- **Baseline Default Rate:** **19.98%** (80.02% Fully Paid vs. 19.98% Defaulted).

---

## 🛠️ Tech Stack & Skills
- **Core Languages & Libraries:** Python 3.11, Pandas, NumPy, SciPy
- **Data Preprocessing & Scaling:** Scikit-Learn `StandardScaler`, Log1p transformation, Outlier capping
- **Unsupervised Learning & Clustering:** K-Means, Agglomerative Hierarchical Clustering (Ward Linkage), DBSCAN, Isolation Forest
- **Dimensionality Reduction & Manifolds:** PCA (Principal Component Analysis), t-SNE, UMAP
- **Metrics & Validation:** Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index, Elbow Method, K-Distance Graph
- **Visualization:** Matplotlib, Seaborn, Plotly Express & Graph Objects
- **Application & Deployment:** Streamlit Interactive Web Application
- **Version Control:** Git, GitHub

---

## 📂 Project Repository Structure
```
credit_portfolio_segmentation/
├── app/
│   └── app.py                              # Full-featured Streamlit Web Dashboard
├── data/
│   ├── raw/                                # Raw dataset storage (Zenodo source)
│   └── processed/                          # Scaled matrices, parquet files, and cluster stats
├── models/
│   ├── scaler.joblib                       # Fitted StandardScaler
│   ├── pca_model.joblib                    # Fitted PCA model
│   ├── kmeans_optimal_model.joblib         # Optimal K-Means model (K=4)
│   └── isolation_forest.joblib             # Fitted Isolation Forest anomaly detector
├── reports/
│   ├── eda_report.md                       # Comprehensive Exploratory Data Analysis Report
│   ├── clustering_comparison.md            # In-depth clustering algorithms comparison
│   ├── final_report.md                     # Executive Capstone Project Report
│   └── figures/                            # High-resolution analysis charts
│       ├── 01_feature_distributions.png
│       ├── 02_correlation_heatmap.png
│       ├── 03_default_by_purpose.png
│       ├── 04_risk_tiers_home_ownership.png
│       ├── 05_loan_vs_income_risk.png
│       ├── 06_pca_scree_plot.png
│       ├── 07_dim_reduction_projections.png
│       ├── 08_kmeans_elbow_silhouette.png
│       ├── 09_hierarchical_dendrogram.png
│       ├── 10_dbscan_k_distance.png
│       ├── 11_cluster_default_risk.png
│       ├── 12_cluster_profiles.png
│       └── 13_anomaly_profiles.png
├── src/
│   ├── download_data.py                    # Dataset downloader from Zenodo
│   ├── inspect_data.py                     # Initial dataset inspection script
│   ├── preprocess.py                       # Preprocessing, encoding & scaling pipeline
│   ├── eda.py                              # Exploratory data analysis & figure generator
│   ├── dimensionality_reduction.py         # PCA, t-SNE & UMAP execution pipeline
│   └── clustering_and_analysis.py          # K-Means, Hierarchical, DBSCAN, profiling & risk
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation
```

---

## 📊 Key Findings & Discovered Borrower Personas

| Segment ID | Persona Name | Share (%) | Median Income | Median Loan | Mean FICO | Mean DTI | Historical Default Rate | Risk Classification |
|---|---|---|---|---|---|---|---|---|
| **Cluster 3** | **Tier 1: Prime Mortgage Borrowers** | **49.66%** | **\$75,000** | \$15,000 | **701.3** | 18.2% | **17.14%** | **Lowest Risk (Prime)** |
| **Cluster 1** | **Tier 2: Established Outright Owners** | **10.61%** | **\$60,000** | \$12,000 | **700.4** | 18.6% | **21.47%** | **Moderate Risk** |
| **Cluster 0** | **Tier 3: Stretched High-DTI Renters** | **39.64%** | **\$56,000** | \$10,375 | **694.0** | 18.0% | **23.13%** | **Elevated Risk** |
| **Cluster 2** | **Tier 4: Non-Standard Edge Borrowers** | **0.08%** | **\$70,000** | \$6,000 | **705.9** | 18.6% | **24.39%** | **Highest Risk** |

> **Validation Note:** The `Default` label was **never seen** by the clustering algorithms. The natural emergence of an 800+ bps spread between Tier 1 (17.14%) and Tier 4 (24.39%) confirms that unsupervised segmentation captures fundamental creditworthiness patterns!

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Rajuzet/Credit-Portfolio-Segmentation-and-Risk-Pattern-Analysis.git
cd Credit-Portfolio-Segmentation-and-Risk-Pattern-Analysis
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Processing & Modeling Pipeline (Optional - Pre-Generated)
```bash
python src/preprocess.py
python src/eda.py
python src/dimensionality_reduction.py
python src/clustering_and_analysis.py
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Modules
1. **🏛️ Portfolio Overview & KPIs:** High-level executive statistics, total portfolio size, and segment distributions.
2. **📊 Exploratory Data Analysis:** Interactive distributions, boxplots, correlation heatmaps, and categorical risk breakdowns.
3. **🌀 Dimensionality Reduction:** Interactive Plotly 2D/3D scatter plots for PCA, t-SNE, and UMAP with hover inspection.
4. **👥 Borrower Segmentation & Clusters:** Detailed customer persona cards and multi-metric comparisons.
5. **⚠️ Historical Default Pattern Analysis:** Empirical default risk comparisons against portfolio baseline.
6. **🚨 Anomaly & Outlier Detection:** Deep-dive into suspicious and non-conforming loan requests flagged by DBSCAN and Isolation Forest.
7. **🎯 Real-Time Borrower Segment Predictor:** What-if simulator allowing underwriters to input applicant parameters and receive instant cluster assignment, default probability, and policy recommendation.

---

## 📈 Clustering Algorithm Performance Comparison

| Algorithm | Clusters | Silhouette Score | Calinski-Harabasz | Davies-Bouldin | Anomaly Detection | Business Recommendation |
|---|---|---|---|---|---|---|
| **K-Means (K=4)** | **4** | **0.1848** | **741.3** | **1.6415** | None (Partitioning) | **Recommended for Operational Underwriting & Tiering** |
| **Hierarchical (Ward)** | 4 | 0.1008 | 686.2 | 2.1506 | None (Agglomerative) | Confirms natural 4-cluster dendrogram branch structure |
| **DBSCAN** | 23 | 0.0820 | 312.4 | 2.4510 | **2.39% Flagged as Noise** | **Recommended for Fraud & Edge Anomaly Filtering** |

---

## 📄 License & Credits
Developed as part of the **GUVI | HCL Capstone Project Evaluation** in Data Science & Machine Learning.
Dataset courtesy of the Lending Club open research repository on Zenodo.
