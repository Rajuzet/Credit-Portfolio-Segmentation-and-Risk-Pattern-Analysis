# Final Capstone Project Report
# Credit Portfolio Segmentation and Risk Pattern Analysis

**Domain:** Finance & Banking — Loan Analytics & Credit Risk Analysis  
**Repository:** [Credit-Portfolio-Segmentation-and-Risk-Pattern-Analysis](https://github.com/Rajuzet/Credit-Portfolio-Segmentation-and-Risk-Pattern-Analysis)  
**Dataset:** Lending Club Loan Dataset for Granting Models (Zenodo: 10.5281/zenodo.11295916)

---

## 1. Executive Summary
Financial institutions deal with millions of borrowers possessing varying financial strengths, debt obligations, employment tenures, and credit scores. Conventional credit assessment often relies on supervised binary classification; however, unsupervised learning offers a powerful lens to **discover organic borrower archetypes**, **uncover hidden risk pockets**, and **detect non-conforming financial anomalies** without outcome bias.

In this project, an unsupervised machine learning architecture was developed on **1,347,681** historical loan records. By strictly excluding the `Default` status from the feature space during clustering, borrower segments were formed purely on financial, credit, and employment characteristics. When cross-referenced against historical repayment records, the discovered segments naturally revealed a **strong risk gradient**—ranging from a prime segment with a default rate of **17.14%** to higher-risk segments exceeding **23.13%**, with pure financial clusters separating risk down to **9.50%**.

---

## 2. Dataset Architecture & Preprocessing
The dataset contains **1,347,681 records across 15 attributes**:
- **Financial Profile:** Annual Income (`revenue`), Debt-to-Income Ratio (`dti_n`), Requested Loan Amount (`loan_amnt`).
- **Credit Profile:** FICO Credit Score (`fico_n`), Employment Experience (`experience_c`), Employment Length (`emp_length`).
- **Loan Purpose & Collateral:** Loan Purpose (`purpose`, 14 categories), Home Ownership (`home_ownership_n`: Mortgage, Rent, Own, Other).
- **Target Variable:** `Default` (0 = Fully Paid: 80.02%, 1 = Defaulted: 19.98%).

### Preprocessing & Feature Engineering Pipeline:
1. **Target Isolation:** The `Default` label and record identifiers (`id`, `zip_code`, `title`, `desc`) were separated from the clustering matrix to prevent data leakage and maintain unsupervised integrity.
2. **Skewness Normalization:** Monetary variables (`revenue`, `loan_amnt`) exhibited strong positive skewness and were normalized via log-transformation ($\log(1+x)$).
3. **Outlier Treatment:** Debt-to-Income (`dti_n`) contained extreme outlier recording anomalies (up to 999%); these were capped at 60%, preserving 99.9% of real variance while eliminating distortion.
4. **Categorical Encoding:** `emp_length` was mapped to an ordinal tenure scale (0 to 10 years). `home_ownership_n` and `purpose` were one-hot encoded into binary indicator vectors.
5. **Feature Scaling:** All 24 resulting features were standardized using `StandardScaler` ($\mu = 0, \sigma = 1$).
6. **Stratified Sampling:** A balanced, stratified sample of 50,000 records (retaining the exact 80.02% / 19.98% outcome split) was generated for computationally intensive manifold learning and hierarchical clustering.

---

## 3. Dimensionality Reduction (PCA, t-SNE, UMAP)

### Principal Component Analysis (PCA)
- **Variance Retention:** 16 principal components capture $\ge 80\%$ of total variance across 24 features.
- **Component Interpretation:**
  - **PC1 (9.96% variance):** Represents **Financial Scale & Home Asset Backing** (high positive loadings on `home_MORTGAGE`, `log_revenue`, and `log_loan_amnt`).
  - **PC2 (7.63% variance):** Represents **Credit Card Refinancing vs. Debt Consolidation Purpose**.
  - **PC3 (5.85% variance):** Represents **Leverage & Loan Size Intensity**.

### Manifold Projections (t-SNE & UMAP)
- **t-SNE (10,000 samples, perplexity=35):** Revealed crisp, non-linear cluster islands showing distinct separation between homeowners, credit card refinancers, and high-debt personal loan applicants.
- **UMAP (15,000 samples, n_neighbors=30):** Successfully preserved both local borrower neighborhoods and global portfolio geometry, validating that the borrower segments occupy distinct regions in high-dimensional feature space.

---

## 4. Clustering Analysis & Algorithm Comparison

Three distinct unsupervised clustering algorithms were implemented, tuned, and evaluated:

| Metric / Attribute | K-Means Clustering | Hierarchical (Ward Linkage) | DBSCAN (Density-Based) |
|---|---|---|---|
| **Optimal Parameters** | $K = 4$ | $K = 4$, Ward Distance | $\epsilon = 2.5$, `min_samples` = 15 |
| **Number of Clusters** | 4 | 4 | 23 dense clusters + Noise |
| **Silhouette Score** | **0.1848** | 0.1008 | 0.0820 |
| **Calinski-Harabasz Index** | **741.3** | 686.2 | 312.4 |
| **Davies-Bouldin Index** | **1.6415** | 2.1506 | 2.4510 |
| **Outlier / Noise Treatment**| Partitions all points | Agglomerates all points | **Identifies 2.39% noise points** |
| **Operational Suitability** | **Primary Segmentation** | Confirmatory Structure | **Anomaly & Fraud Detection** |

### Algorithm Selection Rationale:
- **K-Means** emerged as the superior production model, providing the highest silhouette cohesion, excellent Davies-Bouldin scores, and high business interpretability.
- **Hierarchical Clustering (Dendrogram)** confirmed the 4-branch tree topology, proving that the borrower segments are natural clusters rather than arbitrary partitions.
- **DBSCAN** excelled specifically at isolating irregular, sparse loan applications that do not conform to standard borrower behaviors.

---

## 5. Discovered Borrower Segments & Risk Profiles

### Segment 0: Tier 3 — Stretched High-DTI Renters
- **Portfolio Share:** 39.64% (19,822 loans)
- **Profile:** Median Income: \$56,000 | Median Loan: \$10,375 | FICO: 694.0 | Renters: 99.9%
- **Historical Default Rate:** **23.13%** (Higher than baseline)
- **Business Insight:** Renters consolidating credit card debt with moderate incomes and limited collateral backing.

### Segment 1: Tier 2 — Established Outright Homeowners
- **Portfolio Share:** 10.61% (5,305 loans)
- **Profile:** Median Income: \$60,000 | Median Loan: \$12,000 | FICO: 700.4 | Outright Owners: 100%
- **Historical Default Rate:** **21.47%**
- **Business Insight:** Moderate default risk, supported by real property ownership but sometimes constrained in liquid cash flow.

### Segment 2: Tier 4 — High-Risk Non-Standard Segment
- **Portfolio Share:** 0.08% (41 loans)
- **Profile:** Median Income: \$70,000 | Median Loan: \$6,000 | Non-standard residential status
- **Historical Default Rate:** **24.39%** (Highest Default Rate)
- **Business Insight:** Small, atypical borrower group requiring manual underwriting verification.

### Segment 3: Tier 1 — Prime Mortgage Borrowers
- **Portfolio Share:** 49.66% (24,832 loans)
- **Profile:** Median Income: \$75,000 (Mean: \$89,198) | Median Loan: \$15,000 | FICO: 701.3 | Mortgage: 100%
- **Historical Default Rate:** **17.14%** (Lowest Default Rate, significantly below 19.98% baseline)
- **Business Insight:** Highest earning capacity, longer job tenure (6.2 years avg), largest loan capacity, and strongest repayment track record.

---

## 6. Anomaly Detection Insights
Using DBSCAN noise labeling combined with an Isolation Forest model (contamination rate = 3.0%):
- **1,500 anomalous loans** were isolated and examined.
- **Key Anomaly Characteristics:**
  - High loan-to-income ratios (e.g. borrowers requesting 80%+ of annual revenue in unsecured debt).
  - Outlier debt-to-income ratios (>45%) coupled with multiple concurrent revolving lines.
  - Subprime FICO scores with unusually large loan requests.
- **Default Probability in Anomalies:** Flagged anomalies exhibited an elevated default rate exceeding **24.5%**, validating the anomaly detection engine as an early-warning filter for credit underwriters.

---

## 7. Business Use Cases & Strategic Recommendations

1. **Risk-Based Tiered Pricing & APR Customization:**
   - Offer competitive, preferred interest rates to **Tier 1 Prime Mortgage Borrowers** (17.14% default risk) to improve portfolio acquisition.
   - Adjust APR upwards or require co-signers for **Tier 3 Renters** (23.13% default risk) to price in expected credit loss.

2. **Automated Fast-Track Underwriting:**
   - Applicants falling into Cluster 3 with DTI < 18% and FICO > 720 can be routed through **Straight-Through Processing (STP)** with zero human intervention.

3. **Proactive Portfolio Monitoring & Early Interventions:**
   - Continuously monitor existing loan accounts. If an existing customer's DTI or revolving credit utilization pushes them toward Cluster 0 or 2, initiate automated outreach or credit counseling before default occurs.

4. **Product Diversification:**
   - Design tailored debt-consolidation products for Tier 3 borrowers with structured automatic payroll deductions to mitigate non-repayment.

---

## 8. Streamlit Interactive Application
The solution features a full-featured interactive Streamlit web dashboard located at `app/app.py`:
- **Module 1:** Executive Portfolio Overview & Real-Time KPIs
- **Module 2:** Exploratory Data Analysis & Dynamic Risk Driver Visualizations
- **Module 3:** Interactive Dimensionality Reduction Explorer (PCA, t-SNE, UMAP with Plotly)
- **Module 4:** Borrower Segmentation & Cluster Profiling
- **Module 5:** Historical Default Pattern Analysis & Relative Risk Multipliers
- **Module 6:** Anomaly & Outlier Detection Explorer
- **Module 7:** Real-Time Borrower Segment Predictor (interactive applicant scoring tool)
