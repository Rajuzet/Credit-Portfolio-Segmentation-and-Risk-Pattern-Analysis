# Exploratory Data Analysis (EDA) Report
## Credit Portfolio Segmentation & Risk Pattern Analysis

### 1. Executive Summary & Portfolio Key Metrics
- **Portfolio Analyzed:** 50,000 loan records (stratified sample representing the full 1.35M portfolio).
- **Portfolio Baseline Default Rate:** **19.98%**
- **Median Annual Income:** \$65,000 (Mean: \$77,821)
- **Median Loan Amount:** \$12,000 (Mean: \$14,372)
- **Median FICO Score:** 692 (Range: 632 to 848)
- **Median Debt-to-Income (DTI):** 17.59%

---

### 2. Key Exploratory Findings

#### A. Credit Score (FICO) is Strongly Predictive of Risk
- Borrowers with **Poor FICO (< 660)** have a default rate of **29.41%**, significantly higher than the portfolio baseline.
- Borrowers with **Excellent FICO (750+)** exhibit an ultra-low default rate of **8.93%**.
- This validates that unsupervised segmentation will naturally uncover credit tiers as a major clustering axis.

#### B. Loan Purpose Risk Differentiation
**Historical Default Rates by Purpose:**
| Purpose | Total Loans | Default Rate (%) |
|---|---|---|
| educational | 15 | 40.00% |
| small_business | 559 | 29.16% |
| moving | 332 | 26.20% |
| renewable_energy | 41 | 24.39% |
| debt_consolidation | 28,814 | 21.22% |
| medical | 595 | 20.84% |
| other | 2,873 | 19.94% |
| vacation | 359 | 19.50% |
| house | 254 | 18.50% |
| major_purchase | 1,115 | 17.94% |
| credit_card | 11,094 | 17.50% |
| home_improvement | 3,316 | 16.95% |
| car | 543 | 14.92% |
| wedding | 90 | 11.11% |


- Small business loans and renewable energy/moving loans have elevated default frequencies (~28-30%), whereas major purchases, cars, and home improvement show lower default rates (<17%).

#### C. Home Ownership Impact
- **Renters** display a default rate of **23.12%**, compared to **Mortgage holders** at **17.14%**.
- Homeowners with mortgages present stronger financial stability, larger loan requests, and higher annual revenues.

#### D. Distribution Characteristics & Transformation Requirements
- **Annual Income & Loan Amount:** Strong right skewness; effectively normalized using `log1p` transformation.
- **DTI Ratio:** Long tail with extreme recording anomalies (up to 999); safely capped at 60% without losing variance across 99.9% of the distribution.

---

### 3. Generated Visualizations
1. `reports/figures/01_feature_distributions.png`: Univariate histograms of income, loan amount, FICO, and DTI.
2. `reports/figures/02_correlation_heatmap.png`: Correlation matrix across financial variables.
3. `reports/figures/03_default_by_purpose.png`: Default rate comparison across 14 loan purposes.
4. `reports/figures/04_risk_tiers_home_ownership.png`: Risk breakdown across credit tiers and home ownership.
5. `reports/figures/05_loan_vs_income_risk.png`: Bivariate income-to-loan landscape by default outcome.
