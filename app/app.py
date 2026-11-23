import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(
    page_title="Credit Portfolio Segmentation & Risk Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 18px;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    parquet_path = os.path.join(PROCESSED_DIR, "segmented_loans_sample.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = os.path.join(PROCESSED_DIR, "segmented_loans_sample.csv")
        df = pd.read_csv(csv_path)
    
    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)

    # Coordinates
    pca_coords = np.load(os.path.join(PROCESSED_DIR, "pca_coordinates.npy")) if os.path.exists(os.path.join(PROCESSED_DIR, "pca_coordinates.npy")) else None
    tsne_coords = np.load(os.path.join(PROCESSED_DIR, "tsne_coordinates.npy")) if os.path.exists(os.path.join(PROCESSED_DIR, "tsne_coordinates.npy")) else None
    tsne_idx = np.load(os.path.join(PROCESSED_DIR, "tsne_sample_indices.npy")) if os.path.exists(os.path.join(PROCESSED_DIR, "tsne_sample_indices.npy")) else None
    umap_coords = np.load(os.path.join(PROCESSED_DIR, "umap_coordinates.npy")) if os.path.exists(os.path.join(PROCESSED_DIR, "umap_coordinates.npy")) else None
    umap_idx = np.load(os.path.join(PROCESSED_DIR, "umap_sample_indices.npy")) if os.path.exists(os.path.join(PROCESSED_DIR, "umap_sample_indices.npy")) else None

    # Models
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "scaler.joblib")) else None
    kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans_optimal_model.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "kmeans_optimal_model.joblib")) else None

    return df, meta, pca_coords, tsne_coords, tsne_idx, umap_coords, umap_idx, scaler, kmeans

df, meta, pca_coords, tsne_coords, tsne_idx, umap_coords, umap_idx, scaler, kmeans = load_data()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
st.sidebar.title("Credit Risk Intelligence")
st.sidebar.caption("Unsupervised Borrower Segmentation & Risk Engine")

navigation = st.sidebar.radio(
    "Go To Module:",
    [
        "🏛️ Portfolio Overview & KPIs",
        "📊 Exploratory Data Analysis",
        "🌀 Dimensionality Reduction (PCA / t-SNE / UMAP)",
        "👥 Borrower Segmentation & Clusters",
        "⚠️ Historical Default Pattern Analysis",
        "🚨 Anomaly & Outlier Detection",
        "🎯 Real-Time Borrower Segment Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** Lending Club 1.35M Loan Portfolio\n\n"
    "**Target Isolation:** `Default` column was strictly excluded during clustering to ensure true unsupervised discovery."
)

# ==============================================================================
# TAB 1: PORTFOLIO OVERVIEW & KPIS
# ==============================================================================
if navigation == "🏛️ Portfolio Overview & KPIs":
    st.markdown('<div class="main-header">Credit Portfolio Segmentation & Risk Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive Dashboard: Unsupervised Customer Segmentation, Credit Risk Stratification, and Anomaly Detection</div>', unsafe_allow_html=True)

    # Top KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Portfolio Loans</div>
            <div class="kpi-value">1,347,681</div>
            <div class="kpi-sub">50,000 Sample Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Loan Amount</div>
            <div class="kpi-value">${df['loan_amnt'].mean():,.0f}</div>
            <div class="kpi-sub">Median: ${df['loan_amnt'].median():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Annual Revenue</div>
            <div class="kpi-value">${df['revenue'].mean():,.0f}</div>
            <div class="kpi-sub">Median: ${df['revenue'].median():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Mean FICO Score</div>
            <div class="kpi-value">{df['fico_n'].mean():.0f}</div>
            <div class="kpi-sub">Range: 612 - 848</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        base_def = df['Default'].mean() * 100
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Portfolio Default Rate</div>
            <div class="kpi-value" style="color: #d9534f;">{base_def:.2f}%</div>
            <div class="kpi-sub" style="color: #6c757d;">Historical Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Summary Charts
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Distribution of Discovered Borrower Segments")
        segment_counts = df['segment_name'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Borrowers']
        fig_pie = px.pie(
            segment_counts,
            names='Segment',
            values='Borrowers',
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Historical Default Rate Across Borrower Tiers")
        risk_by_seg = df.groupby('segment_name')['Default'].mean().reset_index()
        risk_by_seg['Default_Pct'] = (risk_by_seg['Default'] * 100).round(2)
        risk_by_seg = risk_by_seg.sort_values(by='Default_Pct')

        fig_bar = px.bar(
            risk_by_seg,
            x='Default_Pct',
            y='segment_name',
            orientation='h',
            text='Default_Pct',
            color='Default_Pct',
            color_continuous_scale='RdYlGn_r',
            labels={'Default_Pct': 'Default Rate (%)', 'segment_name': 'Borrower Segment'}
        )
        fig_bar.add_vline(x=base_def, line_dash='dash', line_color='black', annotation_text=f"Portfolio Avg ({base_def:.1f}%)")
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Data Sample Explorer")
    st.dataframe(
        df[['id', 'issue_d', 'revenue', 'loan_amnt', 'fico_n', 'dti_n', 'emp_length', 'home_ownership_n', 'purpose', 'Default', 'segment_name']].head(100),
        use_container_width=True
    )

# ==============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif navigation == "📊 Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Examine distributions, credit profiles, and historical default drivers across the loan portfolio.")

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Distributions", "Risk Drivers by Category", "Correlation Heatmap"])

    with eda_tab1:
        st.subheader("Feature Distributions")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            metric_to_plot = st.selectbox("Select Numerical Feature:", ["revenue", "loan_amnt", "fico_n", "dti_capped"])
            fig_hist = px.histogram(
                df,
                x=metric_to_plot,
                color='Default',
                barmode='overlay',
                nbins=40,
                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                title=f"Distribution of {metric_to_plot} by Default Status"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with fcol2:
            st.write("#### Boxplot Quartile Analysis")
            fig_box = px.box(
                df,
                x='Default',
                y=metric_to_plot,
                color='Default',
                color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                title=f"{metric_to_plot} Comparison: Paid (0) vs Default (1)"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with eda_tab2:
        st.subheader("Risk Drivers by Categorical Attributes")
        cat_select = st.selectbox("Choose Category:", ["purpose", "home_ownership_n", "emp_length"])
        cat_risk = df.groupby(cat_select).agg(
            total=('id', 'count'),
            default_rate=('Default', 'mean')
        ).reset_index()
        cat_risk['default_rate_pct'] = (cat_risk['default_rate'] * 100).round(2)
        cat_risk = cat_risk.sort_values(by='default_rate_pct', ascending=False)

        fig_cat = px.bar(
            cat_risk,
            x=cat_select,
            y='default_rate_pct',
            color='default_rate_pct',
            color_continuous_scale='Viridis_r',
            text='default_rate_pct',
            title=f"Default Rate across {cat_select}"
        )
        fig_cat.add_hline(y=df['Default'].mean() * 100, line_dash='dash', line_color='red', annotation_text="Average")
        st.plotly_chart(fig_cat, use_container_width=True)

    with eda_tab3:
        st.subheader("Correlation Matrix of Key Credit Variables")
        corr_cols = ['revenue', 'log_revenue', 'loan_amnt', 'log_loan_amnt', 'fico_n', 'dti_capped', 'emp_length_years', 'Default']
        corr_matrix = df[corr_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-0.5,
            zmax=0.5,
            title="Correlation Matrix Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# TAB 3: DIMENSIONALITY REDUCTION (PCA / t-SNE / UMAP)
# ==============================================================================
elif navigation == "🌀 Dimensionality Reduction (PCA / t-SNE / UMAP)":
    st.title("Dimensionality Reduction & Manifold Learning")
    st.write("2D & 3D Projections to uncover latent customer patterns and separate borrower risk archetypes.")

    proj_choice = st.radio("Choose Dimensionality Reduction Technique:", ["PCA (Principal Component Analysis)", "t-SNE (t-Distributed Stochastic Neighbor Embedding)", "UMAP (Uniform Manifold Approximation & Projection)"], horizontal=True)

    color_by = st.selectbox("Color Projection By:", ["kmeans_cluster", "Default", "home_ownership_n", "fico_n", "loan_amnt"])

    if proj_choice.startswith("PCA"):
        if pca_coords is not None:
            plot_df = df.iloc[:len(pca_coords)].copy()
            plot_df['PC1'] = pca_coords[:, 0]
            plot_df['PC2'] = pca_coords[:, 1]
            plot_df['PC3'] = pca_coords[:, 2]

            fig_pca = px.scatter(
                plot_df.sample(n=min(5000, len(plot_df)), random_state=42),
                x='PC1',
                y='PC2',
                color=color_by,
                hover_data=['revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'Default'],
                title="PCA Projection (PC1 vs PC2)",
                opacity=0.6,
                color_continuous_scale='Turbo' if color_by in ['fico_n', 'loan_amnt'] else None
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            with st.expander("View PCA Scree Plot & Explained Variance"):
                scree_img_path = os.path.join(FIGURES_DIR, "06_pca_scree_plot.png")
                if os.path.exists(scree_img_path):
                    st.image(scree_img_path, caption="PCA Explained Variance & Scree Plot", use_container_width=True)

    elif proj_choice.startswith("t-SNE"):
        if tsne_coords is not None and tsne_idx is not None:
            tsne_df = df.iloc[tsne_idx].copy()
            tsne_df['tSNE_1'] = tsne_coords[:, 0]
            tsne_df['tSNE_2'] = tsne_coords[:, 1]

            fig_tsne = px.scatter(
                tsne_df,
                x='tSNE_1',
                y='tSNE_2',
                color=color_by,
                hover_data=['revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'Default'],
                title="t-SNE 2D Manifold Projection",
                opacity=0.6,
                color_continuous_scale='Turbo' if color_by in ['fico_n', 'loan_amnt'] else None
            )
            st.plotly_chart(fig_tsne, use_container_width=True)

    elif proj_choice.startswith("UMAP"):
        if umap_coords is not None and umap_idx is not None:
            umap_df = df.iloc[umap_idx].copy()
            umap_df['UMAP_1'] = umap_coords[:, 0]
            umap_df['UMAP_2'] = umap_coords[:, 1]

            fig_umap = px.scatter(
                umap_df,
                x='UMAP_1',
                y='UMAP_2',
                color=color_by,
                hover_data=['revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'Default'],
                title="UMAP 2D Manifold Projection",
                opacity=0.6,
                color_continuous_scale='Turbo' if color_by in ['fico_n', 'loan_amnt'] else None
            )
            st.plotly_chart(fig_umap, use_container_width=True)

# ==============================================================================
# TAB 4: BORROWER SEGMENTATION & CLUSTERS
# ==============================================================================
elif navigation == "👥 Borrower Segmentation & Clusters":
    st.title("Borrower Segmentation & Cluster Profiling")
    st.write("Compare the financial, credit, and employment characteristics of each discovered customer segment.")

    summary_path = os.path.join(PROCESSED_DIR, "cluster_summary_stats.csv")
    if os.path.exists(summary_path):
        cluster_summary = pd.read_csv(summary_path)
        st.dataframe(
            cluster_summary[['kmeans_cluster', 'persona_name', 'share_of_portfolio', 'default_rate_pct', 'mean_income', 'mean_loan', 'mean_fico', 'mean_dti', 'mean_tenure']],
            use_container_width=True
        )

    st.subheader("Financial Comparison Across Segments")
    metric = st.selectbox(
        "Compare Metric Across Segments:",
        ["revenue", "loan_amnt", "fico_n", "dti_capped", "emp_length_years"]
    )
    fig_comp = px.box(
        df,
        x='segment_name',
        y=metric,
        color='segment_name',
        title=f"{metric} Distribution Across Discovered Borrower Segments",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("Clustering Algorithm Performance Comparison")
    comp_md_path = os.path.join(BASE_DIR, "reports", "clustering_comparison.md")
    if os.path.exists(comp_md_path):
        with open(comp_md_path, encoding='utf-8') as f:
            st.markdown(f.read())

# ==============================================================================
# TAB 5: HISTORICAL DEFAULT PATTERN ANALYSIS
# ==============================================================================
elif navigation == "⚠️ Historical Default Pattern Analysis":
    st.title("Historical Default Pattern Analysis")
    st.markdown(
        "> **Key Principle:** The target column `Default` was **strictly excluded** during clustering. "
        "The patterns observed below represent natural risk stratification purely from financial and credit behavior."
    )

    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Cluster Default Rate vs Baseline")
        fig_risk = px.bar(
            df.groupby('segment_name')['Default'].mean().reset_index().rename(columns={'Default': 'Default_Rate'}),
            x='segment_name',
            y='Default_Rate',
            color='Default_Rate',
            color_continuous_scale='Reds',
            text_auto='.1%'
        )
        fig_risk.add_hline(y=df['Default'].mean(), line_dash='dash', line_color='black', annotation_text="Baseline")
        fig_risk.update_layout(xaxis_tickangle=-30, height=400)
        st.plotly_chart(fig_risk, use_container_width=True)

    with colB:
        st.subheader("Risk Multiplier / Odds Ratio")
        base = df['Default'].mean()
        risk_summary = df.groupby('segment_name').agg(
            borrowers=('id', 'count'),
            default_rate=('Default', 'mean')
        ).reset_index()
        risk_summary['relative_risk'] = (risk_summary['default_rate'] / base).round(2)
        risk_summary['risk_tier'] = np.where(risk_summary['relative_risk'] < 0.9, 'Low Risk',
                                    np.where(risk_summary['relative_risk'] < 1.1, 'Moderate Risk', 'High Risk'))
        st.dataframe(risk_summary, use_container_width=True)

# ==============================================================================
# TAB 6: ANOMALY & OUTLIER DETECTION
# ==============================================================================
elif navigation == "🚨 Anomaly & Outlier Detection":
    st.title("Anomaly Detection & Extreme Profile Investigation")
    st.write("Identifies suspicious, abnormal, or edge-case credit applications flagged by DBSCAN and Isolation Forest.")

    anomaly_count = df['is_anomaly'].sum()
    st.warning(f"**Total Anomalies Flagged:** {anomaly_count:,} ({anomaly_count/len(df)*100:.2f}% of portfolio)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Normal vs Anomalous DTI Distribution")
        fig_kde = px.histogram(
            df,
            x='dti_capped',
            color='is_anomaly',
            barmode='overlay',
            nbins=40,
            color_discrete_map={0: '#2980b9', 1: '#c0392b'},
            title="DTI Comparison (Normal vs Flagged Anomalies)"
        )
        st.plotly_chart(fig_kde, use_container_width=True)

    with col2:
        st.write("#### Loan Amount vs Annual Income (Anomalies Highlighted)")
        fig_anom_scatter = px.scatter(
            df.sample(n=min(5000, len(df)), random_state=42),
            x='revenue',
            y='loan_amnt',
            color='is_anomaly',
            color_discrete_map={0: '#bdc3c7', 1: '#e74c3c'},
            title="Income vs Loan Size with Flagged Anomalies"
        )
        st.plotly_chart(fig_anom_scatter, use_container_width=True)

    st.subheader("Inspect Flagged Anomalous Records")
    st.dataframe(
        df[df['is_anomaly'] == 1][['id', 'revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'home_ownership_n', 'Default', 'segment_name']].head(50),
        use_container_width=True
    )

# ==============================================================================
# TAB 7: REAL-TIME BORROWER SEGMENT PREDICTOR
# ==============================================================================
elif navigation == "🎯 Real-Time Borrower Segment Predictor":
    st.title("🎯 Real-Time Borrower Segment Predictor")
    st.write("Enter an applicant's financial attributes to predict their assigned borrower segment, historical risk tier, and underwriting recommendation.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            inp_income = st.number_input("Annual Income ($):", min_value=10000, max_value=1000000, value=75000, step=5000)
            inp_loan = st.number_input("Requested Loan Amount ($):", min_value=1000, max_value=50000, value=15000, step=1000)
            inp_fico = st.slider("FICO Credit Score:", min_value=600, max_value=850, value=710)

        with col2:
            inp_dti = st.slider("Debt-to-Income (DTI) Ratio (%):", min_value=0.0, max_value=60.0, value=18.5, step=0.5)
            inp_tenure = st.selectbox("Employment Length:", ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], index=10)
            inp_experience = st.selectbox("Employment History Flag:", [1, 0], index=0)

        with col3:
            inp_home = st.selectbox("Home Ownership:", ['MORTGAGE', 'RENT', 'OWN', 'OTHER'])
            inp_purpose = st.selectbox("Loan Purpose:", [
                'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase',
                'small_business', 'car', 'medical', 'moving', 'vacation', 'house', 'other'
            ])

        submit = st.form_submit_button("Predict Borrower Segment & Risk Profile", use_container_width=True)

    if submit and scaler is not None and kmeans is not None:
        # Preprocess input row
        emp_map = {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0}
        emp_years = emp_map.get(inp_tenure, 0)
        log_rev = np.log1p(float(inp_income))
        log_loan = np.log1p(float(inp_loan))
        dti_cap = min(float(inp_dti), 60.0)

        # Feature vector
        feature_cols = meta["clustering_feature_cols"]
        row_dict = {col: 0.0 for col in feature_cols}
        row_dict['log_revenue'] = log_rev
        row_dict['log_loan_amnt'] = log_loan
        row_dict['fico_n'] = float(inp_fico)
        row_dict['dti_capped'] = dti_cap
        row_dict['emp_length_years'] = float(emp_years)
        row_dict['experience_c'] = float(inp_experience)

        # One-hot flags
        home_col = f"home_{inp_home}"
        if home_col in row_dict:
            row_dict[home_col] = 1.0

        purpose_col = f"purpose_{inp_purpose}"
        if purpose_col in row_dict:
            row_dict[purpose_col] = 1.0

        input_df = pd.DataFrame([row_dict])[feature_cols]
        input_scaled = scaler.transform(input_df)

        pred_cluster = int(kmeans.predict(input_scaled)[0])

        # Cluster persona lookup
        cluster_stats_df = pd.read_csv(os.path.join(PROCESSED_DIR, "cluster_summary_stats.csv"))
        matched_cluster = cluster_stats_df[cluster_stats_df['kmeans_cluster'] == pred_cluster].iloc[0]
        persona = matched_cluster['persona_name']
        hist_def_rate = matched_cluster['default_rate_pct']

        st.success(f"### Predicted Cluster ID: **Cluster {pred_cluster}**")
        st.info(f"**Segment Persona:** {persona}")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Historical Default Rate for Segment", f"{hist_def_rate:.2f}%", delta=f"{hist_def_rate - 19.98:.2f}% vs Baseline", delta_color="inverse")
        with res_col2:
            st.metric("Segment Share of Portfolio", f"{matched_cluster['share_of_portfolio']:.1f}%")
        with res_col3:
            recommendation = "✅ Fast-Track Approval / Preferred Rate" if hist_def_rate < 18 else ("⚠️ Standard Underwriting" if hist_def_rate < 22 else "🛑 Enhanced Due Diligence / Collateral Required")
            st.metric("Underwriting Recommendation", recommendation)

st.markdown("---")
st.caption("Credit Portfolio Segmentation & Risk Pattern Analysis | Antigravity AI Data Science Solution")
