import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Portfolio Risk & Segmentation Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

# -----------------------------------------------------------------------------
# HIGHLY DISTINGUISHABLE CLUSTER PALETTES
# -----------------------------------------------------------------------------
CLUSTER_COLORS = {
    0: "#E11D48",  # Vivid Crimson Rose (Tier 3 - Renters)
    1: "#F59E0B",  # Bright Warm Amber (Tier 2 - Owners)
    2: "#8B5CF6",  # Electric Violet Purple (Tier 4 - Edge)
    3: "#059669"   # Deep Emerald Green (Tier 1 - Prime)
}

CLUSTER_STR_COLORS = {
    "0": "#E11D48",
    "1": "#F59E0B",
    "2": "#8B5CF6",
    "3": "#059669"
}

SEGMENT_COLORS = {
    "Tier 1: Prime Prime Borrowers (Lowest Default Risk)": "#059669",
    "Tier 2: Established Moderate-Risk Borrowers": "#F59E0B",
    "Tier 3: Stretched High-DTI Borrowers": "#E11D48",
    "Tier 4: Subprime High-Default Segment (Highest Risk)": "#8B5CF6"
}

DEFAULT_STATUS_COLORS = {
    0: "#2563EB",  # Solid Blue for Fully Paid
    1: "#DC2626"   # Sharp Red for Defaulted
}

PLOT_TEMPLATE = "plotly_white"

# -----------------------------------------------------------------------------
# ADVANCED FINTECH UI STYLING (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        font-weight: 400;
        max-width: 900px;
        line-height: 1.5;
    }
    .hero-badges {
        margin-top: 1rem;
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Modern Glass KPI Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-accent-blue { border-top: 4px solid #2563eb; }
    .metric-accent-emerald { border-top: 4px solid #059669; }
    .metric-accent-amber { border-top: 4px solid #f59e0b; }
    .metric-accent-rose { border-top: 4px solid #e11d48; }
    .metric-accent-purple { border-top: 4px solid #8b5cf6; }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-sub {
        margin-top: 0.4rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: #475569;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Persona Callout Cards */
    .persona-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .persona-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA & MODEL LOADING
# -----------------------------------------------------------------------------
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

    # Ensure kmeans_cluster is string formatted for categorical plotting
    df['cluster_label'] = "Cluster " + df['kmeans_cluster'].astype(str)

    return df, meta, pca_coords, tsne_coords, tsne_idx, umap_coords, umap_idx, scaler, kmeans

df, meta, pca_coords, tsne_coords, tsne_idx, umap_coords, umap_idx, scaler, kmeans = load_data()

CLUSTER_LABEL_COLORS = {
    "Cluster 0": "#E11D48",  # Rose Red
    "Cluster 1": "#F59E0B",  # Amber Gold
    "Cluster 2": "#8B5CF6",  # Electric Purple
    "Cluster 3": "#059669"   # Emerald Green
}

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
        <span style="font-size: 2.2rem;">🏛️</span>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #0f172a; line-height: 1.2;">APEX CREDIT</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 600; letter-spacing: 0.05em;">RISK INTELLIGENCE SUITE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    navigation = st.radio(
        "NAVIGATION MODULES",
        [
            "🏛️ Executive Overview & KPIs",
            "📊 Exploratory Credit Analysis",
            "🌀 2D & 3D Manifold Visualizer",
            "👥 Borrower Cluster Profiles",
            "⚠️ Default Pattern Analysis",
            "🚨 Anomaly & Risk Detection",
            "🎯 Real-Time Underwriter Simulator"
        ]
    )

    st.markdown("---")
    st.markdown("### 🎨 Color Spectrum Legend")
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.8rem; font-weight: 600;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 14px; height: 14px; border-radius: 50%; background: #059669; display: inline-block;"></span>
            <span>Cluster 3: Prime Mortgage (Low Risk)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 14px; height: 14px; border-radius: 50%; background: #F59E0B; display: inline-block;"></span>
            <span>Cluster 1: Homeowners (Moderate Risk)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 14px; height: 14px; border-radius: 50%; background: #E11D48; display: inline-block;"></span>
            <span>Cluster 0: Stretched Renters (Elevated Risk)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 14px; height: 14px; border-radius: 50%; background: #8B5CF6; display: inline-block;"></span>
            <span>Cluster 2: Edge Segment (Highest Risk)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔒 **Guaranteed Unsupervised Separation:** `Default` outcome was completely excluded from model training to prevent target leakage.")

# ==============================================================================
# MODULE 1: EXECUTIVE OVERVIEW & KPIS
# ==============================================================================
if navigation == "🏛️ Executive Overview & KPIs":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Credit Portfolio Segmentation & Risk Analytics</div>
        <div class="hero-subtitle">
            Enterprise Unsupervised Learning architecture isolating high-dimensional borrower risk archetypes, empirical default patterns, and non-conforming financial anomalies across 1.35 million historical loan applications.
        </div>
        <div class="hero-badges">
            <span class="hero-badge">📊 1,347,681 Historical Loans</span>
            <span class="hero-badge">🧠 Unsupervised K-Means & Manifolds</span>
            <span class="hero-badge">🎯 Zero Target Leakage</span>
            <span class="hero-badge">🛡️ Automated Anomaly Quarantine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card metric-accent-blue">
            <div class="metric-label">Total Portfolio</div>
            <div class="metric-value">1,347,681</div>
            <div class="metric-sub">50,000 Sample Evaluated</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card metric-accent-emerald">
            <div class="metric-label">Average Loan</div>
            <div class="metric-value">${df['loan_amnt'].mean():,.0f}</div>
            <div class="metric-sub">Median: ${df['loan_amnt'].median():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card metric-accent-amber">
            <div class="metric-label">Median Income</div>
            <div class="metric-value">${df['revenue'].median():,.0f}</div>
            <div class="metric-sub">Mean: ${df['revenue'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card metric-accent-purple">
            <div class="metric-label">Mean FICO Score</div>
            <div class="metric-value">{df['fico_n'].mean():.0f}</div>
            <div class="metric-sub">IQR: 672 – 712</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        base_def = df['Default'].mean() * 100
        st.markdown(f"""
        <div class="metric-card metric-accent-rose">
            <div class="metric-label">Portfolio Default Rate</div>
            <div class="metric-value" style="color: #dc2626;">{base_def:.2f}%</div>
            <div class="metric-sub">Unsupervised Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Analytics Row
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown('<div class="section-title">📊 Borrower Cluster Distribution</div>', unsafe_allow_html=True)
        seg_counts = df['cluster_label'].value_counts().reset_index()
        seg_counts.columns = ['Cluster', 'Borrowers']
        
        fig_donut = px.pie(
            seg_counts,
            names='Cluster',
            values='Borrowers',
            hole=0.55,
            color='Cluster',
            color_discrete_map=CLUSTER_LABEL_COLORS
        )
        fig_donut.update_traces(
            textinfo='percent+label',
            marker=dict(line=dict(color='#ffffff', width=2)),
            pull=[0.05 if c == "Cluster 3" else 0 for c in seg_counts['Cluster']]
        )
        fig_donut.update_layout(
            template=PLOT_TEMPLATE,
            margin=dict(t=20, b=20, l=20, r=20),
            height=370,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="section-title">⚠️ Historical Default Rate by Discovered Cluster</div>', unsafe_allow_html=True)
        risk_df = df.groupby('cluster_label')['Default'].mean().reset_index()
        risk_df['Default_Pct'] = (risk_df['Default'] * 100).round(2)
        risk_df = risk_df.sort_values(by='Default_Pct')

        fig_bar = px.bar(
            risk_df,
            x='cluster_label',
            y='Default_Pct',
            color='cluster_label',
            color_discrete_map=CLUSTER_LABEL_COLORS,
            text='Default_Pct',
            labels={'Default_Pct': 'Historical Default Rate (%)', 'cluster_label': 'Discovered Cluster'}
        )
        fig_bar.add_hline(
            y=base_def,
            line_dash='dash',
            line_color='#0f172a',
            line_width=2,
            annotation_text=f"Portfolio Baseline ({base_def:.1f}%)",
            annotation_position="top right"
        )
        fig_bar.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside',
            marker=dict(line=dict(color='#0f172a', width=1))
        )
        fig_bar.update_layout(
            template=PLOT_TEMPLATE,
            margin=dict(t=30, b=20, l=20, r=20),
            height=370,
            showlegend=False,
            yaxis=dict(range=[0, max(risk_df['Default_Pct']) * 1.25])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🔍 Quick Borrower Records Inspector</div>', unsafe_allow_html=True)
    st.dataframe(
        df[['id', 'issue_d', 'revenue', 'loan_amnt', 'fico_n', 'dti_n', 'emp_length', 'home_ownership_n', 'purpose', 'cluster_label', 'Default']].head(100),
        use_container_width=True
    )

# ==============================================================================
# MODULE 2: EXPLORATORY CREDIT ANALYSIS
# ==============================================================================
elif navigation == "📊 Exploratory Credit Analysis":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Exploratory Credit Analysis</div>
        <div class="hero-subtitle">
            Univariate, bivariate, and correlation diagnostics across borrower income, loan request size, FICO score distributions, and debt burdens.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_dist, tab_cat, tab_corr = st.tabs(["📈 Numerical Distributions", "🏷️ Categorical Risk Drivers", "🔥 Financial Correlation Matrix"])

    with tab_dist:
        col1, col2 = st.columns([1, 1])
        with col1:
            feature_selected = st.selectbox(
                "Select Feature for Distribution Inspection:",
                ["revenue", "loan_amnt", "fico_n", "dti_capped"],
                format_func=lambda x: {"revenue": "Annual Income ($)", "loan_amnt": "Loan Amount ($)", "fico_n": "FICO Credit Score", "dti_capped": "Debt-to-Income Ratio (%)"}[x]
            )
            fig_hist = px.histogram(
                df,
                x=feature_selected,
                color='Default',
                barmode='overlay',
                nbins=50,
                color_discrete_map=DEFAULT_STATUS_COLORS,
                opacity=0.65,
                title=f"Distribution of {feature_selected} Partitioned by Default Outcome"
            )
            fig_hist.update_layout(template=PLOT_TEMPLATE, height=420)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.markdown(f"#### Boxplot Quartile Analysis: `{feature_selected}`")
            fig_box = px.box(
                df,
                x='Default',
                y=feature_selected,
                color='Default',
                color_discrete_map=DEFAULT_STATUS_COLORS,
                points="outliers",
                title=f"Quartile Spread Comparison (0 = Fully Paid, 1 = Default)"
            )
            fig_box.update_layout(template=PLOT_TEMPLATE, height=420)
            st.plotly_chart(fig_box, use_container_width=True)

    with tab_cat:
        st.subheader("Category Default Rates & Concentration")
        cat_var = st.selectbox("Select Categorical Factor:", ["purpose", "home_ownership_n", "emp_length"])
        
        cat_df = df.groupby(cat_var).agg(
            total_loans=('id', 'count'),
            default_rate=('Default', 'mean')
        ).reset_index()
        cat_df['default_rate_pct'] = (cat_df['default_rate'] * 100).round(2)
        cat_df = cat_df.sort_values(by='default_rate_pct', ascending=False)

        fig_cat_bar = px.bar(
            cat_df,
            x=cat_var,
            y='default_rate_pct',
            color='default_rate_pct',
            color_continuous_scale='Reds',
            text='default_rate_pct',
            labels={'default_rate_pct': 'Default Rate (%)'}
        )
        fig_cat_bar.add_hline(y=df['Default'].mean() * 100, line_dash='dash', line_color='#0f172a', annotation_text="Portfolio Avg")
        fig_cat_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_cat_bar.update_layout(template=PLOT_TEMPLATE, height=450)
        st.plotly_chart(fig_cat_bar, use_container_width=True)

    with tab_corr:
        st.subheader("Correlation Heatmap of Financial Drivers")
        corr_cols = ['revenue', 'log_revenue', 'loan_amnt', 'log_loan_amnt', 'fico_n', 'dti_capped', 'emp_length_years', 'Default']
        corr_mat = df[corr_cols].corr()

        fig_heat = px.imshow(
            corr_mat,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-0.4,
            zmax=0.4,
            aspect='auto',
            title="Pearson Correlation Matrix (Financial Metrics vs. Historical Default)"
        )
        fig_heat.update_layout(template=PLOT_TEMPLATE, height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

# ==============================================================================
# MODULE 3: 2D & 3D MANIFOLD VISUALIZER (DISTINGUISHABLE CLUSTERS)
# ==============================================================================
elif navigation == "🌀 2D & 3D Manifold Visualizer":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Dimensionality Reduction & Manifold Explorer</div>
        <div class="hero-subtitle">
            Project high-dimensional customer space (24 features) into human-interpretable 2D and 3D manifold topologies using PCA, t-SNE, and UMAP with vibrant, distinguishable cluster palettes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    m_col1, m_col2, m_col3 = st.columns([1, 1, 1])
    with m_col1:
        technique = st.radio("Reduction Algorithm:", ["PCA (Linear)", "t-SNE (Manifold)", "UMAP (Topological)"], horizontal=True)
    with m_col2:
        color_feature = st.selectbox(
            "Color Scatter Points By:",
            ["cluster_label", "Default", "home_ownership_n", "fico_n", "loan_amnt", "revenue"]
        )
    with m_col3:
        is_3d = st.checkbox("Enable 3D View (PCA Only)", value=False) if technique.startswith("PCA") else False

    # Dynamic color mapping for distinguishable rendering
    if color_feature == "cluster_label":
        discrete_map = CLUSTER_LABEL_COLORS
        continuous_scale = None
    elif color_feature == "Default":
        discrete_map = DEFAULT_STATUS_COLORS
        continuous_scale = None
    elif color_feature == "home_ownership_n":
        discrete_map = {"MORTGAGE": "#059669", "RENT": "#E11D48", "OWN": "#F59E0B", "OTHER": "#8B5CF6"}
        continuous_scale = None
    else:
        discrete_map = None
        continuous_scale = "Turbo"

    if technique.startswith("PCA"):
        if pca_coords is not None:
            pca_df = df.iloc[:len(pca_coords)].copy()
            pca_df['PC1'] = pca_coords[:, 0]
            pca_df['PC2'] = pca_coords[:, 1]
            pca_df['PC3'] = pca_coords[:, 2]

            sample_plot = pca_df.sample(n=min(6000, len(pca_df)), random_state=42)

            if is_3d:
                fig_proj = px.scatter_3d(
                    sample_plot,
                    x='PC1', y='PC2', z='PC3',
                    color=color_feature,
                    color_discrete_map=discrete_map,
                    color_continuous_scale=continuous_scale,
                    opacity=0.75,
                    hover_data=['revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'cluster_label'],
                    title="Interactive 3D PCA Projection (Top 3 Principal Components)"
                )
                fig_proj.update_traces(marker=dict(size=4))
                fig_proj.update_layout(height=650, margin=dict(t=30, b=10, l=10, r=10))
            else:
                fig_proj = px.scatter(
                    sample_plot,
                    x='PC1', y='PC2',
                    color=color_feature,
                    color_discrete_map=discrete_map,
                    color_continuous_scale=continuous_scale,
                    opacity=0.65,
                    hover_data=['revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'cluster_label'],
                    title="2D PCA Projection (PC1 vs PC2)"
                )
                fig_proj.update_traces(marker=dict(size=5, line=dict(width=0.4, color='#ffffff')))
                fig_proj.update_layout(template=PLOT_TEMPLATE, height=600)

            st.plotly_chart(fig_proj, use_container_width=True)

            with st.expander("📊 View PCA Scree Plot & Cumulative Variance"):
                scree_path = os.path.join(FIGURES_DIR, "06_pca_scree_plot.png")
                if os.path.exists(scree_path):
                    st.image(scree_path, use_container_width=True)

    elif technique.startswith("t-SNE"):
        if tsne_coords is not None and tsne_idx is not None:
            tsne_df = df.iloc[tsne_idx].copy()
            tsne_df['tSNE_1'] = tsne_coords[:, 0]
            tsne_df['tSNE_2'] = tsne_coords[:, 1]

            fig_tsne = px.scatter(
                tsne_df,
                x='tSNE_1', y='tSNE_2',
                color=color_feature,
                color_discrete_map=discrete_map,
                color_continuous_scale=continuous_scale,
                opacity=0.7,
                hover_data=['revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'cluster_label'],
                title="t-SNE 2D Manifold Embedding (Perplexity=35, 10,000 Points)"
            )
            fig_tsne.update_traces(marker=dict(size=4.5, line=dict(width=0.3, color='#ffffff')))
            fig_tsne.update_layout(template=PLOT_TEMPLATE, height=600)
            st.plotly_chart(fig_tsne, use_container_width=True)

    elif technique.startswith("UMAP"):
        if umap_coords is not None and umap_idx is not None:
            umap_df = df.iloc[umap_idx].copy()
            umap_df['UMAP_1'] = umap_coords[:, 0]
            umap_df['UMAP_2'] = umap_coords[:, 1]

            fig_umap = px.scatter(
                umap_df,
                x='UMAP_1', y='UMAP_2',
                color=color_feature,
                color_discrete_map=discrete_map,
                color_continuous_scale=continuous_scale,
                opacity=0.7,
                hover_data=['revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'cluster_label'],
                title="UMAP 2D Manifold Projection (n_neighbors=30, 15,000 Points)"
            )
            fig_umap.update_traces(marker=dict(size=4.5, line=dict(width=0.3, color='#ffffff')))
            fig_umap.update_layout(template=PLOT_TEMPLATE, height=600)
            st.plotly_chart(fig_umap, use_container_width=True)

# ==============================================================================
# MODULE 4: BORROWER CLUSTER PROFILES
# ==============================================================================
elif navigation == "👥 Borrower Cluster Profiles":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Borrower Segmentation & Cluster Personas</div>
        <div class="hero-subtitle">
            Comprehensive financial, demographic, and risk persona breakdown for each discovered borrower segment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    summary_file = os.path.join(PROCESSED_DIR, "cluster_summary_stats.csv")
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        
        # Display Persona Cards in a 4-Column Row
        p_cols = st.columns(4)
        for i, row in summary_df.iterrows():
            cid = int(row['kmeans_cluster'])
            c_color = CLUSTER_COLORS.get(cid, '#2563eb')
            with p_cols[i]:
                st.markdown(f"""
                <div class="persona-card" style="border-top: 5px solid {c_color};">
                    <span class="persona-tag" style="background-color: {c_color};">Cluster {cid}</span>
                    <div style="font-weight: 800; font-size: 1.05rem; color: #0f172a; margin-top: 4px;">{row['persona_name']}</div>
                    <hr style="margin: 8px 0; border: 0; border-top: 1px solid #e2e8f0;">
                    <div style="font-size: 0.85rem; line-height: 1.6;">
                        <b>Share of Portfolio:</b> {row['share_of_portfolio']:.1f}%<br>
                        <b>Default Rate:</b> <span style="font-weight: 700; color: {c_color};">{row['default_rate_pct']:.2f}%</span><br>
                        <b>Mean Income:</b> ${row['mean_income']:,.0f}<br>
                        <b>Mean Loan:</b> ${row['mean_loan']:,.0f}<br>
                        <b>Mean FICO:</b> {row['mean_fico']:.0f}<br>
                        <b>Mean DTI:</b> {row['mean_dti']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Multi-Metric Comparison Across Clusters</div>', unsafe_allow_html=True)
    comp_metric = st.selectbox(
        "Select Financial Dimension to Compare:",
        ["revenue", "loan_amnt", "fico_n", "dti_capped", "emp_length_years"],
        format_func=lambda x: {"revenue": "Annual Income ($)", "loan_amnt": "Loan Amount ($)", "fico_n": "FICO Credit Score", "dti_capped": "Debt-to-Income (%)", "emp_length_years": "Employment Tenure (Years)"}[x]
    )

    fig_cluster_box = px.box(
        df,
        x='cluster_label',
        y=comp_metric,
        color='cluster_label',
        color_discrete_map=CLUSTER_LABEL_COLORS,
        points=False,
        title=f"Distribution of {comp_metric} Across Discovered Clusters"
    )
    fig_cluster_box.update_layout(template=PLOT_TEMPLATE, height=450, showlegend=False)
    st.plotly_chart(fig_cluster_box, use_container_width=True)

    st.markdown("---")
    st.subheader("Clustering Algorithm Performance Comparison Matrix")
    comp_md_path = os.path.join(BASE_DIR, "reports", "clustering_comparison.md")
    if os.path.exists(comp_md_path):
        with open(comp_md_path, encoding='utf-8') as f:
            st.markdown(f.read())

# ==============================================================================
# MODULE 5: DEFAULT PATTERN ANALYSIS
# ==============================================================================
elif navigation == "⚠️ Default Pattern Analysis":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Historical Default Pattern Analysis</div>
        <div class="hero-subtitle">
            Validating that unsupervised segmentation captures genuine credit risk gradients without having seen the Default outcome during cluster assignment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Empirical Default Rate vs. Portfolio Baseline")
        c_risk = df.groupby('cluster_label')['Default'].mean().reset_index()
        c_risk['Default_Pct'] = (c_risk['Default'] * 100).round(2)
        c_risk = c_risk.sort_values(by='Default_Pct')

        fig_def_bar = px.bar(
            c_risk,
            x='cluster_label',
            y='Default_Pct',
            color='cluster_label',
            color_discrete_map=CLUSTER_LABEL_COLORS,
            text='Default_Pct',
            title="Cluster Default Rate (%)"
        )
        fig_def_bar.add_hline(y=df['Default'].mean()*100, line_dash='dash', line_color='#0f172a', annotation_text="Portfolio Baseline (19.98%)")
        fig_def_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_def_bar.update_layout(template=PLOT_TEMPLATE, height=400, showlegend=False)
        st.plotly_chart(fig_def_bar, use_container_width=True)

    with colB:
        st.subheader("Risk Multiplier / Relative Odds")
        base = df['Default'].mean()
        risk_tbl = df.groupby('cluster_label').agg(
            Loans=('id', 'count'),
            Default_Rate=('Default', 'mean')
        ).reset_index()
        risk_tbl['Relative_Risk'] = (risk_tbl['Default_Rate'] / base).round(2)
        risk_tbl['Default_Rate'] = (risk_tbl['Default_Rate'] * 100).round(2).astype(str) + "%"
        risk_tbl['Risk_Tier'] = np.where(risk_tbl['Relative_Risk'] < 0.9, '🟢 Low Risk (Prime)',
                                np.where(risk_tbl['Relative_Risk'] < 1.1, '🟡 Moderate Risk', '🔴 High Risk'))
        st.dataframe(risk_tbl, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Strategic Underwriting Recommendations")
    st.markdown("""
    - **Cluster 3 (Tier 1 - Emerald):** Lowest default rate (17.14%). Ideal for **Straight-Through Processing (STP)**, premium loan limits, and preferential interest rates.
    - **Cluster 1 (Tier 2 - Amber):** Moderate risk (21.47%). Outright homeowners with sound asset backing; suitable for standard underwriting and home equity cross-selling.
    - **Cluster 0 (Tier 3 - Rose):** Elevated risk (23.13%). High-DTI renters consolidating debt. Requires tighter debt-to-income caps and automatic payroll deductions.
    - **Cluster 2 (Tier 4 - Purple):** High-risk outlier applications (24.39%). Mandatory manual underwriting and collateral verification.
    """)

# ==============================================================================
# MODULE 6: ANOMALY & RISK DETECTION
# ==============================================================================
elif navigation == "🚨 Anomaly & Risk Detection":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Anomaly Detection & High-Risk Quarantine</div>
        <div class="hero-subtitle">
            Density-based spatial clustering (DBSCAN) and Isolation Forest algorithms flagging non-conforming, high-risk, or potentially fraudulent credit profiles.
        </div>
    </div>
    """, unsafe_allow_html=True)

    anom_count = df['is_anomaly'].sum()
    st.warning(f"🚨 **Flagged Anomaly Portfolio:** {anom_count:,} loan records ({anom_count/len(df)*100:.2f}% of portfolio flagged)")

    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.markdown("#### Debt-to-Income (DTI) Anomaly Profile")
        fig_dti_anom = px.histogram(
            df,
            x='dti_capped',
            color='is_anomaly',
            barmode='overlay',
            nbins=40,
            color_discrete_map={0: '#2563eb', 1: '#dc2626'},
            opacity=0.65,
            title="DTI Density: Standard Applications vs. Flagged Anomalies"
        )
        fig_dti_anom.update_layout(template=PLOT_TEMPLATE, height=420)
        st.plotly_chart(fig_dti_anom, use_container_width=True)

    with a_col2:
        st.markdown("#### Bivariate Income vs. Loan Discrepancies")
        fig_scat_anom = px.scatter(
            df.sample(n=min(5000, len(df)), random_state=42),
            x='revenue',
            y='loan_amnt',
            color='is_anomaly',
            color_discrete_map={0: '#94a3b8', 1: '#dc2626'},
            opacity=0.75,
            title="Loan Amount vs. Annual Revenue (Anomalies in Red)"
        )
        fig_scat_anom.update_layout(template=PLOT_TEMPLATE, height=420)
        st.plotly_chart(fig_scat_anom, use_container_width=True)

    st.markdown("---")
    st.subheader("Quarantined Anomaly Records (First 50 Rows)")
    st.dataframe(
        df[df['is_anomaly'] == 1][['id', 'revenue', 'loan_amnt', 'fico_n', 'dti_n', 'purpose', 'home_ownership_n', 'cluster_label', 'Default']].head(50),
        use_container_width=True
    )

# ==============================================================================
# MODULE 7: REAL-TIME UNDERWRITER SIMULATOR
# ==============================================================================
elif navigation == "🎯 Real-Time Underwriter Simulator":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Real-Time Borrower Segment Predictor</div>
        <div class="hero-subtitle">
            Interactive scoring engine allowing loan officers to test applicant criteria, project cluster assignment, and view empirical risk ratings instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("underwriting_simulator_form"):
        st.markdown("### 📋 Applicant Financial Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            in_rev = st.number_input("Annual Income ($):", min_value=10000, max_value=2000000, value=75000, step=5000)
            in_loan = st.number_input("Requested Loan Amount ($):", min_value=1000, max_value=40000, value=15000, step=1000)
            in_fico = st.slider("FICO Credit Score:", min_value=600, max_value=850, value=715)

        with col2:
            in_dti = st.slider("Debt-to-Income (DTI) Ratio (%):", min_value=0.0, max_value=60.0, value=17.5, step=0.5)
            in_tenure = st.selectbox("Employment Length:", ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], index=10)
            in_exp = st.selectbox("Employment Experience Record:", [1, 0], index=0)

        with col3:
            in_home = st.selectbox("Residential / Home Status:", ['MORTGAGE', 'RENT', 'OWN', 'OTHER'])
            in_purpose = st.selectbox("Loan Purpose:", [
                'debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase',
                'small_business', 'car', 'medical', 'moving', 'vacation', 'house', 'other'
            ])

        sim_submit = st.form_submit_button("⚡ Predict Segment & Underwrite Application", use_container_width=True)

    if sim_submit and scaler is not None and kmeans is not None:
        emp_map = {'10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3, '2 years': 2, '1 year': 1, '< 1 year': 0}
        emp_years = emp_map.get(in_tenure, 0)
        log_rev = np.log1p(float(in_rev))
        log_loan = np.log1p(float(in_loan))
        dti_cap = min(float(in_dti), 60.0)

        # Assemble full 24-feature vector
        feature_cols = meta["clustering_feature_cols"]
        vector_dict = {col: 0.0 for col in feature_cols}
        vector_dict['log_revenue'] = log_rev
        vector_dict['log_loan_amnt'] = log_loan
        vector_dict['fico_n'] = float(in_fico)
        vector_dict['dti_capped'] = dti_cap
        vector_dict['emp_length_years'] = float(emp_years)
        vector_dict['experience_c'] = float(in_exp)

        home_key = f"home_{in_home}"
        if home_key in vector_dict:
            vector_dict[home_key] = 1.0

        purpose_key = f"purpose_{in_purpose}"
        if purpose_key in vector_dict:
            vector_dict[purpose_key] = 1.0

        input_frame = pd.DataFrame([vector_dict])[feature_cols]
        scaled_vec = scaler.transform(input_frame)

        pred_cid = int(kmeans.predict(scaled_vec)[0])
        pred_color = CLUSTER_COLORS.get(pred_cid, "#2563eb")

        # Lookup cluster statistics
        stats_file = os.path.join(PROCESSED_DIR, "cluster_summary_stats.csv")
        stats_data = pd.read_csv(stats_file)
        match_row = stats_data[stats_data['kmeans_cluster'] == pred_cid].iloc[0]

        persona_str = match_row['persona_name']
        def_rate_val = match_row['default_rate_pct']
        share_val = match_row['share_of_portfolio']

        st.markdown(f"""
        <div style="background: white; border-radius: 14px; border: 2px solid {pred_color}; padding: 1.5rem; margin-top: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="background: {pred_color}; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">
                        ASSIGNED TO CLUSTER {pred_cid}
                    </span>
                    <h2 style="margin-top: 10px; margin-bottom: 4px; color: #0f172a;">{persona_str}</h2>
                    <p style="color: #64748b; margin: 0;">Unsupervised Machine Learning Classification Result</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">PORTFOLIO SHARE</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #0f172a;">{share_val:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Historical Default Rate", f"{def_rate_val:.2f}%", delta=f"{def_rate_val - 19.98:.2f}% vs Baseline", delta_color="inverse")
        with r2:
            tier_name = "🟢 Prime Low-Risk" if def_rate_val < 18 else ("🟡 Established Moderate" if def_rate_val < 22 else "🔴 Subprime / Stretched")
            st.metric("Assigned Risk Tier", tier_name)
        with r3:
            rec_badge = "✅ Straight-Through Approval (STP)" if def_rate_val < 18 else ("⚠️ Standard Credit Underwriting" if def_rate_val < 22 else "🛑 Collateral or Co-Signer Required")
            st.metric("Policy Recommendation", rec_badge)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 1rem;">
    Credit Portfolio Segmentation and Risk Pattern Analysis | Antigravity AI Data Science Engineering
</div>
""", unsafe_allow_html=True)
