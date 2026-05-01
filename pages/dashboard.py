"""
CrimeScope — Main Dashboard Page
"""
 
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import time
from datetime import datetime
 
from utils.data_fetcher import fetch_city_data, get_city_config, get_available_cities
from utils.visualizations import (
    build_crime_map, crime_by_hour_chart, crime_type_donut, weekly_heatmap,
    trend_forecast_chart, severity_distribution, district_bar_chart,
    anomaly_chart, ml_scores_gauge,
)
from models.crime_models import (
    train_hotspot_clustering, train_crime_classifier, train_anomaly_detector,
    forecast_crime_trend, train_severity_predictor,
)
 
# ─── CSS ────────────────────────────────────────────────────────────────────
 
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0A0E1A !important;
    color: #F9FAFB !important;
}
 
.main { background-color: #0A0E1A; }
.block-container { padding: 1rem 2rem 2rem; max-width: 1600px; }
 
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #111827 100%) !important;
    border-right: 1px solid #1F2937;
}
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }
 
/* Header */
.crime-header {
    background: linear-gradient(135deg, #0D1117 0%, #1a0a2e 50%, #0D1117 100%);
    border: 1px solid #1F2937;
    border-left: 4px solid #FF4B4B;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.crime-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF4B4B, #F59E0B, #FF4B4B);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
    margin: 0;
    line-height: 1.1;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.crime-header .subtitle {
    color: #6B7280;
    font-size: 0.78rem;
    letter-spacing: 0.15em;
    margin-top: 0.4rem;
}
.live-badge {
    background: rgba(255,75,75,0.15);
    border: 1px solid #FF4B4B;
    color: #FF4B4B;
    padding: 0.2rem 0.6rem;
    border-radius: 99px;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
 
/* KPI Cards */
.kpi-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--kpi-color, #6366F1);
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--kpi-color, #6366F1);
}
.kpi-label { font-size: 0.7rem; color: #6B7280; letter-spacing: 0.1em; margin-top: 0.2rem; }
.kpi-delta { font-size: 0.75rem; margin-top: 0.3rem; }
 
/* Section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: #6B7280;
    text-transform: uppercase;
    border-bottom: 1px solid #1F2937;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}
 
/* Alert boxes */
.alert-high {
    background: rgba(255,75,75,0.1);
    border-left: 3px solid #FF4B4B;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-size: 0.82rem;
    margin: 0.4rem 0;
}
.alert-medium {
    background: rgba(245,158,11,0.1);
    border-left: 3px solid #F59E0B;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-size: 0.82rem;
    margin: 0.4rem 0;
}
.alert-low {
    background: rgba(16,185,129,0.1);
    border-left: 3px solid #10B981;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-size: 0.82rem;
    margin: 0.4rem 0;
}
 
/* Hotspot cards */
.hotspot-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 0.9rem;
    margin: 0.4rem 0;
}
.hotspot-rank {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #FF4B4B;
}
 
/* Metric overrides */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 1rem;
}
 
/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    color: #FF4B4B !important;
    border-bottom-color: #FF4B4B !important;
}
 
/* Buttons */
.stButton > button {
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid #6366F1 !important;
    color: #F9FAFB !important;
    font-family: 'DM Mono', monospace !important;
    border-radius: 6px !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.3) !important;
    border-color: #818CF8 !important;
}
 
/* Select boxes */
.stSelectbox > div > div {
    background: #111827 !important;
    border-color: #1F2937 !important;
    color: #F9FAFB !important;
}
 
/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1F2937; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #374151; }
</style>
"""
 
 
def render_dashboard():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size:2.5rem'>🔍</div>
            <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#FF4B4B'>CRIMESCOPE</div>
            <div style='font-size:0.65rem; color:#6B7280; letter-spacing:0.15em'>URBAN CRIME INTELLIGENCE</div>
        </div>
        <hr style='border-color:#1F2937; margin: 0.5rem 0 1rem'>
        """, unsafe_allow_html=True)
 
        st.markdown("**🏙️ CITY**")
        city = st.selectbox("", get_available_cities(), label_visibility="collapsed")
 
        st.markdown("**📅 TIME WINDOW**")
        days_back = st.slider("", 7, 90, 30, label_visibility="collapsed")
 
        st.markdown("**🗺️ MAP MODE**")
        map_type = st.radio("", ["heatmap", "clusters", "markers"], label_visibility="collapsed", horizontal=False)
 
        st.markdown("**⚠️ SEVERITY FILTER**")
        min_sev, max_sev = st.slider("", 1, 10, (1, 10), label_visibility="collapsed")
 
        st.markdown("**🤖 ML CLUSTERS**")
        n_clusters = st.slider("", 4, 15, 8, label_visibility="collapsed")
 
        st.markdown("---")
        refresh = st.button("🔄 Refresh Data", use_container_width=True)
 
        st.markdown("""
        <div style='font-size:0.65rem; color:#374151; margin-top:1rem; text-align:center'>
        Data: City Open Data Portals<br>
        Updated every 30 min<br>
        <a href='https://github.com/yourusername/urban-crime-heatmap' style='color:#6366F1'>GitHub →</a>
        </div>
        """, unsafe_allow_html=True)
 
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='crime-header'>
        <div style='font-size:3rem'>🔍</div>
        <div>
            <h1>CRIMESCOPE</h1>
            <div class='subtitle'>URBAN CRIME INTELLIGENCE PLATFORM · {city.upper()} · LAST {days_back} DAYS</div>
            <div style='margin-top:0.4rem'>
                <span class='live-badge'>● LIVE DATA</span>
                <span style='font-size:0.7rem; color:#374151; margin-left:0.8rem'>
                    Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Load Data ────────────────────────────────────────────────────────────
    with st.spinner("⚡ Fetching live crime data..."):
        df = fetch_city_data(city, days_back=days_back, limit=10000)
 
    if df.empty:
        st.error("No data available. Please try another city or time window.")
        return
 
    # Apply severity filter
    df_filtered = df[df["severity"].between(min_sev, max_sev)].copy()
 
    # ── KPI Row ──────────────────────────────────────────────────────────────
    total = len(df_filtered)
    avg_sev = df_filtered["severity"].mean()
    high_sev = len(df_filtered[df_filtered["severity"] >= 8])
    unique_types = df_filtered["crime_type"].nunique()
    peak_hour = int(df_filtered["hour"].mode()[0]) if len(df_filtered) > 0 else 0
    districts = df_filtered["district"].nunique() if "district" in df_filtered.columns else 0
 
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, f"{total:,}", "TOTAL INCIDENTS", "#6366F1", f"Last {days_back} days"),
        (c2, f"{avg_sev:.1f}/10", "AVG SEVERITY", "#FF4B4B", "Crime severity score"),
        (c3, f"{high_sev:,}", "HIGH SEVERITY", "#F59E0B", "Severity ≥ 8"),
        (c4, str(unique_types), "CRIME TYPES", "#10B981", "Unique categories"),
        (c5, f"{peak_hour:02d}:00", "PEAK HOUR", "#EC4899", "Highest activity"),
        (c6, str(districts), "DISTRICTS", "#06B6D4", "Active districts"),
    ]
 
    for col, val, label, color, sub in kpis:
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='--kpi-color:{color}'>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-delta' style='color:#6B7280'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown("<div style='margin: 1rem 0'></div>", unsafe_allow_html=True)
 
    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ HEATMAP", "📊 ANALYTICS", "🤖 ML INSIGHTS", "⚠️ ANOMALIES", "🔮 FORECAST"
    ])
 
    # ── TAB 1: Map ────────────────────────────────────────────────────────────
    with tab1:
        col_map, col_info = st.columns([2.5, 1])
 
        with col_map:
            cfg = get_city_config(city)
            city_center = cfg.get("center", [40.7128, -74.006])
 
            with st.spinner("🗺️ Rendering map..."):
                _, cluster_stats, _ = train_hotspot_clustering(df_filtered, n_clusters=n_clusters)
                folium_map = build_crime_map(
                    df_filtered, city_center,
                    cluster_stats=cluster_stats,
                    map_type=map_type,
                )
                map_html = folium_map._repr_html_()
                components.html(map_html, height=560, scrolling=False)
 
        with col_info:
            st.markdown("<div class='section-title'>🔴 TOP HOTSPOTS</div>", unsafe_allow_html=True)
 
            if cluster_stats is not None and not cluster_stats.empty:
                top5 = cluster_stats.sort_values("risk_score", ascending=False).head(5)
                for i, (_, row) in enumerate(top5.iterrows()):
                    risk_pct = min(100, int(row["risk_score"] * 10))
                    st.markdown(f"""
                    <div class='hotspot-card'>
                        <div style='display:flex; justify-content:space-between; align-items:center'>
                            <span class='hotspot-rank'>#{i+1}</span>
                            <span style='font-size:0.75rem; color:#FF4B4B'>{risk_pct}% risk</span>
                        </div>
                        <div style='font-size:0.8rem; margin: 0.3rem 0; color:#D1D5DB'>{row['top_crime']}</div>
                        <div style='font-size:0.7rem; color:#6B7280'>
                            {int(row['count'])} incidents · Sev {row['avg_severity']:.1f}/10
                        </div>
                        <div style='margin-top:0.4rem; background:#1F2937; border-radius:2px; height:4px'>
                            <div style='width:{risk_pct}%; background:linear-gradient(90deg,#FF4B4B,#F59E0B); height:4px; border-radius:2px'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
 
            st.markdown("<div class='section-title'>🚨 ACTIVE ALERTS</div>", unsafe_allow_html=True)
 
            recent = df_filtered[df_filtered["date"] >= (df_filtered["date"].max() - pd.Timedelta(hours=24))]
            if len(recent) > 0:
                if avg_sev >= 7:
                    st.markdown(f"<div class='alert-high'>🔴 HIGH SEVERITY — Avg {avg_sev:.1f}/10</div>", unsafe_allow_html=True)
                if high_sev > total * 0.15:
                    st.markdown(f"<div class='alert-high'>🔴 {high_sev} critical incidents in window</div>", unsafe_allow_html=True)
                if len(recent) > total / days_back * 1.5:
                    st.markdown(f"<div class='alert-medium'>🟡 Activity spike — {len(recent)} in last 24h</div>", unsafe_allow_html=True)
 
            st.markdown(f"<div class='alert-low'>🟢 Data refreshed {datetime.now().strftime('%H:%M')}</div>", unsafe_allow_html=True)
 
    # ── TAB 2: Analytics ──────────────────────────────────────────────────────
    with tab2:
        row1_l, row1_r = st.columns(2)
 
        with row1_l:
            st.plotly_chart(crime_by_hour_chart(df_filtered), use_container_width=True)
 
        with row1_r:
            st.plotly_chart(crime_type_donut(df_filtered), use_container_width=True)
 
        st.plotly_chart(weekly_heatmap(df_filtered), use_container_width=True)
 
        row3_l, row3_r = st.columns(2)
        with row3_l:
            st.plotly_chart(severity_distribution(df_filtered), use_container_width=True)
        with row3_r:
            st.plotly_chart(district_bar_chart(df_filtered), use_container_width=True)
 
    # ── TAB 3: ML Insights ───────────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-title'>🤖 MACHINE LEARNING MODELS</div>", unsafe_allow_html=True)
 
        with st.spinner("Training ML models on live data..."):
            clf, le, clf_accuracy, feat_importance = train_crime_classifier(df_filtered)
            _, cluster_stats_ml, sil_score = train_hotspot_clustering(df_filtered, n_clusters=n_clusters)
            gb_model, gb_score = train_severity_predictor(df_filtered)
 
        # Model score gauges
        if clf_accuracy and sil_score is not None and gb_score:
            st.plotly_chart(
                ml_scores_gauge(clf_accuracy, max(0, sil_score), max(0, gb_score)),
                use_container_width=True
            )
 
        ml_l, ml_r = st.columns(2)
 
        with ml_l:
            st.markdown("<div class='section-title'>📊 FEATURE IMPORTANCE (Random Forest)</div>", unsafe_allow_html=True)
            if feat_importance:
                fi_df = pd.DataFrame(list(feat_importance.items()), columns=["Feature", "Importance"])
                fi_df = fi_df.sort_values("Importance", ascending=True)
 
                import plotly.graph_objects as go
                fig_fi = go.Figure(go.Bar(
                    x=fi_df["Importance"],
                    y=fi_df["Feature"],
                    orientation="h",
                    marker=dict(
                        color=fi_df["Importance"],
                        colorscale=[[0, "#1F2937"], [1, "#6366F1"]],
                    ),
                ))
                fig_fi.update_layout(
                    paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                    font=dict(color="#F9FAFB", family="DM Mono, monospace"),
                    margin=dict(l=10, r=10, t=30, b=10),
                    title="Feature Importances",
                    height=260,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
 
        with ml_r:
            st.markdown("<div class='section-title'>🔵 CLUSTER SUMMARY (KMeans)</div>", unsafe_allow_html=True)
            if cluster_stats_ml is not None and not cluster_stats_ml.empty:
                display_cols = ["cluster_id", "count", "avg_severity", "top_crime", "risk_score"]
                display = cluster_stats_ml[display_cols].sort_values("risk_score", ascending=False)
                display.columns = ["Cluster", "Incidents", "Avg Sev", "Top Crime", "Risk Score"]
                display["Risk Score"] = display["Risk Score"].round(2)
                display["Avg Sev"] = display["Avg Sev"].round(2)
                st.dataframe(
                    display.style.background_gradient(subset=["Risk Score"], cmap="Reds"),
                    use_container_width=True,
                    hide_index=True,
                )
 
        # DBSCAN density
        st.markdown("<div class='section-title'>📍 CRIME TYPE × HOUR PIVOT (Top 6 Types)</div>", unsafe_allow_html=True)
        top6 = df_filtered["crime_type"].value_counts().head(6).index.tolist()
        pivot_data = df_filtered[df_filtered["crime_type"].isin(top6)]
        pivot = pivot_data.groupby(["crime_type", "hour"]).size().unstack(fill_value=0)
 
        import plotly.graph_objects as go
        fig_pivot = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in pivot.columns],
            y=[t[:30] for t in pivot.index],
            colorscale=[[0, "#0A0E1A"], [0.5, "#6366F1"], [1, "#FF4B4B"]],
            hovertemplate="<b>%{y}</b><br>Hour: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_pivot.update_layout(
            paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
            font=dict(color="#F9FAFB", family="DM Mono, monospace"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
        )
        st.plotly_chart(fig_pivot, use_container_width=True)
 
    # ── TAB 4: Anomalies ─────────────────────────────────────────────────────
    with tab4:
        with st.spinner("Running anomaly detection..."):
            iso_model, anomaly_df = train_anomaly_detector(df_filtered)
 
        if anomaly_df is not None and not anomaly_df.empty:
            n_anomalies = anomaly_df["is_anomaly"].sum()
 
            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("🔍 Anomalous Days", int(n_anomalies))
            with a2:
                st.metric("📅 Days Analyzed", len(anomaly_df))
            with a3:
                anomaly_rate = n_anomalies / len(anomaly_df) * 100
                st.metric("📊 Anomaly Rate", f"{anomaly_rate:.1f}%")
 
            st.plotly_chart(anomaly_chart(anomaly_df), use_container_width=True)
 
            if n_anomalies > 0:
                st.markdown("<div class='section-title'>⚠️ ANOMALOUS DATES</div>", unsafe_allow_html=True)
                anom_days = anomaly_df[anomaly_df["is_anomaly"]].sort_values("count", ascending=False)
                for _, row in anom_days.iterrows():
                    score = abs(row.get("anomaly_score", 0))
                    st.markdown(f"""
                    <div class='alert-high'>
                        <b>📅 {row['date_only']}</b> —
                        {int(row['count'])} incidents
                        (anomaly score: {score:.3f})
                    </div>
                    """, unsafe_allow_html=True)
 
    # ── TAB 5: Forecast ───────────────────────────────────────────────────────
    with tab5:
        forecast_days = st.slider("Forecast horizon (days)", 7, 30, 14)
 
        with st.spinner("Generating forecast..."):
            result = forecast_crime_trend(df_filtered, forecast_days=forecast_days)
 
        if result and len(result) == 3:
            combined_df, slope, intercept = result
            direction = "Rising 📈" if slope > 0 else "Declining 📉"
            daily_change = abs(slope)
 
            f1, f2, f3 = st.columns(3)
            with f1:
                st.metric("📈 Trend Direction", direction)
            with f2:
                st.metric("📊 Daily Change", f"{slope:+.1f} incidents/day")
            with f3:
                actual = combined_df[combined_df["type"] == "actual"]
                forecast = combined_df[combined_df["type"] == "forecast"]
                if len(forecast) > 0:
                    predicted_total = int(forecast["count"].sum())
                    st.metric("🔮 Forecasted Total", f"{predicted_total:,}", help=f"Next {forecast_days} days")
 
            st.plotly_chart(trend_forecast_chart(combined_df, slope), use_container_width=True)
 
            st.markdown("""
            <div class='alert-medium' style='margin-top:1rem'>
            ⚠️ <b>Disclaimer:</b> Forecasts use linear regression on historical patterns.
            Actual crime rates depend on many socioeconomic factors not captured in this model.
            Use for research & awareness purposes only.
            </div>
            """, unsafe_allow_html=True)
 
    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <hr style='border-color:#1F2937; margin: 2rem 0 1rem'>
    <div style='text-align:center; font-size:0.7rem; color:#374151'>
        CrimeScope · Built with Streamlit, Scikit-learn, Folium, Plotly ·
        Data: City Open Data Portals (Socrata) ·
        <a href='https://github.com/yourusername/urban-crime-heatmap' style='color:#6366F1'>GitHub</a> ·
        For research & awareness purposes only
    </div>
    """, unsafe_allow_html=True)
