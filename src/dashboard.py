"""
Interactive Plotly Dash Dashboard for Corporate Compliance.
Displays risk metrics, anomaly detection results, and model performance.
"""

import math
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# path resolution
import os
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data")
_OUTPUT_DIR  = os.path.join(_PROJECT_DIR, "outputs")

# load data
def load_data() -> pd.DataFrame:
    """Load the scored workbook or raw CSV."""
    paths = [
        os.path.join(_OUTPUT_DIR, "metallurgical_ledgers_scored.xlsx"),
        os.path.join(_DATA_DIR, "metallurgical_ledgers.csv"),
    ]
    for path in paths:
        try:
            if path.endswith(".xlsx"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)

            # Re-attach ground-truth if missing
            if "Is_Fraud_Ground_Truth" not in df.columns:
                try:
                    raw = pd.read_csv(os.path.join(_DATA_DIR, "metallurgical_ledgers.csv"))
                    df  = df.merge(raw[["Transaction_ID","Is_Fraud_Ground_Truth","Fraud_Type"]],
                                   on="Transaction_ID", how="left")
                except FileNotFoundError:
                    pass

            df["Date"] = pd.to_datetime(df.get("Date", pd.NaT))
            df["month"] = df["Date"].dt.to_period("M").astype(str)
            print(f"  [Dashboard] Loaded '{path}': {len(df):,} rows")
            return df
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No data file found. Run src/compliance_risk_scorer.py first.")


# benford's law helper
BENFORD_EXPECTED = {d: math.log10(1 + 1/d) * 100 for d in range(1, 10)}


def leading_digit(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0:
        return None
    s = f"{v:.10e}"
    for ch in s.split("e")[0].replace(".", "").lstrip("0"):
        if ch.isdigit() and ch != "0":
            return int(ch)
    return None


def compute_benford(df: pd.DataFrame, col: str = "Total_Value_USD") -> pd.DataFrame:
    digits = df[col].apply(leading_digit).dropna().astype(int)
    total  = len(digits)
    rows   = []
    for d in range(1, 10):
        obs_pct = (digits == d).sum() / total * 100
        exp_pct = BENFORD_EXPECTED[d]
        rows.append({
            "digit":    d,
            "observed": round(obs_pct, 3),
            "expected": round(exp_pct, 3),
            "deviation": round(abs(obs_pct - exp_pct), 3),
        })
    return pd.DataFrame(rows)


# mahalanobis distance
def compute_mahalanobis_dash(df: pd.DataFrame) -> pd.Series:
    features = ["Volume_MT", "Unit_Price_USD", "Total_Value_USD"]
    X       = df[features].fillna(df[features].median()).values.astype(float)
    mu      = X.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(X, rowvar=False))
    diff    = X - mu
    sq      = (diff @ cov_inv * diff).sum(axis=1)
    return pd.Series(np.sqrt(np.maximum(sq, 0.0)), index=df.index)


# score columns
SCORE_COLS = [
    "ofac_risk_score",
    "price_delta_risk_score",
    "smurfing_risk_score",
    "benfords_law_risk_score",
    "geopolitical_risk_score",
    "mahalanobis_risk_score",
    "composite_risk_score",
]

SCORE_LABELS = {
    "ofac_risk_score":          "OFAC / Sanctions",
    "price_delta_risk_score":   "Price Delta",
    "smurfing_risk_score":      "Smurfing",
    "benfords_law_risk_score":  "Benford's Law",
    "geopolitical_risk_score":  "Geopolitical",
    "mahalanobis_risk_score":   "Mahalanobis",
    "composite_risk_score":     "Composite",
}

RISK_COLORS = {
    "CRITICAL": "#e74c3c",
    "HIGH":     "#f39c12",
    "MEDIUM":   "#f1c40f",
    "LOW":      "#27ae60",
}

# app
df_global = load_data()

# Compute mahalanobis if column absent
if "mahalanobis_risk_score" not in df_global.columns:
    dist = compute_mahalanobis_dash(df_global)
    threshold = math.sqrt(9.348)
    df_global["mahalanobis_risk_score"] = (dist / threshold * 100).clip(upper=100).round(4)

# Compute composite if absent
available_score_cols = [c for c in SCORE_COLS[:-1] if c in df_global.columns]
if "composite_risk_score" not in df_global.columns and available_score_cols:
    df_global["composite_risk_score"] = df_global[available_score_cols].mean(axis=1).round(4)

if "risk_tier" not in df_global.columns and "composite_risk_score" in df_global.columns:
    df_global["risk_tier"] = df_global["composite_risk_score"].apply(
        lambda s: "CRITICAL" if s >= 75 else "HIGH" if s >= 50 else "MEDIUM" if s >= 25 else "LOW"
    )

benford_df = compute_benford(df_global)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Corporate Compliance | CCFRAD Dashboard",
    suppress_callback_exceptions=True,
)

# kpi card helper
def kpi_card(title: str, value: str, subtitle: str = "", color: str = "#3498db") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.85rem", "fontWeight": "600"}),
            html.H3(value, style={"color": color, "fontWeight": "800", "marginBottom": "2px"}),
            html.P(subtitle, className="text-muted mb-0", style={"fontSize": "0.75rem"}),
        ]),
        className="shadow-sm",
        style={"background": "#1e2a38", "border": f"1px solid {color}33", "borderRadius": "10px"},
    )


# layout
COUNTRIES   = sorted(df_global["Vendor_Country"].unique().tolist())
COMMODITIES = sorted(df_global["Commodity"].unique().tolist())
RISK_TIERS  = sorted(df_global.get("risk_tier", pd.Series(["LOW"])).unique().tolist()) if "risk_tier" in df_global.columns else ["LOW"]

total_txns    = f"{len(df_global):,}"
total_fraud   = f"{df_global.get('Is_Fraud_Ground_Truth', pd.Series(0)).sum():,}"
fraud_rate    = f"{df_global.get('Is_Fraud_Ground_Truth', pd.Series(0)).mean()*100:.1f}%"
n_critical    = (df_global.get("risk_tier", pd.Series("LOW")) == "CRITICAL").sum()
avg_composite = f"{df_global.get('composite_risk_score', pd.Series([0])).mean():.1f}"
total_vol     = f"${df_global['Total_Value_USD'].sum()/1e9:.2f}B"

app.layout = dbc.Container(
    fluid=True,
    style={"background": "#141e2a", "minHeight": "100vh", "padding": "20px"},
    children=[

        # ── Header ────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2("🛡️ Corporate Compliance | Financial Risk Dashboard",
                            style={"color": "#e8edf5", "fontWeight": "800", "marginBottom": "4px"}),
                    html.P("Trade-Based Money Laundering · Benford's Law · Mahalanobis · Regression Anomaly Detection",
                           style={"color": "#7f8fa6", "fontSize": "0.9rem"}),
                ]),
            ]),
            dbc.Col([
                html.Div([
                    html.Span("● LIVE", style={"color": "#27ae60", "fontWeight": "700", "fontSize": "0.9rem"}),
                    html.Span(" | Metallurgical Ledger 2023",
                              style={"color": "#7f8fa6", "fontSize": "0.85rem"}),
                ], style={"textAlign": "right", "paddingTop": "15px"}),
            ], width=3),
        ], className="mb-3"),

        html.Hr(style={"borderColor": "#2c3e50", "marginBottom": "20px"}),

        # ── Filters ───────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Label("Filter by Country", style={"color": "#aab", "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="filter-country",
                    options=[{"label": c, "value": c} for c in COUNTRIES],
                    multi=True, placeholder="All Countries",
                    style={"backgroundColor": "#1e2a38", "color": "#000"},
                ),
            ], width=4),
            dbc.Col([
                html.Label("Filter by Commodity", style={"color": "#aab", "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="filter-commodity",
                    options=[{"label": c, "value": c} for c in COMMODITIES],
                    multi=True, placeholder="All Commodities",
                    style={"backgroundColor": "#1e2a38"},
                ),
            ], width=4),
            dbc.Col([
                html.Label("Filter by Risk Tier", style={"color": "#aab", "fontSize": "0.8rem"}),
                dcc.Dropdown(
                    id="filter-tier",
                    options=[{"label": t, "value": t} for t in RISK_TIERS],
                    multi=True, placeholder="All Tiers",
                    style={"backgroundColor": "#1e2a38"},
                ),
            ], width=4),
        ], className="mb-4"),

        # ── KPI Cards ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(kpi_card("Total Transactions", total_txns, "All vendors", "#3498db"), width=2),
            dbc.Col(kpi_card("Total Volume", total_vol, "USD (2023)", "#9b59b6"), width=2),
            dbc.Col(kpi_card("Fraud Cases", total_fraud, "Ground-truth labels", "#e74c3c"), width=2),
            dbc.Col(kpi_card("Fraud Rate", fraud_rate, "Of total transactions", "#e67e22"), width=2),
            dbc.Col(kpi_card("CRITICAL Risk", f"{n_critical:,}", "Composite score ≥ 75", "#e74c3c"), width=2),
            dbc.Col(kpi_card("Avg Risk Score", avg_composite, "Composite (0–100)", "#f39c12"), width=2),
        ], className="mb-4", id="kpi-row"),

        # ── Row 1: TBML Heatmap + Benford's Law ───────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 TBML Risk Heatmap — Country × Commodity",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="tbml-heatmap", style={"height": "380px"})),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Benford's Law Analysis — Invoice Leading Digits",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="benfords-chart", style={"height": "380px"})),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=6),
        ], className="mb-4"),

        # ── Row 2: Mahalanobis Scatter + Risk Distribution ─────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎯 Mahalanobis Distance — Multivariate Outlier Detection",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="mahal-scatter", style={"height": "380px"})),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📉 Risk Score Distribution by Dimension",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="score-dim-selector",
                            options=[{"label": SCORE_LABELS.get(c, c), "value": c}
                                     for c in SCORE_COLS if c in df_global.columns],
                            value="composite_risk_score",
                            clearable=False,
                            style={"marginBottom": "8px", "backgroundColor": "#1e2a38"},
                        ),
                        dcc.Graph(id="risk-dist-chart", style={"height": "340px"}),
                    ]),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=6),
        ], className="mb-4"),

        # ── Row 3: Time Series + Fraud Type Breakdown ─────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📅 Monthly Transaction Volume & Avg Composite Risk",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="time-series-chart", style={"height": "320px"})),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔍 Fraud Type Breakdown",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="fraud-pie", style={"height": "320px"})),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=4),
        ], className="mb-4"),

        # ── Row 4: Model Metrics + Country Risk Bar ───────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🤖 ML Model Performance — Fraud Detection",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="model-metrics-chart", style={"height": "320px"})),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🌍 Avg Composite Risk by Vendor Country",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody(dcc.Graph(id="country-risk-bar", style={"height": "320px"})),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=6),
        ], className="mb-4"),

        # ── Row 5: Top-Risk Vendor Table ──────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🏭 Top 50 Highest-Risk Vendors",
                                   style={"background": "#1a2533", "color": "#e8edf5", "fontWeight": "600"}),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="vendor-table",
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": "#0f1923",
                                "color": "#e8edf5",
                                "fontWeight": "700",
                                "border": "1px solid #2c3e50",
                            },
                            style_cell={
                                "backgroundColor": "#1a2533",
                                "color": "#ccd6f6",
                                "border": "1px solid #2c3e50",
                                "fontSize": "0.82rem",
                                "padding": "6px 10px",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": '{risk_tier} = "CRITICAL"'},
                                    "backgroundColor": "#3d1515",
                                    "color": "#ff6b6b",
                                },
                                {
                                    "if": {"filter_query": '{risk_tier} = "HIGH"'},
                                    "backgroundColor": "#3d2a10",
                                    "color": "#f39c12",
                                },
                            ],
                            sort_action="native",
                            filter_action="native",
                            page_size=15,
                        ),
                    ]),
                ], style={"background": "#1a2533", "border": "1px solid #2c3e50"}),
            ], width=12),
        ], className="mb-4"),

        # Footer
        html.Hr(style={"borderColor": "#2c3e50"}),
        html.P(
            "Corporate Compliance – Financial Risk Anomaly Detection | "
            "Data: UN Comtrade · Yahoo Finance · World Bank | "
            "Models: Mahalanobis Distance · Logistic Regression · Random Forest",
            style={"color": "#4a5568", "fontSize": "0.75rem", "textAlign": "center"}
        ),
    ]
)


# callbacks
def apply_filters(countries, commodities, tiers) -> pd.DataFrame:
    dff = df_global.copy()
    if countries:
        dff = dff[dff["Vendor_Country"].isin(countries)]
    if commodities:
        dff = dff[dff["Commodity"].isin(commodities)]
    if tiers and "risk_tier" in dff.columns:
        dff = dff[dff["risk_tier"].isin(tiers)]
    return dff


@app.callback(
    Output("tbml-heatmap",       "figure"),
    Output("benfords-chart",     "figure"),
    Output("mahal-scatter",      "figure"),
    Output("time-series-chart",  "figure"),
    Output("fraud-pie",          "figure"),
    Output("country-risk-bar",   "figure"),
    Output("vendor-table",       "data"),
    Output("vendor-table",       "columns"),
    Input("filter-country",      "value"),
    Input("filter-commodity",    "value"),
    Input("filter-tier",         "value"),
)
def update_charts(countries, commodities, tiers):
    dff = apply_filters(countries, commodities, tiers)

    DARK_TEMPLATE = dict(
        paper_bgcolor="#1a2533",
        plot_bgcolor="#1a2533",
        font=dict(color="#ccd6f6", size=11),
    )

    # ── 1. TBML Heatmap ─────────────────────────────────────────────────────
    if "composite_risk_score" in dff.columns:
        heat_data = dff.groupby(["Vendor_Country","Commodity"])["composite_risk_score"].mean().reset_index()
        heat_pivot = heat_data.pivot(index="Vendor_Country", columns="Commodity", values="composite_risk_score").round(2)
        tbml_fig = go.Figure(go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.tolist(),
            y=heat_pivot.index.tolist(),
            colorscale=[
                [0.0,  "#0d3349"], [0.25, "#1a6b8a"],
                [0.5,  "#f39c12"], [0.75, "#e74c3c"],
                [1.0,  "#7b0000"],
            ],
            showscale=True,
            colorbar=dict(title="Risk Score"),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Avg Risk: %{z:.1f}<extra></extra>",
        ))
        tbml_fig.update_layout(
            **DARK_TEMPLATE,
            xaxis_title="Commodity", yaxis_title="Vendor Country",
            margin=dict(l=150, r=20, t=20, b=60),
        )
    else:
        tbml_fig = go.Figure()

    # ── 2. Benford's Law ─────────────────────────────────────────────────────
    bdf = compute_benford(dff)
    ben_fig = go.Figure([
        go.Bar(x=bdf["digit"], y=bdf["expected"], name="Expected (Benford's)",
               marker_color="#95a5a6", opacity=0.8),
        go.Bar(x=bdf["digit"], y=bdf["observed"], name="Observed (Ledger)",
               marker_color="#e67e22", opacity=0.85,
               hovertemplate="Digit %{x}<br>Observed: %{y:.2f}%<extra></extra>"),
    ])
    ben_fig.update_layout(
        **DARK_TEMPLATE,
        barmode="group",
        xaxis=dict(title="Leading Digit", tickmode="linear"),
        yaxis=dict(title="Frequency (%)"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=30, b=40),
    )

    # ── 3. Mahalanobis Scatter ───────────────────────────────────────────────
    dist = compute_mahalanobis_dash(dff)
    threshold = math.sqrt(9.348)
    x_vals = np.log1p(dff["Total_Value_USD"].clip(lower=0))
    y_vals = ((dff["Unit_Price_USD"] - dff["Market_Spot_Price"]).abs()
              / dff["Market_Spot_Price"].replace(0, np.nan) * 100).clip(upper=150).fillna(0)
    is_out = dist > threshold
    is_fr  = dff.get("Is_Fraud_Ground_Truth", pd.Series(0, index=dff.index)).fillna(0).astype(int)

    mahal_fig = go.Figure([
        go.Scattergl(
            x=x_vals[~is_out & (is_fr==0)], y=y_vals[~is_out & (is_fr==0)],
            mode="markers", name="Normal",
            marker=dict(color="#3498db", size=4, opacity=0.4),
        ),
        go.Scattergl(
            x=x_vals[is_out & (is_fr==0)], y=y_vals[is_out & (is_fr==0)],
            mode="markers", name=f"Mahalanobis Outlier ({is_out.sum():,})",
            marker=dict(color="#f39c12", size=8, opacity=0.8, symbol="diamond"),
        ),
        go.Scattergl(
            x=x_vals[is_fr==1], y=y_vals[is_fr==1],
            mode="markers", name=f"Fraud ({is_fr.sum():,})",
            marker=dict(color="#e74c3c", size=9, opacity=0.9, symbol="x"),
        ),
    ])
    mahal_fig.update_layout(
        **DARK_TEMPLATE,
        xaxis_title="log(Total Value USD)",
        yaxis_title="Price Deviation from Spot (%)",
        legend=dict(orientation="h", y=1.05, font=dict(size=10)),
        margin=dict(l=50, r=20, t=30, b=50),
    )

    # ── 4. Time Series ───────────────────────────────────────────────────────
    if "composite_risk_score" in dff.columns:
        ts = dff.groupby("month").agg(
            txn_count=("Transaction_ID","count"),
            avg_risk=("composite_risk_score","mean"),
        ).reset_index()
        ts_fig = make_subplots(specs=[[{"secondary_y": True}]])
        ts_fig.add_trace(go.Bar(
            x=ts["month"], y=ts["txn_count"],
            name="Transactions", marker_color="#3498db", opacity=0.7,
        ), secondary_y=False)
        ts_fig.add_trace(go.Scatter(
            x=ts["month"], y=ts["avg_risk"].round(2),
            name="Avg Risk Score", line=dict(color="#e74c3c", width=2.5),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=True)
        ts_fig.update_layout(
            **DARK_TEMPLATE,
            xaxis_title="Month",
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=50, r=50, t=30, b=60),
        )
        ts_fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        ts_fig.update_yaxes(title_text="Avg Risk Score", secondary_y=True)
    else:
        ts_fig = go.Figure()

    # ── 5. Fraud Pie ─────────────────────────────────────────────────────────
    if "Fraud_Type" in dff.columns:
        fraud_only = dff[dff.get("Is_Fraud_Ground_Truth", 0) == 1]
        fraud_counts = fraud_only["Fraud_Type"].value_counts().reset_index()
        fraud_counts.columns = ["Fraud_Type", "count"]
        pie_fig = go.Figure(go.Pie(
            labels=fraud_counts["Fraud_Type"],
            values=fraud_counts["count"],
            hole=0.45,
            marker=dict(colors=["#e74c3c","#f39c12","#3498db","#9b59b6"]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value}<extra></extra>",
        ))
        pie_fig.update_layout(
            **DARK_TEMPLATE,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
    else:
        pie_fig = go.Figure()

    # ── 6. Country Risk Bar ──────────────────────────────────────────────────
    if "composite_risk_score" in dff.columns:
        country_risk = dff.groupby("Vendor_Country")["composite_risk_score"].mean().sort_values().reset_index()
        colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 25 else "#27ae60"
                  for v in country_risk["composite_risk_score"]]
        cr_fig = go.Figure(go.Bar(
            x=country_risk["composite_risk_score"].round(2),
            y=country_risk["Vendor_Country"],
            orientation="h",
            marker_color=colors,
            text=country_risk["composite_risk_score"].round(1),
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        ))
        cr_fig.update_layout(
            **DARK_TEMPLATE,
            xaxis_title="Avg Composite Risk Score",
            margin=dict(l=160, r=50, t=20, b=40),
        )
    else:
        cr_fig = go.Figure()

    # ── 7. Vendor Table ──────────────────────────────────────────────────────
    score_avail = [c for c in ["ofac_risk_score","price_delta_risk_score",
                                "smurfing_risk_score","benfords_law_risk_score",
                                "geopolitical_risk_score","mahalanobis_risk_score",
                                "composite_risk_score"] if c in dff.columns]
    vendor_agg_cols = {c: "mean" for c in score_avail}
    vendor_agg_cols["Transaction_ID"] = "count"
    vendor_agg_cols["Total_Value_USD"] = "sum"

    vt = (
        dff.groupby(["Vendor_ID","Vendor_Country"])
        .agg(vendor_agg_cols)
        .reset_index()
        .rename(columns={
            "Transaction_ID": "txn_count",
            "Total_Value_USD": "total_value_usd",
        })
    )
    for c in score_avail:
        vt[c] = vt[c].round(2)
    vt["total_value_usd"] = vt["total_value_usd"].round(0).astype(int)

    if "composite_risk_score" in vt.columns:
        vt = vt.sort_values("composite_risk_score", ascending=False).head(50)

    if "composite_risk_score" in vt.columns:
        vt["risk_tier"] = vt["composite_risk_score"].apply(
            lambda s: "CRITICAL" if s >= 75 else "HIGH" if s >= 50 else "MEDIUM" if s >= 25 else "LOW"
        )

    display_cols = (["Vendor_ID","Vendor_Country","txn_count","total_value_usd"]
                    + score_avail
                    + (["risk_tier"] if "risk_tier" in vt.columns else []))
    display_cols = [c for c in display_cols if c in vt.columns]

    table_data = vt[display_cols].to_dict("records")
    table_cols = [{"name": c.replace("_", " ").title(), "id": c} for c in display_cols]

    return tbml_fig, ben_fig, mahal_fig, ts_fig, pie_fig, cr_fig, table_data, table_cols


@app.callback(
    Output("risk-dist-chart", "figure"),
    Input("score-dim-selector", "value"),
    Input("filter-country",     "value"),
    Input("filter-commodity",   "value"),
    Input("filter-tier",        "value"),
)
def update_dist(score_col, countries, commodities, tiers):
    dff = apply_filters(countries, commodities, tiers)
    if score_col not in dff.columns:
        return go.Figure()

    DARK_TEMPLATE = dict(
        paper_bgcolor="#1a2533",
        plot_bgcolor="#1a2533",
        font=dict(color="#ccd6f6", size=11),
    )

    label = SCORE_LABELS.get(score_col, score_col)
    fig   = go.Figure(go.Histogram(
        x=dff[score_col], nbinsx=60,
        marker=dict(
            color=dff[score_col],
            colorscale=[[0,"#27ae60"],[0.5,"#f39c12"],[1,"#e74c3c"]],
            showscale=False,
        ),
        opacity=0.85,
        hovertemplate="Score: %{x:.1f}<br>Count: %{y}<extra></extra>",
    ))
    mean_val = dff[score_col].mean()
    fig.add_vline(x=mean_val, line_color="#3498db", line_dash="dash",
                  annotation_text=f"Mean: {mean_val:.1f}", annotation_font_color="#3498db")
    fig.update_layout(
        **DARK_TEMPLATE,
        xaxis_title=f"{label} Score",
        yaxis_title="Transaction Count",
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


@app.callback(
    Output("model-metrics-chart", "figure"),
    Input("filter-country",  "value"),
    Input("filter-commodity","value"),
    Input("filter-tier",     "value"),
)
def update_model_metrics(countries, commodities, tiers):
    """
    Run a fast logistic regression on the filtered subset and display real metrics.
    Falls back to pre-trained estimates if sklearn is not available.
    """
    DARK_TEMPLATE = dict(
        paper_bgcolor="#1a2533",
        plot_bgcolor="#1a2533",
        font=dict(color="#ccd6f6", size=11),
    )

    dff = apply_filters(countries, commodities, tiers)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        if "Is_Fraud_Ground_Truth" not in dff.columns or dff["Is_Fraud_Ground_Truth"].sum() < 5:
            raise ValueError("Not enough fraud labels")

        feat_cols = [c for c in [
            "ofac_risk_score","price_delta_risk_score","smurfing_risk_score",
            "benfords_law_risk_score","geopolitical_risk_score","mahalanobis_risk_score",
        ] if c in dff.columns]

        if not feat_cols:
            for col in ["Commodity","Payment_Method","Vendor_Country"]:
                le = LabelEncoder()
                dff[f"enc_{col}"] = le.fit_transform(dff[col].fillna("Unknown"))
            feat_cols = [f"enc_{c}" for c in ["Commodity","Payment_Method","Vendor_Country"]]
            feat_cols += ["Volume_MT","Unit_Price_USD","Total_Value_USD"]

        X = dff[feat_cols].fillna(0)
        y = dff["Is_Fraud_Ground_Truth"].fillna(0).astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        model.fit(X_tr_s, y_tr)
        y_pred  = model.predict(X_te_s)
        y_prob  = model.predict_proba(X_te_s)[:, 1]

        metrics = {
            "Accuracy":  accuracy_score(y_te, y_pred) * 100,
            "Precision": precision_score(y_te, y_pred, zero_division=0) * 100,
            "Recall":    recall_score(y_te, y_pred, zero_division=0) * 100,
            "F1-Score":  f1_score(y_te, y_pred, zero_division=0) * 100,
            "ROC-AUC":   roc_auc_score(y_te, y_prob) * 100,
        }
        subtitle = f"Logistic Regression | n={len(X_te):,} test samples"
    except Exception as e:
        # Fallback: display a placeholder
        metrics = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "ROC-AUC": 0}
        subtitle = f"Insufficient data: {e}"

    colors = ["#2c3e50","#e74c3c","#27ae60","#2980b9","#9b59b6"]
    fig = go.Figure(go.Bar(
        x=list(metrics.keys()),
        y=[round(v, 2) for v in metrics.values()],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in metrics.values()],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **DARK_TEMPLATE,
        yaxis=dict(range=[0, 115], title="Score (%)"),
        xaxis_title="Metric",
        annotations=[dict(
            text=subtitle, x=0.5, y=-0.18, xref="paper", yref="paper",
            showarrow=False, font=dict(size=9, color="#7f8fa6"),
        )],
        margin=dict(l=40, r=20, t=30, b=60),
    )
    return fig


# export as html
def export_html(filename: str = "compliance_dashboard.html") -> None:
    """Generate a static HTML snapshot of the key charts."""
    print(f"  [Dashboard] Generating static HTML export …")
    bdf = compute_benford(df_global)
    dist = compute_mahalanobis_dash(df_global)
    threshold = math.sqrt(9.348)

    figs = {}

    # Benford
    ben_fig = go.Figure([
        go.Bar(x=bdf["digit"], y=bdf["expected"], name="Expected (Benford's)",
               marker_color="#95a5a6"),
        go.Bar(x=bdf["digit"], y=bdf["observed"], name="Observed (Ledger)",
               marker_color="#e67e22"),
    ])
    ben_fig.update_layout(title="Benford's Law - Invoice Leading Digits",
                           barmode="group", template="plotly_dark")
    figs["benfords"] = ben_fig

    # Mahalanobis
    x_v = np.log1p(df_global["Total_Value_USD"].clip(lower=0))
    y_v = ((df_global["Unit_Price_USD"] - df_global["Market_Spot_Price"]).abs()
           / df_global["Market_Spot_Price"].replace(0, np.nan) * 100).clip(upper=150).fillna(0)
    is_out = dist > threshold
    is_fr  = df_global.get("Is_Fraud_Ground_Truth", pd.Series(0)).fillna(0).astype(int)

    mah_fig = go.Figure([
        go.Scattergl(x=x_v[~is_out&(is_fr==0)], y=y_v[~is_out&(is_fr==0)],
                     mode="markers", name="Normal", marker=dict(color="#3498db", size=4, opacity=0.4)),
        go.Scattergl(x=x_v[is_out&(is_fr==0)], y=y_v[is_out&(is_fr==0)],
                     mode="markers", name="Mahalanobis Outlier",
                     marker=dict(color="#f39c12", size=8, symbol="diamond")),
        go.Scattergl(x=x_v[is_fr==1], y=y_v[is_fr==1],
                     mode="markers", name="Fraud",
                     marker=dict(color="#e74c3c", size=9, symbol="x")),
    ])
    mah_fig.update_layout(title="Mahalanobis Distance Outlier Detection (Real Transactions)",
                           template="plotly_dark",
                           xaxis_title="log(Total Value USD)",
                           yaxis_title="Price Deviation (%)")
    figs["mahalanobis"] = mah_fig

    # TBML Heatmap
    if "composite_risk_score" in df_global.columns:
        hp = (df_global.groupby(["Vendor_Country","Commodity"])["composite_risk_score"]
              .mean().reset_index()
              .pivot(index="Vendor_Country", columns="Commodity", values="composite_risk_score"))
        heat = go.Figure(go.Heatmap(
            z=hp.values, x=hp.columns.tolist(), y=hp.index.tolist(),
            colorscale="RdYlGn_r", colorbar=dict(title="Risk"),
        ))
        heat.update_layout(title="TBML Risk Heatmap – Country × Commodity", template="plotly_dark")
        figs["tbml"] = heat

    html_parts = ["<html><head><title>CCFRAD Dashboard</title></head><body style='background:#141e2a'>"]
    html_parts.append("<h1 style='color:#e8edf5;font-family:sans-serif;padding:20px'>"
                      "🛡️ Corporate Compliance | Financial Risk Dashboard</h1>")
    for name, fig in figs.items():
        html_parts.append(f"<h3 style='color:#aab;font-family:sans-serif;padding:10px 20px'>{name.upper()}</h3>")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append("</body></html>")

    with open(filename, "w") as f:
        f.write("\n".join(html_parts))
    print(f"  [Dashboard] Static export saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Corporate Compliance – Interactive Dashboard")
    print("=" * 65)
    export_html()
    print("\n  Starting Dash server …")
    print("  Open: http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)
