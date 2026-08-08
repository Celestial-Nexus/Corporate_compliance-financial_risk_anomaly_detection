# PowerBI Dashboard — Import Guide

The scored transaction data is structured to work directly with PowerBI Desktop. This guide walks through how to import the data and build the key compliance dashboards.

## Files to import

| File | Description | How to load |
|------|-------------|-------------|
| `outputs/metallurgical_ledgers_scored.xlsx` | Main scored dataset — 15,030 rows, 20 columns | Get Data → Excel |
| `outputs/tbml_alerts.csv` | Vendor-level TBML risk summary | Get Data → CSV |
| `outputs/country_risk_summary.csv` | Country-level aggregated stats | Get Data → CSV |
| `outputs/commodity_anomalies.csv` | Price anomaly z-scores | Get Data → CSV |

## Column reference for the scored workbook

| Column | Type | Description |
|--------|------|-------------|
| `Transaction_ID` | Text | Unique transaction identifier |
| `Date` | Date | Transaction date |
| `Vendor_ID` | Text | Vendor identifier |
| `Vendor_Country` | Text | Vendor's country of origin |
| `Commodity` | Text | Gold_Bullion, Silver_Ingot, Copper_Cathode, or Aluminum_Ingot |
| `Volume_MT` | Number | Transaction volume in metric tons |
| `Unit_Price_USD` | Number | Invoice unit price |
| `Market_Spot_Price` | Number | Market benchmark price |
| `Total_Value_USD` | Number | Total invoice value |
| `Payment_Method` | Text | Wire, Letter_of_Credit, or Open_Account |
| `ofac_risk_score` | Number (0–100) | OFAC sanctions match score |
| `price_delta_risk_score` | Number (0–100) | Price deviation from market |
| `smurfing_risk_score` | Number (0–100) | Structuring pattern score |
| `benfords_law_risk_score` | Number (0–100) | Benford's digit anomaly score |
| `geopolitical_risk_score` | Number (0–100) | Country risk (World Bank + Comtrade) |
| `mahalanobis_risk_score` | Number (0–100) | Multivariate outlier score |
| `composite_risk_score` | Number (0–100) | Weighted combination of all scores |
| `risk_tier` | Text | CRITICAL / HIGH / MEDIUM / LOW |
| `Is_Fraud_Ground_Truth` | Number (0/1) | Ground truth fraud label |
| `Fraud_Type` | Text | Type of fraud (if any) |

## Suggested visuals

### 1. KPI cards
- Total transaction count
- Total volume (sum of `Total_Value_USD`, formatted as $B)
- Fraud count (count where `Is_Fraud_Ground_Truth` = 1)
- Average `composite_risk_score`
- Count where `risk_tier` = "CRITICAL"

### 2. TBML risk heatmap
- Visual: **Matrix**
- Rows: `Vendor_Country`
- Columns: `Commodity`
- Values: Average `composite_risk_score`
- Conditional formatting: green → yellow → red

### 3. Benford's Law bar chart
- You'll need a DAX measure for this — extract leading digit from `Total_Value_USD`
- Visual: **Clustered bar chart**
- X-axis: Leading digit (1–9)
- Y-axis: Observed frequency %
- Add a reference line for expected Benford frequency

### 4. Risk score distribution
- Visual: **Histogram** or **Stacked bar chart**
- Group by `risk_tier` (CRITICAL, HIGH, MEDIUM, LOW)
- Or use a slicer to pick which risk dimension to display

### 5. Country risk bar chart
- Visual: **Bar chart**
- X-axis: `Vendor_Country`
- Y-axis: Average `geopolitical_risk_score`
- Conditional coloring by score range

### 6. Monthly trend line
- Visual: **Line chart**
- X-axis: `Date` (by month)
- Y-axis 1: Count of transactions
- Y-axis 2: Average `composite_risk_score`

### 7. Fraud type breakdown
- Visual: **Donut chart**
- Legend: `Fraud_Type`
- Values: Count of transactions

### 8. Top vendor risk table
- Visual: **Table**
- Columns: `Vendor_ID`, `Vendor_Country`, avg `composite_risk_score`, transaction count, `risk_tier`
- Sort by avg composite score descending
- Conditional formatting on score column

## Note

An interactive Plotly Dash equivalent of these dashboards is also available — just run `python src/dashboard.py` and open `http://127.0.0.1:8050`.
