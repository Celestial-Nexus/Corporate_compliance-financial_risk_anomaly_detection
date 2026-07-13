import math
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

INPUT_FILE  = "metallurgical_ledgers.xlsx"
OUTPUT_FILE = "metallurgical_ledgers_scored.xlsx"

OFAC_SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"

SANCTIONED_JURISDICTIONS = {
    "cuba", "iran", "north korea", "dprk", "syria", "russia",
    "myanmar", "burma", "belarus", "venezuela", "zimbabwe",
    "somalia", "south sudan", "sudan", "libya", "mali",
    "nicaragua", "yemen", "eritrea",
    "sanctioned_proxy_alpha", "sanctioned_proxy_beta",
}

CTR_THRESHOLD_USD = 10_000

WB_BASE_URL      = "https://api.worldbank.org/v2"
WB_LPI_INDICATOR = "LP.LPI.OVRL.XQ"
WB_GDP_INDICATOR = "NY.GDP.PCAP.CD"

COUNTRY_ISO2 = {
    "Australia":              "AU",
    "Canada":                 "CA",
    "Germany":                "DE",
    "Japan":                  "JP",
    "Singapore":              "SG",
    "Switzerland":            "CH",
    "United Arab Emirates":   "AE",
    "United States":          "US",
    "Sanctioned_Proxy_Alpha": None,
    "Sanctioned_Proxy_Beta":  None,
}

COMTRADE_URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"

COUNTRY_M49 = {
    "Australia":             36,
    "Canada":               124,
    "Germany":              276,
    "Japan":                392,
    "Singapore":            702,
    "Switzerland":          756,
    "United Arab Emirates": 784,
    "United States":        840,
}

COMMODITY_HS_CHAPTERS = {
    "Gold_Bullion":   "71",
    "Silver_Ingot":   "71",
    "Copper_Cathode": "74",
    "Aluminum_Ingot": "76",
}

BENFORD_EXPECTED = {d: math.log10(1.0 + 1.0 / d) for d in range(1, 10)}


def _normalise_country(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def fetch_ofac_sanctioned_set() -> set:
    sanctioned = {_normalise_country(c) for c in SANCTIONED_JURISDICTIONS}
    print("  [OFAC] Downloading SDN list from U.S. Treasury …")
    try:
        resp = requests.get(OFAC_SDN_URL, timeout=30)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                parts = line.split(",")
                if len(parts) > 11:
                    remarks = parts[11].lower()
                    for marker in ["nationality:", "country:", "citizen of "]:
                        if marker in remarks:
                            idx   = remarks.find(marker) + len(marker)
                            token = remarks[idx:idx + 40].strip().split(";")[0].strip().rstrip(".")
                            if token:
                                sanctioned.add(_normalise_country(token))
            print(f"  [OFAC] SDN list loaded. Total sanctioned tokens: {len(sanctioned)}")
        else:
            print(f"  [OFAC] HTTP {resp.status_code} – using hardcoded list only.")
    except Exception as exc:
        print(f"  [OFAC] Download failed ({exc}) – using hardcoded list only.")
    return sanctioned


def score_ofac(df: pd.DataFrame, sanctioned_set: set) -> pd.Series:
    def _flag(country: str) -> int:
        norm = _normalise_country(country)
        if norm in sanctioned_set or "sanctioned" in norm:
            return 100
        for s in sanctioned_set:
            if len(s) > 4 and s in norm:
                return 100
        return 0
    return df["Vendor_Country"].apply(_flag).astype(float)


def score_price_delta(df: pd.DataFrame) -> pd.Series:
    delta_pct = (
        (df["Unit_Price_USD"] - df["Market_Spot_Price"]).abs()
        / df["Market_Spot_Price"].replace(0, np.nan)
        * 100
    )
    return delta_pct.clip(upper=100.0).round(4)


def score_smurfing(df: pd.DataFrame) -> pd.Series:
    df_work = df.copy()
    df_work["Date"] = pd.to_datetime(df_work["Date"])

    vendor_stats: dict = {}
    for vendor_id, grp in df_work.groupby("Vendor_ID"):
        n_tx    = len(grp)
        span    = max(1, (grp["Date"].max() - grp["Date"].min()).days + 1)
        tx_rate = n_tx / span

        near = grp[
            (grp["Total_Value_USD"] >= CTR_THRESHOLD_USD * 0.75) &
            (grp["Total_Value_USD"] <  CTR_THRESHOLD_USD)
        ]
        near_frac = len(near) / n_tx
        avg_val   = grp["Total_Value_USD"].mean()

        vendor_stats[vendor_id] = {
            "tx_rate":   tx_rate,
            "near_frac": near_frac,
            "avg_val":   avg_val,
        }

    stats_df = pd.DataFrame(vendor_stats).T
    mu_freq  = stats_df["tx_rate"].mean()
    sig_freq = stats_df["tx_rate"].std(ddof=1) + 1e-9

    vendor_scores: dict = {}
    for vendor_id, s in vendor_stats.items():
        z_freq     = (s["tx_rate"] - mu_freq) / sig_freq
        freq_score = min(100.0, max(0.0, z_freq / 3.0 * 100.0))

        prox_score = min(100.0, s["near_frac"] * 500.0)

        if s["avg_val"] > 0:
            log_ratio = math.log10(s["avg_val"] + 1) / math.log10(CTR_THRESHOLD_USD + 1)
            val_score = (1.0 - min(1.0, log_ratio)) * 100.0
        else:
            val_score = 100.0

        final = 0.50 * freq_score + 0.30 * prox_score + 0.20 * val_score
        vendor_scores[vendor_id] = round(min(100.0, final), 4)

    return df_work["Vendor_ID"].map(vendor_scores)


def _leading_digit(value) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value <= 0:
        return None
    s        = f"{value:.10e}"
    mantissa = s.split("e")[0]
    for ch in mantissa.replace(".", "").lstrip("0"):
        if ch.isdigit() and ch != "0":
            return int(ch)
    return None


def score_benfords_law(df: pd.DataFrame) -> pd.Series:
    df_work = df.copy()
    df_work["_ld"] = df_work["Total_Value_USD"].apply(_leading_digit)

    vendor_scores: dict = {}
    for vendor_id, grp in df_work.groupby("Vendor_ID"):
        digits = grp["_ld"].dropna()
        total  = len(digits)

        if total < 5:
            vendor_scores[vendor_id] = 50.0
            continue

        observed = {d: (digits == d).sum() / total for d in range(1, 10)}
        mad      = sum(abs(observed[d] - BENFORD_EXPECTED[d]) for d in range(1, 10)) / 9.0
        vendor_scores[vendor_id] = round(min(100.0, mad / 0.15 * 100.0), 4)

    return df_work["Vendor_ID"].map(vendor_scores)


def _wb_fetch_indicator(indicator: str) -> dict:
    url    = f"{WB_BASE_URL}/country/all/indicator/{indicator}"
    params = {"format": "json", "per_page": 1000, "mrv": 1}
    out    = {}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            payload = resp.json()
            if len(payload) > 1 and payload[1]:
                for rec in payload[1]:
                    iso2 = rec.get("country", {}).get("id", "")
                    val  = rec.get("value")
                    if iso2 and val is not None:
                        try:
                            out[iso2] = float(val)
                        except (TypeError, ValueError):
                            pass
        else:
            print(f"  [WorldBank] HTTP {resp.status_code} for {indicator}")
    except Exception as exc:
        print(f"  [WorldBank] Failed to fetch {indicator}: {exc}")
    return out


def _comtrade_fetch_value(partner_m49: int, hs_chapter: str) -> float | None:
    params = {
        "reporterCode": 842,
        "partnerCode":  partner_m49,
        "period":       2023,
        "flowCode":     "M",
        "cmdCode":      hs_chapter,
    }
    try:
        resp = requests.get(COMTRADE_URL, params=params, timeout=30)
        if resp.status_code == 200:
            records = resp.json().get("data", [])
            if records:
                return float(sum(r.get("primaryValue", 0) or 0 for r in records))
    except Exception as exc:
        print(f"  [Comtrade] Request error (partner={partner_m49}, HS={hs_chapter}): {exc}")
    return None


def score_geopolitical(df: pd.DataFrame) -> pd.Series:
    print("  [WorldBank] Fetching LPI …")
    lpi_by_iso2 = _wb_fetch_indicator(WB_LPI_INDICATOR)
    time.sleep(1)

    print("  [WorldBank] Fetching GDP per capita …")
    gdp_by_iso2 = _wb_fetch_indicator(WB_GDP_INDICATOR)
    time.sleep(1)

    unique_hs = set(COMMODITY_HS_CHAPTERS.values())

    country_comtrade: dict = {}
    print("  [Comtrade] Fetching trade flows per country …")
    for country, m49 in COUNTRY_M49.items():
        total = 0.0
        for hs in unique_hs:
            val = _comtrade_fetch_value(m49, hs)
            if val is not None:
                total += val
            time.sleep(0.6)
        country_comtrade[country] = total if total > 0 else None
        label = f"${total:,.0f}" if total > 0 else "no data"
        print(f"    {country}: Comtrade official import = {label}")

    country_ledger_total = df.groupby("Vendor_Country")["Total_Value_USD"].sum().to_dict()

    country_scores: dict = {}
    for country, iso2 in COUNTRY_ISO2.items():
        if iso2 is None or "sanctioned" in _normalise_country(country):
            country_scores[country] = 100.0
            continue

        lpi_val  = lpi_by_iso2.get(iso2)
        lpi_risk = ((5.0 - lpi_val) / 4.0) if lpi_val is not None else 0.5
        lpi_risk = max(0.0, min(1.0, lpi_risk))

        gdp_val = gdp_by_iso2.get(iso2)
        if gdp_val and gdp_val > 0:
            gdp_risk = 1.0 - min(1.0, math.log10(gdp_val + 1) / math.log10(80_000))
        else:
            gdp_risk = 0.5

        ledger_val   = country_ledger_total.get(country, 0.0)
        comtrade_val = country_comtrade.get(country)

        if comtrade_val and comtrade_val > 0 and ledger_val > 0:
            share         = ledger_val / comtrade_val
            comtrade_risk = min(1.0, max(0.0, math.log10(share + 0.01) / 2.0 + 0.5))
        else:
            comtrade_risk = 0.5

        geo_raw = 0.40 * lpi_risk + 0.40 * gdp_risk + 0.20 * comtrade_risk
        country_scores[country] = round(min(100.0, geo_raw * 100.0), 4)

    return df["Vendor_Country"].map(country_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Mahalanobis Distance (real implementation on actual transaction features)
# ─────────────────────────────────────────────────────────────────────────────
MAHAL_FEATURES = ["Volume_MT", "Unit_Price_USD", "Total_Value_USD"]
# Chi-squared threshold for 3-feature space at p=0.975
MAHAL_CHI2_THRESH = 9.348  # scipy.stats.chi2.ppf(0.975, df=3)


def _mahal_distance(X: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """Vectorised Mahalanobis distances for all rows of X."""
    diff = X - mu
    # (n,d) @ (d,d) → (n,d) then element-wise * diff → sum per row
    left = diff @ cov_inv
    sq   = (left * diff).sum(axis=1)
    return np.sqrt(np.maximum(sq, 0.0))


def score_mahalanobis(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute Mahalanobis distance for each transaction using three numeric features.
    Returns:
        dist_series  – raw Mahalanobis distance per row
        score_series – 0–100 risk score (distance relative to chi-sq threshold)
    """
    X = df[MAHAL_FEATURES].fillna(df[MAHAL_FEATURES].median()).values.astype(float)
    mu      = X.mean(axis=0)
    cov     = np.cov(X, rowvar=False)
    cov_inv = np.linalg.pinv(cov)

    distances = _mahal_distance(X, mu, cov_inv)
    threshold = math.sqrt(MAHAL_CHI2_THRESH)

    # Normalise: transactions beyond threshold score toward 100
    scores = np.clip(distances / threshold * 100.0, 0.0, 100.0)

    dist_series  = pd.Series(distances, index=df.index, name="mahalanobis_distance")
    score_series = pd.Series(np.round(scores, 4), index=df.index, name="mahalanobis_risk_score")
    return dist_series, score_series


def plot_mahalanobis_scatter(df: pd.DataFrame, dist_series: pd.Series,
                             threshold: float,
                             filename: str = "mahalanobis_outliers.png") -> None:
    """Regenerate mahalanobis_outliers.png from real transaction data."""
    plt.style.use("ggplot")

    x = np.log1p(df["Total_Value_USD"].clip(lower=0))
    y = (
        (df["Unit_Price_USD"] - df["Market_Spot_Price"]).abs()
        / df["Market_Spot_Price"].replace(0, np.nan) * 100.0
    ).clip(upper=200).fillna(0)

    is_outlier = dist_series > threshold
    is_fraud   = df.get("Is_Fraud_Ground_Truth", pd.Series(0, index=df.index)).fillna(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Mahalanobis Distance – Multivariate Outlier Detection\n"
                 "(Real Metallurgical Ledger – 15,030 Transactions)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Distance histogram
    ax0 = axes[0]
    ax0.hist(dist_series, bins=80, color="#3498db", edgecolor="white", alpha=0.75)
    ax0.axvline(threshold, color="#e74c3c", linewidth=2.5, linestyle="--",
                label=f"χ² threshold = {threshold:.2f}")
    ax0.set_xlabel("Mahalanobis Distance", fontsize=11)
    ax0.set_ylabel("Transaction Count", fontsize=11)
    ax0.set_title("Distance Distribution", fontweight="bold")
    ax0.legend()
    n_out = int(is_outlier.sum())
    ax0.text(threshold * 1.05, ax0.get_ylim()[1] * 0.7,
             f"{n_out:,} outliers\n({n_out/len(df)*100:.1f}%)",
             color="#e74c3c", fontsize=9, fontweight="bold")

    # Panel 2: Feature-space scatter
    ax1 = axes[1]
    mask_clean = (~is_outlier) & (is_fraud == 0)
    mask_out   = is_outlier   & (is_fraud == 0)
    mask_fraud = is_fraud == 1

    ax1.scatter(x[mask_clean],  y[mask_clean],  c="#3498db", alpha=0.35, s=10,
                label="Normal", edgecolors="none")
    ax1.scatter(x[mask_out],    y[mask_out],    c="#f39c12", alpha=0.80, s=45,
                marker="D", label=f"Mahalanobis Outlier ({n_out:,})")
    ax1.scatter(x[mask_fraud],  y[mask_fraud],  c="#e74c3c", alpha=0.90, s=55,
                marker="X", label=f"Ground-Truth Fraud ({is_fraud.sum():,})")

    ax1.set_xlabel("log(Total Value USD)", fontsize=11)
    ax1.set_ylabel("Price Deviation from Spot (%)", fontsize=11)
    ax1.set_title("Transaction Feature Space", fontweight="bold")
    ax1.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"            Saved → {filename}")


def main() -> pd.DataFrame:
    DIVIDER = "=" * 65

    print(DIVIDER)
    print("  Corporate Compliance – Financial Risk Anomaly Detection")
    print(DIVIDER)

    print(f"\n[STEP 1/8]  Loading '{INPUT_FILE}' …")
    df = pd.read_excel(INPUT_FILE)

    # Re-attach ground-truth columns if they were dropped
    if "Is_Fraud_Ground_Truth" not in df.columns or "Fraud_Type" not in df.columns:
        try:
            raw = pd.read_csv("metallurgical_ledgers.csv")
            for col in ["Is_Fraud_Ground_Truth", "Fraud_Type"]:
                if col not in df.columns and col in raw.columns:
                    df = df.merge(raw[["Transaction_ID", col]], on="Transaction_ID", how="left")
        except FileNotFoundError:
            pass

    print(f"            {len(df):,} transactions | {df['Vendor_ID'].nunique()} unique vendors")

    print("\n[STEP 2/8]  OFAC List Cross-Reference …")
    sanctioned_set = fetch_ofac_sanctioned_set()
    df["ofac_risk_score"] = score_ofac(df, sanctioned_set)
    n_flagged = int((df["ofac_risk_score"] == 100).sum())
    print(f"            Flagged transactions: {n_flagged:,} ({n_flagged / len(df) * 100:.1f} %)")

    print("\n[STEP 3/8]  Price Delta (Trade-Based Arbitrage) …")
    df["price_delta_risk_score"] = score_price_delta(df)
    print(f"            Mean = {df['price_delta_risk_score'].mean():.2f}  Max = {df['price_delta_risk_score'].max():.2f}")

    print("\n[STEP 4/8]  Smurfing & Structuring (Volume Anomalies) …")
    df["smurfing_risk_score"] = score_smurfing(df)
    print(f"            Mean = {df['smurfing_risk_score'].mean():.2f}  Max = {df['smurfing_risk_score'].max():.2f}")

    print("\n[STEP 5/8]  Benford's Law Deviation …")
    df["benfords_law_risk_score"] = score_benfords_law(df)
    print(f"            Mean = {df['benfords_law_risk_score'].mean():.2f}  Max = {df['benfords_law_risk_score'].max():.2f}")

    print("\n[STEP 6/8]  Geopolitical Risk (World Bank + Comtrade) …")
    df["geopolitical_risk_score"] = score_geopolitical(df)
    print(f"            Mean = {df['geopolitical_risk_score'].mean():.2f}  Max = {df['geopolitical_risk_score'].max():.2f}")

    print("\n[STEP 7/8]  Mahalanobis Distance Anomaly Detection …")
    dist_series, mahal_score = score_mahalanobis(df)
    df["mahalanobis_risk_score"] = mahal_score
    threshold = math.sqrt(MAHAL_CHI2_THRESH)
    n_out = int((dist_series > threshold).sum())
    print(f"            Threshold : {threshold:.4f} (χ² 97.5th pct, 3 features)")
    print(f"            Outliers  : {n_out:,} ({n_out / len(df) * 100:.2f}%)")
    print(f"            Mean score= {mahal_score.mean():.2f}  Max = {mahal_score.max():.2f}")
    plot_mahalanobis_scatter(df, dist_series, threshold)

    # ── Composite risk score (weighted sum) ─────────────────────────────────
    score_cols = [
        "ofac_risk_score",
        "price_delta_risk_score",
        "smurfing_risk_score",
        "benfords_law_risk_score",
        "geopolitical_risk_score",
        "mahalanobis_risk_score",
    ]
    WEIGHTS = {
        "ofac_risk_score":          0.30,
        "price_delta_risk_score":   0.20,
        "smurfing_risk_score":      0.15,
        "benfords_law_risk_score":  0.10,
        "geopolitical_risk_score":  0.15,
        "mahalanobis_risk_score":   0.10,
    }
    df["composite_risk_score"] = sum(
        df[col] * w for col, w in WEIGHTS.items() if col in df.columns
    ).round(4)
    df["risk_tier"] = df["composite_risk_score"].apply(
        lambda s: "CRITICAL" if s >= 75 else
                  "HIGH"     if s >= 50 else
                  "MEDIUM"   if s >= 25 else "LOW"
    )

    print(f"\n[STEP 8/8]  Writing '{OUTPUT_FILE}' …")
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"            Saved {len(df):,} rows × {len(df.columns)} columns.")

    print("\n" + DIVIDER)
    print("  RISK SCORE DISTRIBUTION SUMMARY")
    print(DIVIDER)
    all_score_cols = score_cols + ["composite_risk_score"]
    summary = df[all_score_cols].describe().round(2)
    summary.index.name = "Statistic"
    print(summary.to_string())

    print("\n" + DIVIDER)
    print("  RISK TIER DISTRIBUTION")
    print(DIVIDER)
    tier_counts = df["risk_tier"].value_counts()
    for tier, cnt in tier_counts.items():
        print(f"    {tier:<10}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    print("\n" + DIVIDER)
    print("  TOP-10 HIGHEST-RISK TRANSACTIONS (composite score)")
    print(DIVIDER)
    top10 = df.nlargest(10, "composite_risk_score")[
        ["Transaction_ID", "Date", "Vendor_ID", "Vendor_Country", "Fraud_Type",
         "composite_risk_score", "risk_tier"] + score_cols
    ]
    print(top10.to_string(index=False))

    print(f"\n✓ Done. Scored workbook saved to: {OUTPUT_FILE}\n")
    return df


if __name__ == "__main__":
    main()
