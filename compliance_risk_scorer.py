import math
import time
import warnings

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


def main() -> None:
    DIVIDER = "=" * 65

    print(DIVIDER)
    print("  Corporate Compliance – Financial Risk Anomaly Detection")
    print(DIVIDER)

    print(f"\n[STEP 1/7]  Loading '{INPUT_FILE}' …")
    df = pd.read_excel(INPUT_FILE)
    print(f"            {len(df):,} transactions | {df['Vendor_ID'].nunique()} unique vendors")

    print("\n[STEP 2/7]  OFAC List Cross-Reference …")
    sanctioned_set = fetch_ofac_sanctioned_set()
    df["ofac_risk_score"] = score_ofac(df, sanctioned_set)
    n_flagged = int((df["ofac_risk_score"] == 100).sum())
    print(f"            Flagged transactions: {n_flagged:,} ({n_flagged / len(df) * 100:.1f} %)")

    print("\n[STEP 3/7]  Price Delta (Trade-Based Arbitrage) …")
    df["price_delta_risk_score"] = score_price_delta(df)
    print(f"            Mean = {df['price_delta_risk_score'].mean():.2f}  Max = {df['price_delta_risk_score'].max():.2f}")

    print("\n[STEP 4/7]  Smurfing & Structuring (Volume Anomalies) …")
    df["smurfing_risk_score"] = score_smurfing(df)
    print(f"            Mean = {df['smurfing_risk_score'].mean():.2f}  Max = {df['smurfing_risk_score'].max():.2f}")

    print("\n[STEP 5/7]  Benford's Law Deviation …")
    df["benfords_law_risk_score"] = score_benfords_law(df)
    print(f"            Mean = {df['benfords_law_risk_score'].mean():.2f}  Max = {df['benfords_law_risk_score'].max():.2f}")

    print("\n[STEP 6/7]  Geopolitical Risk (World Bank + Comtrade) …")
    df["geopolitical_risk_score"] = score_geopolitical(df)
    print(f"            Mean = {df['geopolitical_risk_score'].mean():.2f}  Max = {df['geopolitical_risk_score'].max():.2f}")

    print(f"\n[STEP 7/7]  Writing '{OUTPUT_FILE}' …")
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"            Saved {len(df):,} rows × {len(df.columns)} columns.")

    score_cols = [
        "ofac_risk_score",
        "price_delta_risk_score",
        "smurfing_risk_score",
        "benfords_law_risk_score",
        "geopolitical_risk_score",
    ]

    print("\n" + DIVIDER)
    print("  RISK SCORE DISTRIBUTION SUMMARY")
    print(DIVIDER)
    summary = df[score_cols].describe().round(2)
    summary.index.name = "Statistic"
    print(summary.to_string())

    print("\n" + DIVIDER)
    print("  TOP-10 HIGHEST-RISK TRANSACTIONS (by sum of all scores)")
    print(DIVIDER)
    df["_total_risk"] = df[score_cols].sum(axis=1)
    top10 = df.nlargest(10, "_total_risk")[
        ["Transaction_ID", "Date", "Vendor_ID", "Vendor_Country"] + score_cols + ["_total_risk"]
    ]
    print(top10.to_string(index=False))
    df.drop(columns=["_total_risk"], inplace=True)

    print(f"\n✓ Done. Scored workbook saved to: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
