"""
worldbankapi.py
────────────────────────────────────────────────────────────────────────────
Corporate Compliance – World Bank Indicator Fetcher

Fetches geopolitical and economic risk indicators from the World Bank API
for all vendor countries present in the metallurgical ledger:

    Indicators fetched
    ──────────────────
    • LP.LPI.OVRL.XQ  – Logistics Performance Index (overall score)
    • NY.GDP.PCAP.CD   – GDP per capita (current USD)
    • GE.EST           – Government Effectiveness (World Governance Indicator)
    • CC.EST           – Control of Corruption (WGI)
    • FP.CPI.TOTL.ZG   – Inflation rate (CPI % change)

Outputs
───────
    • pandas DataFrame with all indicators per country
    • CSV export: worldbank_risk_indicators.csv
    • Risk scores (0–100) derived from each indicator for use in the
      composite compliance score
"""

import time
import warnings

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
WB_BASE_URL   = "https://api.worldbank.org/v2"
REQUEST_DELAY = 0.5
TIMEOUT       = 30

INDICATORS = {
    "LP.LPI.OVRL.XQ": {
        "name":      "Logistics Performance Index",
        "direction": "lower_worse",     # higher = better logistics = lower risk
        "scale":     (1.0, 5.0),
    },
    "NY.GDP.PCAP.CD": {
        "name":      "GDP per capita (USD)",
        "direction": "lower_worse",
        "scale":     (0.0, 80_000.0),   # log-normalised
        "log_scale": True,
    },
    "GE.EST": {
        "name":      "Government Effectiveness",
        "direction": "lower_worse",     # WGI: typically -2.5 to +2.5
        "scale":     (-2.5, 2.5),
    },
    "CC.EST": {
        "name":      "Control of Corruption",
        "direction": "lower_worse",
        "scale":     (-2.5, 2.5),
    },
    "FP.CPI.TOTL.ZG": {
        "name":      "Inflation Rate (%)",
        "direction": "higher_worse",    # high inflation → higher risk
        "scale":     (0.0, 50.0),
    },
}

# All vendor countries → ISO2 codes
COUNTRY_ISO2 = {
    "Australia":              "AU",
    "Canada":                 "CA",
    "Germany":                "DE",
    "Japan":                  "JP",
    "Singapore":              "SG",
    "Switzerland":            "CH",
    "United Arab Emirates":   "AE",
    "United States":          "US",
}


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────
def fetch_indicator(indicator_code: str, iso2_codes: list[str]) -> dict:
    """
    Fetch the most-recent value for `indicator_code` for a list of ISO2 codes.
    Returns { iso2: value } dict.
    """
    # Batch all countries in one request
    country_str = ";".join(iso2_codes)
    url = f"{WB_BASE_URL}/country/{country_str}/indicator/{indicator_code}"
    params = {"format": "json", "per_page": 500, "mrv": 3}

    results = {}
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            payload = resp.json()
            if len(payload) > 1 and payload[1]:
                for rec in payload[1]:
                    iso2 = rec.get("country", {}).get("id", "")
                    val  = rec.get("value")
                    if iso2 and val is not None and iso2 not in results:
                        try:
                            results[iso2] = float(val)
                        except (TypeError, ValueError):
                            pass
        else:
            print(f"    HTTP {resp.status_code} for {indicator_code}")
    except Exception as exc:
        print(f"    Error fetching {indicator_code}: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Risk score derivation
# ─────────────────────────────────────────────────────────────────────────────
def _derive_risk_score(value: float, cfg: dict) -> float:
    """
    Normalise an indicator value to a 0–100 risk score.
    0 = lowest risk, 100 = highest risk.
    """
    import math
    lo, hi = cfg["scale"]
    direction = cfg["direction"]

    if cfg.get("log_scale"):
        # Log-normalise
        norm = math.log10(max(value, 1)) / math.log10(hi) if hi > 0 else 0.5
    else:
        norm = (value - lo) / (hi - lo) if hi != lo else 0.5

    norm = max(0.0, min(1.0, norm))

    if direction == "lower_worse":
        # Low value → high risk
        risk = 1.0 - norm
    else:
        # High value → high risk
        risk = norm

    return round(risk * 100.0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main fetch function
# ─────────────────────────────────────────────────────────────────────────────
def fetch_all_indicators() -> pd.DataFrame:
    """
    Fetch all indicators for all vendor countries.
    Returns a pivoted DataFrame: rows = countries, columns = indicators.
    """
    iso2_list = list(COUNTRY_ISO2.values())
    all_data  = {}

    for code, cfg in INDICATORS.items():
        print(f"  [WorldBank] {cfg['name']} ({code}) …")
        values = fetch_indicator(code, iso2_list)
        all_data[code] = values
        time.sleep(REQUEST_DELAY)

    # Build wide DataFrame
    rows = []
    for country, iso2 in COUNTRY_ISO2.items():
        row = {"country": country, "iso2": iso2}
        composite_risk = 0.0
        n_valid = 0
        for code, cfg in INDICATORS.items():
            val  = all_data.get(code, {}).get(iso2)
            row[cfg["name"]] = round(val, 4) if val is not None else None
            if val is not None:
                row[f"{cfg['name']}_risk"] = _derive_risk_score(val, cfg)
                composite_risk += row[f"{cfg['name']}_risk"]
                n_valid += 1
        row["composite_wb_risk"] = round(composite_risk / max(n_valid, 1), 2)
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def plot_country_risk(df: pd.DataFrame, filename: str = "worldbank_risk.png") -> None:
    """Horizontal bar chart of composite World Bank risk per country."""
    plt.style.use("ggplot")

    df_sorted = df.sort_values("composite_wb_risk", ascending=True)
    colors = ["#e74c3c" if r > 50 else "#f39c12" if r > 30 else "#27ae60"
              for r in df_sorted["composite_wb_risk"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_sorted["country"], df_sorted["composite_wb_risk"],
                   color=colors, edgecolor="white")
    ax.set_xlabel("Composite WB Risk Score (0–100)", fontsize=11)
    ax.set_title("World Bank Geopolitical Risk by Vendor Country\n"
                 "(LPI + GDP + Governance + Corruption + Inflation)", fontweight="bold")
    ax.axvline(50, color="#e74c3c", linestyle="--", alpha=0.5, label="Risk threshold = 50")
    for bar, val in zip(bars, df_sorted["composite_wb_risk"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> pd.DataFrame:
    DIVIDER = "=" * 65
    print(DIVIDER)
    print("  World Bank API – Geopolitical Risk Indicator Fetcher")
    print(DIVIDER)

    print("\n[STEP 1/3]  Fetching World Bank indicators …")
    df = fetch_all_indicators()

    print("\n[STEP 2/3]  Results summary:")
    display_cols = ["country", "Logistics Performance Index",
                    "GDP per capita (USD)", "Government Effectiveness",
                    "Control of Corruption", "composite_wb_risk"]
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))

    print("\n[STEP 3/3]  Exporting …")
    df.to_csv("worldbank_risk_indicators.csv", index=False)
    print("  Exported → worldbank_risk_indicators.csv")
    plot_country_risk(df)

    print(f"\n✓ World Bank fetch complete.\n")
    return df


if __name__ == "__main__":
    main()