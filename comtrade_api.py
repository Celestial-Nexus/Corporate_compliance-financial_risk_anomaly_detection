"""
comtrade_api.py
────────────────────────────────────────────────────────────────────────────
Corporate Compliance – UN Comtrade Trade Flow Fetcher

Fetches official import/export trade flow data from the UN Comtrade public
API for the commodities present in the metallurgical ledger:
    • HS Chapter 71 – Gold, Silver, precious metals
    • HS Chapter 74 – Copper and articles thereof
    • HS Chapter 76 – Aluminum and articles thereof

Country coverage matches the ledger's vendor countries.
Results are used to validate whether ledger transaction volumes are
plausible relative to official bilateral trade statistics (TBML check).
"""

import time
import warnings

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL     = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
REQUEST_DELAY = 0.8          # seconds between requests (rate-limit friendly)
TIMEOUT       = 30

# UN M49 reporter/partner codes
REPORTER_USA = 842           # United States (reporter for bilateral flows)

# All vendor countries in the ledger → M49 codes
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

# HS chapters relevant to the ledger commodities
HS_CHAPTERS = {
    "71": "Precious Metals (Gold / Silver)",
    "74": "Copper & Articles",
    "76": "Aluminum & Articles",
}

# Commodity → HS chapter mapping
COMMODITY_HS = {
    "Gold_Bullion":   "71",
    "Silver_Ingot":   "71",
    "Copper_Cathode": "74",
    "Aluminum_Ingot": "76",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core fetcher
# ─────────────────────────────────────────────────────────────────────────────
def fetch_trade_flow(
    reporter_code: int,
    partner_code:  int,
    period:        int,
    flow_code:     str,   # "M" = Imports, "X" = Exports
    hs_chapter:    str,
) -> float | None:
    """
    Fetch a single bilateral trade flow value from Comtrade.
    Returns the total primaryValue in USD, or None on error.
    """
    params = {
        "reporterCode": reporter_code,
        "partnerCode":  partner_code,
        "period":       period,
        "flowCode":     flow_code,
        "cmdCode":      hs_chapter,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            records = resp.json().get("data", [])
            if records:
                total = float(sum(r.get("primaryValue", 0) or 0 for r in records))
                return total
        else:
            print(f"    HTTP {resp.status_code} — reporter={reporter_code} "
                  f"partner={partner_code} HS={hs_chapter}")
    except requests.exceptions.Timeout:
        print(f"    Timeout — partner={partner_code} HS={hs_chapter}")
    except Exception as exc:
        print(f"    Error — {exc}")
    return None


def fetch_all_flows(
    period: int = 2023,
    flow_code: str = "M",
    reporter: int = REPORTER_USA,
) -> pd.DataFrame:
    """
    Fetch import flows for all country × HS chapter combinations.
    Returns a DataFrame with columns:
        country | m49_code | hs_chapter | hs_description | trade_value_usd
    """
    print(f"  [Comtrade] Fetching {flow_code} flows | "
          f"Reporter=USA({reporter}) | Period={period} …")

    rows = []
    total_calls = len(COUNTRY_M49) * len(HS_CHAPTERS)
    call_n = 0

    for country, m49 in COUNTRY_M49.items():
        for hs, desc in HS_CHAPTERS.items():
            call_n += 1
            val = fetch_trade_flow(reporter, m49, period, flow_code, hs)
            label = f"${val:,.0f}" if val is not None else "no data"
            print(f"    [{call_n:>2}/{total_calls}] {country:<28} HS {hs} → {label}")
            rows.append({
                "country":          country,
                "m49_code":         m49,
                "hs_chapter":       hs,
                "hs_description":   desc,
                "trade_value_usd":  val,
                "flow":             "Import" if flow_code == "M" else "Export",
                "period":           period,
            })
            time.sleep(REQUEST_DELAY)

    return pd.DataFrame(rows)


def compare_with_ledger(
    flows_df:   pd.DataFrame,
    ledger_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare official Comtrade bilateral totals vs. ledger-recorded totals
    per country + commodity cluster (HS chapter).
    High ledger/comtrade ratio signals potential TBML over-invoicing.
    """
    # Map commodities to HS chapters
    ledger_df = ledger_df.copy()
    ledger_df["hs_chapter"] = ledger_df["Commodity"].map(COMMODITY_HS)

    ledger_agg = (
        ledger_df.groupby(["Vendor_Country", "hs_chapter"])["Total_Value_USD"]
        .sum()
        .reset_index()
        .rename(columns={
            "Vendor_Country":  "country",
            "Total_Value_USD": "ledger_total_usd",
        })
    )

    # Aggregate comtrade per country + hs
    ct_agg = (
        flows_df[flows_df["trade_value_usd"].notna()]
        .groupby(["country", "hs_chapter"])["trade_value_usd"]
        .sum()
        .reset_index()
        .rename(columns={"trade_value_usd": "comtrade_total_usd"})
    )

    merged = ledger_agg.merge(ct_agg, on=["country", "hs_chapter"], how="left")
    merged["ledger_comtrade_ratio"] = (
        merged["ledger_total_usd"] / merged["comtrade_total_usd"].replace(0, float("nan"))
    ).round(6)
    merged["tbml_flag"] = merged["ledger_comtrade_ratio"].apply(
        lambda r: "⚠ HIGH" if r is not None and r > 0.05
                  else ("✓ OK" if r is not None else "— No Comtrade Data")
    )
    return merged.sort_values("ledger_comtrade_ratio", ascending=False)


def main() -> dict:
    DIVIDER = "=" * 65
    print(DIVIDER)
    print("  UN Comtrade API – Trade Flow Fetcher")
    print(DIVIDER)

    print("\n[STEP 1/3]  Fetching bilateral US import flows (2023) …")
    flows = fetch_all_flows(period=2023, flow_code="M")

    print(f"\n[STEP 2/3]  Trade flow summary:")
    pivot = flows.pivot_table(
        index="country", columns="hs_chapter",
        values="trade_value_usd", aggfunc="sum"
    ).round(0)
    print(pivot.to_string())

    print("\n[STEP 3/3]  Cross-referencing with ledger …")
    try:
        ledger = pd.read_csv("metallurgical_ledgers.csv")
        comparison = compare_with_ledger(flows, ledger)
        print(comparison[["country","hs_chapter","ledger_total_usd",
                            "comtrade_total_usd","ledger_comtrade_ratio","tbml_flag"]].to_string(index=False))
        comparison.to_csv("comtrade_comparison.csv", index=False)
        print("  Exported → comtrade_comparison.csv")
    except FileNotFoundError:
        print("  Ledger CSV not found – skipping comparison.")
        comparison = pd.DataFrame()

    flows.to_csv("comtrade_flows.csv", index=False)
    print("  Exported → comtrade_flows.csv")

    print(f"\n✓ Comtrade fetch complete.\n")
    return {"flows": flows, "comparison": comparison}


if __name__ == "__main__":
    main()