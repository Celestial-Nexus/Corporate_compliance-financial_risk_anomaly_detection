"""
yfinance_api.py
────────────────────────────────────────────────────────────────────────────
Corporate Compliance – Metal Spot Price Fetcher (Yahoo Finance)

Fetches real-time and historical spot prices for the four commodities in the
metallurgical ledger:
    • Gold Bullion   → GC=F (COMEX Gold Futures)
    • Silver Ingot   → SI=F (COMEX Silver Futures)
    • Copper Cathode → HG=F (COMEX Copper Futures, USD/lb → converted to MT)
    • Aluminum Ingot → ALI=F (LME Aluminum Futures, USD/MT)

Also computes a price-validity check against the ledger's Market_Spot_Price
column to flag systematic over/under-invoicing.
"""

import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Ticker map + unit conversion factors to USD / troy-oz or USD / MT
# ─────────────────────────────────────────────────────────────────────────────
METAL_TICKERS = {
    "Gold_Bullion":   {"ticker": "GC=F",  "unit": "USD/troy-oz",  "factor": 1.0},
    "Silver_Ingot":   {"ticker": "SI=F",  "unit": "USD/troy-oz",  "factor": 1.0},
    "Copper_Cathode": {"ticker": "HG=F",  "unit": "USD/lb",       "factor": 1.0},
    "Aluminum_Ingot": {"ticker": "ALI=F", "unit": "USD/MT",       "factor": 1.0},
}

LEDGER_PRICE_UNITS = {
    # The ledger stores Market_Spot_Price in the following units per commodity
    "Gold_Bullion":   "USD/troy-oz",
    "Silver_Ingot":   "USD/troy-oz",
    "Copper_Cathode": "USD/lb",
    "Aluminum_Ingot": "USD/MT",
}


def fetch_current_spot_prices() -> dict:
    """
    Fetch the latest closing price for each metal from Yahoo Finance.
    Returns a dict: { commodity_name: current_price_float }
    """
    spots = {}
    print("  [yfinance] Fetching current metal spot prices …")
    for commodity, cfg in METAL_TICKERS.items():
        ticker_sym = cfg["ticker"]
        try:
            t     = yf.Ticker(ticker_sym)
            hist  = t.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].dropna().iloc[-1]) * cfg["factor"]
                spots[commodity] = round(price, 4)
                print(f"    {commodity:<20} ({ticker_sym})  →  {price:>10,.4f}  {cfg['unit']}")
            else:
                print(f"    {commodity:<20} ({ticker_sym})  →  No data returned")
                spots[commodity] = None
        except Exception as exc:
            print(f"    {commodity:<20} ({ticker_sym})  →  Error: {exc}")
            spots[commodity] = None
    return spots


def fetch_historical_spot_prices(period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical daily closing prices for all metals.
    Returns a DataFrame with columns = commodity names, index = Date.
    """
    print(f"\n  [yfinance] Fetching {period} historical prices …")
    frames = {}
    for commodity, cfg in METAL_TICKERS.items():
        try:
            t    = yf.Ticker(cfg["ticker"])
            hist = t.history(period=period)[["Close"]].rename(columns={"Close": commodity})
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            frames[commodity] = hist[commodity]
            print(f"    {commodity:<20}  {len(hist):,} trading days")
        except Exception as exc:
            print(f"    {commodity:<20}  Error: {exc}")

    if frames:
        return pd.DataFrame(frames).sort_index()
    return pd.DataFrame()


def validate_ledger_prices(
    ledger_df: pd.DataFrame,
    spot_prices: dict,
    tolerance_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Compare the ledger's Market_Spot_Price against real Yahoo Finance prices.
    Returns a summary DataFrame flagging commodities with significant drift.

    Parameters
    ──────────
    ledger_df     : pd.DataFrame with columns [Commodity, Market_Spot_Price]
    spot_prices   : dict returned by fetch_current_spot_prices()
    tolerance_pct : flag if avg ledger price differs from current spot by > this %
    """
    rows = []
    for commodity, yf_price in spot_prices.items():
        if yf_price is None:
            continue
        subset = ledger_df[ledger_df["Commodity"] == commodity]["Market_Spot_Price"]
        if subset.empty:
            continue
        ledger_avg = subset.mean()
        dev_pct    = abs(ledger_avg - yf_price) / yf_price * 100.0
        rows.append({
            "Commodity":          commodity,
            "Ledger_Avg_Price":   round(ledger_avg, 4),
            "YF_Current_Spot":    round(yf_price, 4),
            "Deviation_Pct":      round(dev_pct, 2),
            "Flag":               "⚠ DRIFT" if dev_pct > tolerance_pct else "✓ OK",
        })
    return pd.DataFrame(rows)


def main() -> dict:
    DIVIDER = "=" * 65
    print(DIVIDER)
    print("  Yahoo Finance – Metal Spot Price Fetcher")
    print(DIVIDER)

    # 1. Current spot prices
    spots = fetch_current_spot_prices()

    # 2. Historical prices (1 year)
    hist_df = fetch_historical_spot_prices(period="1y")
    if not hist_df.empty:
        print(f"\n  Historical price table ({len(hist_df)} trading days):")
        print(hist_df.tail(5).round(4).to_string())

    # 3. Cross-validate against ledger
    try:
        ledger = pd.read_csv("metallurgical_ledgers.csv")
        print("\n  [yfinance] Ledger price validation …")
        validation = validate_ledger_prices(ledger, spots)
        print(validation.to_string(index=False))
    except FileNotFoundError:
        print("  [yfinance] Ledger CSV not found – skipping validation.")
        validation = pd.DataFrame()

    print(f"\n✓ Yahoo Finance fetch complete.\n")
    return {"spots": spots, "history": hist_df, "validation": validation}


if __name__ == "__main__":
    main()