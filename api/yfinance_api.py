"""
Fetches real-time and historical spot prices for commodities.
Validates ledger prices against market spot prices.
"""

import os
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data")
_OUTPUT_DIR  = os.path.join(_PROJECT_DIR, "outputs")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ticker map and unit conversion factors
METAL_TICKERS = {
    "Gold_Bullion":   {"ticker": "GC=F",  "unit": "USD/troy-oz",  "factor": 1.0},
    "Silver_Ingot":   {"ticker": "SI=F",  "unit": "USD/troy-oz",  "factor": 1.0},
    "Copper_Cathode": {"ticker": "HG=F",  "unit": "USD/lb",       "factor": 1.0},
    "Aluminum_Ingot": {"ticker": "ALI=F", "unit": "USD/MT",       "factor": 1.0},
}

LEDGER_PRICE_UNITS = {
    # ledger stores Market_Spot_Price in these units
    "Gold_Bullion":   "USD/troy-oz",
    "Silver_Ingot":   "USD/troy-oz",
    "Copper_Cathode": "USD/lb",
    "Aluminum_Ingot": "USD/MT",
}


def fetch_current_spot_prices() -> dict:
    """Fetch the latest closing price for each metal."""
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
    """Fetch historical daily closing prices for all metals."""
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
    """Compare ledger prices against market prices and flag significant deviations."""
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

    # 1. current spot prices
    spots = fetch_current_spot_prices()

    # 2. historical prices
    hist_df = fetch_historical_spot_prices(period="1y")
    if not hist_df.empty:
        print(f"\n  Historical price table ({len(hist_df)} trading days):")
        print(hist_df.tail(5).round(4).to_string())

    # 3. cross-validate against ledger
    try:
        ledger = pd.read_csv(os.path.join(_DATA_DIR, "metallurgical_ledgers.csv"))
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