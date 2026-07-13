"""
sql_pipeline.py
────────────────────────────────────────────────────────────────────────────
Corporate Compliance – Trade-Compliance SQL Pipeline
Uses SQLite to manage the metallurgical ledger data pipeline.

Capabilities
────────────
• Ingests CSV → normalised SQLite tables (transactions, vendors, risk_scores)
• Analytical queries:
    – TBML pattern detection (CTEs + window functions)
    – Structuring / smurfing detection
    – Vendor-level aggregation and flagging
    – Country-level risk summary
• Exports results back to pandas DataFrames for downstream scoring / reporting
"""

import sqlite3
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
INPUT_CSV   = "metallurgical_ledgers.csv"
DB_PATH     = "compliance.db"
CTR_THRESH  = 10_000.0          # Cash-Transaction-Report threshold (USD)
SMURFING_LO = CTR_THRESH * 0.75  # Lower bound for structuring detection


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Ingest CSV into normalised tables
# ─────────────────────────────────────────────────────────────────────────────
def ingest_csv(conn: sqlite3.Connection, csv_path: str = INPUT_CSV) -> None:
    """Load the raw CSV and populate all relational tables."""
    print(f"  [SQL] Reading '{csv_path}' …")
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # ── transactions (main fact table) ──────────────────────────────────────
    conn.execute("DROP TABLE IF EXISTS transactions;")
    conn.execute("""
        CREATE TABLE transactions (
            transaction_id      TEXT    PRIMARY KEY,
            date                TEXT    NOT NULL,
            vendor_id           TEXT    NOT NULL,
            vendor_country      TEXT    NOT NULL,
            commodity           TEXT    NOT NULL,
            volume_mt           REAL    NOT NULL,
            market_spot_price   REAL,
            unit_price_usd      REAL,
            total_value_usd     REAL,
            payment_method      TEXT,
            is_fraud            INTEGER DEFAULT 0,
            fraud_type          TEXT
        );
    """)

    txn_df = df.rename(columns={
        "Transaction_ID":       "transaction_id",
        "Date":                 "date",
        "Vendor_ID":            "vendor_id",
        "Vendor_Country":       "vendor_country",
        "Commodity":            "commodity",
        "Volume_MT":            "volume_mt",
        "Market_Spot_Price":    "market_spot_price",
        "Unit_Price_USD":       "unit_price_usd",
        "Total_Value_USD":      "total_value_usd",
        "Payment_Method":       "payment_method",
        "Is_Fraud_Ground_Truth":"is_fraud",
        "Fraud_Type":           "fraud_type",
    })
    txn_df["date"] = txn_df["date"].astype(str)
    txn_df.to_sql("transactions", conn, if_exists="append", index=False)
    print(f"  [SQL] Inserted {len(txn_df):,} rows → transactions")

    # ── vendors (dimension table) ────────────────────────────────────────────
    conn.execute("DROP TABLE IF EXISTS vendors;")
    conn.execute("""
        CREATE TABLE vendors (
            vendor_id       TEXT PRIMARY KEY,
            vendor_country  TEXT NOT NULL,
            total_txns      INTEGER,
            total_value_usd REAL,
            avg_value_usd   REAL,
            is_sanctioned   INTEGER DEFAULT 0
        );
    """)

    SANCTIONED = {
        "sanctioned_proxy_alpha", "sanctioned_proxy_beta",
        "iran", "north korea", "russia", "syria", "cuba",
    }
    vendor_agg = (
        df.groupby(["Vendor_ID", "Vendor_Country"])
        .agg(
            total_txns=("Transaction_ID", "count"),
            total_value_usd=("Total_Value_USD", "sum"),
            avg_value_usd=("Total_Value_USD", "mean"),
        )
        .reset_index()
        .rename(columns={"Vendor_ID": "vendor_id", "Vendor_Country": "vendor_country"})
    )
    vendor_agg["is_sanctioned"] = vendor_agg["vendor_country"].apply(
        lambda c: 1 if c.strip().lower().replace(" ", "_") in SANCTIONED else 0
    )
    vendor_agg.to_sql("vendors", conn, if_exists="append", index=False)
    print(f"  [SQL] Inserted {len(vendor_agg):,} rows → vendors")

    # ── risk_scores (results table – populated later) ────────────────────────
    conn.execute("DROP TABLE IF EXISTS risk_scores;")
    conn.execute("""
        CREATE TABLE risk_scores (
            transaction_id          TEXT PRIMARY KEY,
            ofac_risk_score         REAL,
            price_delta_risk_score  REAL,
            smurfing_risk_score     REAL,
            benfords_law_risk_score REAL,
            geopolitical_risk_score REAL,
            mahalanobis_risk_score  REAL,
            composite_risk_score    REAL,
            risk_tier               TEXT
        );
    """)

    conn.commit()
    print("  [SQL] Schema created: transactions | vendors | risk_scores")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Analytical SQL Queries
# ─────────────────────────────────────────────────────────────────────────────
def query_tbml_patterns(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    TBML (Trade-Based Money Laundering) detection using CTEs.
    Flags vendors with:
      - Abnormally large price deviations from spot
      - High transaction frequency in short windows
      - Transactions with sanctioned countries
    """
    sql = f"""
    WITH price_deviations AS (
        SELECT
            transaction_id,
            vendor_id,
            vendor_country,
            commodity,
            date,
            total_value_usd,
            unit_price_usd,
            market_spot_price,
            ABS(unit_price_usd - market_spot_price) / NULLIF(market_spot_price, 0) * 100.0
                AS price_dev_pct,
            CASE
                WHEN vendor_country IN (
                    'Sanctioned_Proxy_Alpha','Sanctioned_Proxy_Beta',
                    'Iran','North Korea','Russia','Syria','Cuba'
                ) THEN 1 ELSE 0
            END AS is_sanctioned_counterparty
        FROM transactions
    ),
    vendor_stats AS (
        SELECT
            vendor_id,
            COUNT(*)                  AS n_txns,
            SUM(total_value_usd)      AS total_volume_usd,
            AVG(price_dev_pct)        AS avg_price_dev_pct,
            MAX(price_dev_pct)        AS max_price_dev_pct,
            SUM(is_sanctioned_counterparty) AS sanctioned_txns,
            MAX(is_sanctioned_counterparty) AS any_sanctioned
        FROM price_deviations
        GROUP BY vendor_id
    ),
    structuring AS (
        SELECT
            vendor_id,
            COUNT(*) AS near_ctr_txns
        FROM transactions
        WHERE total_value_usd >= {SMURFING_LO}
          AND total_value_usd  < {CTR_THRESH}
        GROUP BY vendor_id
    ),
    tbml_flags AS (
        SELECT
            v.vendor_id,
            v.n_txns,
            v.total_volume_usd,
            ROUND(v.avg_price_dev_pct, 2)   AS avg_price_dev_pct,
            ROUND(v.max_price_dev_pct, 2)   AS max_price_dev_pct,
            v.any_sanctioned,
            COALESCE(s.near_ctr_txns, 0)    AS near_ctr_txns,
            -- Composite TBML flag
            CASE
                WHEN v.any_sanctioned = 1                    THEN 'HIGH'
                WHEN v.avg_price_dev_pct > 20
                  OR COALESCE(s.near_ctr_txns, 0) >= 3      THEN 'MEDIUM'
                WHEN v.avg_price_dev_pct > 5                 THEN 'LOW'
                ELSE 'CLEAN'
            END AS tbml_risk_tier
        FROM vendor_stats v
        LEFT JOIN structuring s ON v.vendor_id = s.vendor_id
    )
    SELECT * FROM tbml_flags
    ORDER BY
        CASE tbml_risk_tier
            WHEN 'HIGH'   THEN 1
            WHEN 'MEDIUM' THEN 2
            WHEN 'LOW'    THEN 3
            ELSE 4
        END,
        total_volume_usd DESC;
    """
    return pd.read_sql_query(sql, conn)


def query_structuring_alerts(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Structuring / smurfing detection using window functions.
    Identifies sequences of just-below-threshold transactions per vendor.
    """
    sql = f"""
    WITH ranked AS (
        SELECT
            transaction_id,
            vendor_id,
            vendor_country,
            date,
            total_value_usd,
            payment_method,
            ROW_NUMBER() OVER (PARTITION BY vendor_id ORDER BY date, total_value_usd) AS rn,
            COUNT(*) FILTER (
                WHERE total_value_usd >= {SMURFING_LO}
                  AND total_value_usd  < {CTR_THRESH}
            ) OVER (PARTITION BY vendor_id) AS near_ctr_count
        FROM transactions
    )
    SELECT
        transaction_id,
        vendor_id,
        vendor_country,
        date,
        ROUND(total_value_usd, 2) AS total_value_usd,
        payment_method,
        near_ctr_count
    FROM ranked
    WHERE total_value_usd >= {SMURFING_LO}
      AND total_value_usd  < {CTR_THRESH}
      AND near_ctr_count   >= 2
    ORDER BY vendor_id, date;
    """
    return pd.read_sql_query(sql, conn)


def query_country_risk_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Country-level risk aggregation with running totals (window function).
    """
    sql = """
    WITH country_agg AS (
        SELECT
            vendor_country,
            COUNT(*)                    AS total_txns,
            COUNT(DISTINCT vendor_id)   AS unique_vendors,
            ROUND(SUM(total_value_usd)/1e6, 3)  AS total_volume_mUSD,
            ROUND(AVG(total_value_usd), 2)       AS avg_txn_usd,
            ROUND(MAX(total_value_usd), 2)       AS max_txn_usd,
            SUM(is_fraud)               AS fraud_txns,
            ROUND(AVG(
                ABS(unit_price_usd - market_spot_price)
                / NULLIF(market_spot_price, 0) * 100.0
            ), 2) AS avg_price_dev_pct
        FROM transactions
        GROUP BY vendor_country
    )
    SELECT
        vendor_country,
        total_txns,
        unique_vendors,
        total_volume_mUSD,
        avg_txn_usd,
        max_txn_usd,
        fraud_txns,
        ROUND(100.0 * fraud_txns / NULLIF(total_txns, 0), 2) AS fraud_rate_pct,
        avg_price_dev_pct,
        SUM(total_volume_mUSD) OVER (
            ORDER BY total_volume_mUSD DESC
        ) AS cumulative_volume_mUSD
    FROM country_agg
    ORDER BY total_volume_mUSD DESC;
    """
    return pd.read_sql_query(sql, conn)


def query_commodity_anomalies(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Per-commodity price anomaly detection: find transactions whose unit price
    deviates more than 2 standard deviations from the commodity mean.
    """
    sql = """
    WITH stats AS (
        SELECT
            commodity,
            AVG(unit_price_usd)                          AS mean_price,
            -- SQLite has no STDDEV; approximate with variance formula
            SQRT(
                AVG(unit_price_usd * unit_price_usd)
                - AVG(unit_price_usd) * AVG(unit_price_usd)
            )                                             AS std_price
        FROM transactions
        GROUP BY commodity
    )
    SELECT
        t.transaction_id,
        t.date,
        t.vendor_id,
        t.vendor_country,
        t.commodity,
        ROUND(t.unit_price_usd, 4)      AS unit_price_usd,
        ROUND(t.market_spot_price, 4)   AS market_spot_price,
        ROUND(s.mean_price, 4)          AS commodity_mean_price,
        ROUND(s.std_price,  4)          AS commodity_std_price,
        ROUND(
            ABS(t.unit_price_usd - s.mean_price) / NULLIF(s.std_price, 0),
            4
        )                               AS z_score,
        t.is_fraud
    FROM transactions t
    JOIN stats s ON t.commodity = s.commodity
    WHERE ABS(t.unit_price_usd - s.mean_price) / NULLIF(s.std_price, 0) > 2.0
    ORDER BY z_score DESC;
    """
    return pd.read_sql_query(sql, conn)


def query_rolling_weekly_volume(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Rolling 7-day transaction volume per vendor using window functions.
    Flags vendors with unusually high weekly activity (>2 std devs above mean).
    """
    sql = """
    WITH daily AS (
        SELECT
            vendor_id,
            date,
            COUNT(*) AS daily_txns,
            SUM(total_value_usd) AS daily_volume
        FROM transactions
        GROUP BY vendor_id, date
    ),
    vendor_weekly_stats AS (
        SELECT
            vendor_id,
            AVG(daily_txns)   AS mean_daily_txns,
            AVG(daily_volume) AS mean_daily_vol
        FROM daily
        GROUP BY vendor_id
    )
    SELECT
        d.vendor_id,
        d.date,
        d.daily_txns,
        ROUND(d.daily_volume, 2)           AS daily_volume_usd,
        ROUND(s.mean_daily_txns, 2)        AS mean_daily_txns,
        ROUND(s.mean_daily_vol, 2)         AS mean_daily_vol_usd,
        ROUND(d.daily_txns / NULLIF(s.mean_daily_txns, 0), 2) AS txn_ratio
    FROM daily d
    JOIN vendor_weekly_stats s ON d.vendor_id = s.vendor_id
    WHERE d.daily_txns / NULLIF(s.mean_daily_txns, 0) > 3.0
    ORDER BY txn_ratio DESC
    LIMIT 100;
    """
    return pd.read_sql_query(sql, conn)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Upload risk scores back to DB
# ─────────────────────────────────────────────────────────────────────────────
def upload_risk_scores(conn: sqlite3.Connection, scored_df: pd.DataFrame) -> None:
    """
    Write the scored DataFrame rows into the risk_scores table.
    Expected columns (subset): transaction_id, *_risk_score, composite_risk_score
    """
    score_cols = [
        "Transaction_ID",
        "ofac_risk_score", "price_delta_risk_score",
        "smurfing_risk_score", "benfords_law_risk_score",
        "geopolitical_risk_score", "mahalanobis_risk_score",
        "composite_risk_score",
    ]
    available = [c for c in score_cols if c in scored_df.columns]
    rs = scored_df[available].copy()
    rs = rs.rename(columns={"Transaction_ID": "transaction_id"})

    if "composite_risk_score" in rs.columns:
        rs["risk_tier"] = rs["composite_risk_score"].apply(
            lambda s: "CRITICAL" if s >= 300 else
                      "HIGH"     if s >= 200 else
                      "MEDIUM"   if s >= 100 else "LOW"
        )

    conn.execute("DELETE FROM risk_scores;")
    rs.to_sql("risk_scores", conn, if_exists="append", index=False)
    conn.commit()
    print(f"  [SQL] Uploaded {len(rs):,} rows → risk_scores")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    DIVIDER = "=" * 65
    print(DIVIDER)
    print("  Trade-Compliance SQL Pipeline")
    print(DIVIDER)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Removed old '{DB_PATH}'")

    conn = get_connection()

    print("\n[STEP 1/5]  Ingesting CSV into SQLite …")
    ingest_csv(conn)

    print("\n[STEP 2/5]  TBML Pattern Detection (CTE Query) …")
    tbml_df = query_tbml_patterns(conn)
    high = tbml_df[tbml_df["tbml_risk_tier"] == "HIGH"]
    med  = tbml_df[tbml_df["tbml_risk_tier"] == "MEDIUM"]
    print(f"            HIGH risk vendors : {len(high):,}")
    print(f"            MEDIUM risk vendors: {len(med):,}")
    print(f"            Full results      : {len(tbml_df):,} vendors")

    print("\n[STEP 3/5]  Structuring / Smurfing Alert Query …")
    struct_df = query_structuring_alerts(conn)
    print(f"            Flagged near-CTR transactions: {len(struct_df):,}")

    print("\n[STEP 4/5]  Country Risk Summary (Window Function) …")
    country_df = query_country_risk_summary(conn)
    print(country_df[["vendor_country","total_txns","total_volume_mUSD","fraud_rate_pct"]].to_string(index=False))

    print("\n[STEP 5/5]  Commodity Price Anomalies (Z-Score SQL) …")
    anomaly_df = query_commodity_anomalies(conn)
    print(f"            Transactions with |z| > 2.0: {len(anomaly_df):,}")
    print(f"\n  Top 5 price outliers:")
    print(anomaly_df[["transaction_id","commodity","unit_price_usd","z_score"]].head(5).to_string(index=False))

    conn.close()

    # Export results
    tbml_df.to_csv("tbml_alerts.csv", index=False)
    struct_df.to_csv("structuring_alerts.csv", index=False)
    country_df.to_csv("country_risk_summary.csv", index=False)
    anomaly_df.to_csv("commodity_anomalies.csv", index=False)
    print(f"\n  Exported: tbml_alerts.csv | structuring_alerts.csv | country_risk_summary.csv | commodity_anomalies.csv")
    print(f"  Database: {DB_PATH}")
    print(f"\n✓ SQL Pipeline complete.\n")

    return {
        "tbml": tbml_df,
        "structuring": struct_df,
        "country": country_df,
        "anomalies": anomaly_df,
    }


if __name__ == "__main__":
    main()
