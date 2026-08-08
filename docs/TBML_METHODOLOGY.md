# TBML Detection Methodology

## What is TBML?

Trade-Based Money Laundering (TBML) uses international trade transactions to move dirty money across borders. The basic idea is simple — you manipulate the price, quantity, or type of goods on an invoice to transfer value without it looking like a financial transaction.

The FATF (Financial Action Task Force) identifies TBML as one of the three main methods of laundering money, alongside the traditional financial system and bulk cash smuggling. It's hard to detect because trade involves so many legitimate parties, documents, and jurisdictions.

Common TBML techniques:
- **Over-invoicing:** Seller inflates the price → excess money flows to the seller's account
- **Under-invoicing:** Buyer pays less than market rate → the difference is laundered through other channels  
- **Multiple invoicing:** Same goods invoiced more than once
- **Phantom shipments:** Invoices for goods that don't exist

---

## Detection methods used in this project

### 1. Over/under-invoicing detection

The most straightforward TBML flag. We compare the unit price on each invoice against the real market spot price for that commodity on that date.

**How it works:**
- Fetch live commodity prices from Yahoo Finance (Gold futures `GC=F`, Silver `SI=F`, Copper `HG=F`, Aluminum `ALI=F`)
- Calculate percentage deviation: `|Unit_Price - Spot_Price| / Spot_Price × 100`
- Score is clamped to 0–100

**Code:** `src/compliance_risk_scorer.py` → `score_price_delta()`

**Example from our data:** Gold Bullion transactions where `Unit_Price_USD` is $3,673 when spot is ~$2,000/oz — that's an 83% deviation, which is an immediate red flag for over-invoicing.

---

### 2. Ghost shipments / fabricated invoices

These are volume-based anomalies — transactions where the quantities or values don't make sense relative to historical patterns or official trade statistics.

**How it works:**
- For each vendor × commodity pair, compute the z-score of transaction volume against the group's historical distribution
- Cross-reference total volumes against official UN Comtrade bilateral trade data for the same HS code and country pair
- If a vendor's total volume exceeds a plausible share of official bilateral trade, flag it

**Code:** `src/sql_pipeline.py` → commodity anomaly CTE (z-score query), `api/comtrade_api.py` → `compare_with_ledger()`

**Example:** The z-score query found 209 transactions with |z| > 2.0 for commodity price anomalies. Gold Bullion had the most extreme outliers (z-score up to 12.9).

---

### 3. Structuring / smurfing

Structuring means breaking up large transactions into smaller ones to stay below regulatory reporting thresholds. In the US, banks must file a Currency Transaction Report (CTR) for transactions over $10,000.

**How it works:**
- For each vendor, we look at three things:
  1. **Transaction frequency** — how often they transact (z-score vs all vendors)
  2. **CTR proximity** — what fraction of their transactions fall in the $7,500–$10,000 range (the "just below threshold" zone)
  3. **Average transaction size** — unusually small avg values relative to the CTR threshold
- Final score = 50% frequency + 30% CTR proximity + 20% value pattern

**Code:** `src/compliance_risk_scorer.py` → `score_smurfing()`

**Example:** The SQL pipeline flagged 374 near-CTR transactions. Vendors with high concentrations of transactions in the $7.5k–$10k band get elevated smurfing scores.

---

### 4. Sanctions evasion screening

Checks whether transaction counterparties are on sanctions lists or operating from sanctioned jurisdictions.

**How it works:**
- Downloads the OFAC SDN (Specially Designated Nationals) list directly from the U.S. Treasury website
- Parses the CSV (it's a messy format — the "remarks" field contains nationality info that needs text extraction)
- Normalizes and matches vendor countries against the sanctions set
- Any match against a sanctioned jurisdiction = automatic score of 100

**Code:** `src/compliance_risk_scorer.py` → `fetch_ofac_sanctioned_set()`, `score_ofac()`

**Example:** Transactions from `Sanctioned_Proxy_Alpha` and `Sanctioned_Proxy_Beta` in our dataset represent jurisdictions routing through proxy entities to evade sanctions.

---

### 5. Benford's Law analysis

Benford's Law says that in naturally occurring numeric data, the leading digit '1' appears about 30% of the time, '2' about 17.6%, and so on — with '9' appearing only ~4.6%. When people fabricate numbers, they tend to distribute digits more uniformly or over-represent certain digits.

**How it works:**
- Extract the leading significant digit from each `Total_Value_USD` value
- Group by vendor and compute the observed digit frequency distribution
- Compare observed vs expected (Benford) using Mean Absolute Deviation (MAD):

$$P(d) = \log_{10}\left(1 + \frac{1}{d}\right) \quad \text{for } d = 1, 2, \ldots, 9$$

$$\text{MAD} = \frac{1}{9} \sum_{d=1}^{9} |f_{\text{observed}}(d) - f_{\text{expected}}(d)|$$

- Score = MAD / 0.15 × 100 (normalized so that a MAD of 0.15 or higher maxes out the score)

**Code:** `src/compliance_risk_scorer.py` → `score_benfords_law()`, `plot_benfords_chart()`

**Example:** A vendor whose invoices show digits 7, 8, 9 appearing 40%+ of the time (vs expected ~12%) gets a high Benford deviation score.

---

### 6. Mahalanobis distance (multivariate outlier detection)

A transaction might look normal in price and normal in volume separately, but the *combination* could be highly unusual. Mahalanobis distance measures how far a data point is from the center of a distribution, accounting for correlations between variables.

**How it works:**
- Feature space: `[Volume_MT, Unit_Price_USD, Total_Value_USD]`
- Compute the mean vector μ and covariance matrix Σ across all transactions
- For each transaction x, calculate:

$$D(x) = \sqrt{(x - \mu)^T \, \Sigma^{-1} \, (x - \mu)}$$

- Under normality, D² follows a chi-squared distribution with k degrees of freedom (k = number of features = 3)
- Threshold: χ²(0.975, df=3) = 9.348, so D_threshold = √9.348 ≈ 3.06
- Transactions beyond this threshold are flagged as multivariate outliers
- Risk score = D / D_threshold × 100, clamped to 0–100

**Code:** `src/compliance_risk_scorer.py` → `score_mahalanobis()`, `plot_mahalanobis_scatter()`

**Results:** 381 outliers identified (2.53% of all transactions). When we overlay ground-truth fraud labels on the scatter plot, there's strong overlap with the Mahalanobis outlier region.

---

## Composite scoring

All six individual scores are combined into a weighted composite:

| Component | Weight | Rationale |
|-----------|--------|-----------|
| OFAC sanctions | 30% | Highest weight — sanctions violations have the most severe legal consequences |
| Price deviation | 20% | Direct indicator of invoice manipulation |
| Smurfing | 15% | Common TBML pattern that's relatively easy to detect |
| Geopolitical risk | 15% | Country-level context matters for cross-border transactions |
| Mahalanobis | 10% | Catches subtle multivariate anomalies |
| Benford's Law | 10% | Useful signal but can have false positives |

## Risk tiers

| Tier | Score range | Action |
|------|-----------|--------|
| CRITICAL | ≥ 75 | Freeze transaction, file SAR, escalate immediately |
| HIGH | 50–74 | Manual review by compliance team within 24 hours |
| MEDIUM | 25–49 | Flag for periodic review and trend monitoring |
| LOW | < 25 | Normal processing, keep in baseline |

---

## References

1. FATF, "Trade-Based Money Laundering," FATF Report, 2006
2. FinCEN, "Advisory on Trade-Based Money Laundering," FIN-2010-A001
3. Nigrini, M., "Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection," Wiley, 2012
4. De Maesschalck, R. et al., "The Mahalanobis distance," Chemometrics and Intelligent Laboratory Systems, 2000
5. U.S. Department of the Treasury, OFAC SDN List, https://www.treasury.gov/ofac/downloads/sdn.csv
