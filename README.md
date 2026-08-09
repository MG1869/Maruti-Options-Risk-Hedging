# Quantitative Options Pricing & Risk Hedging — Maruti Suzuki India Ltd

**FRAM (Financial Risk Analytics & Management) Assignment**
Birla Institute of Technology and Science (BITS), Pilani
Nov 2025 – Dec 2025

**Team:**
- Aditya Iyer (2023A3PS0374P)
- Arnabha Acharya (2023A1PS0209P)
- Harshit Jaiswal (2023A4PS1286P)
- Mihir Godawat (2023A3PS0356P)
- Reyansh Verma (2023A3PS1212P)

---

## Overview

An end-to-end quantitative options pricing and risk management engine built on **Maruti Suzuki India Ltd (NSE: MARUTI)** equity. The project covers historical volatility estimation, Black–Scholes–Merton option pricing across multiple strikes and maturities, full Greeks computation, portfolio construction, delta–gamma hedging, and Value-at-Risk (VaR) analysis — all implemented from scratch in Python.

---

## Tech Stack

`Python` · `NumPy` · `Pandas` · `SciPy (stats, optimize)` · `yfinance` · `Matplotlib` · `Seaborn` · `OpenPyXL`

---

## Methodology & Results

### Part A — Data Collection & Statistics
- Pulled 3 months of daily price data for MARUTI.NS via `yfinance`.
- Computed log returns and distributional statistics:

| Metric | Value |
|---|---|
| Current Stock Price | ₹15,900.00 |
| Mean Daily Return | 0.1062% |
| Daily Std Dev | 1.0418% |
| **Annualized Volatility** | **16.54%** |
| Skewness | -0.503 (left-skewed — larger downside moves) |
| Kurtosis | 1.085 (fatter tails than normal distribution) |

### Part B — Option Pricing (Black–Scholes–Merton)
- Priced **30 options** (calls + puts) across 5 strikes × 3 maturities.
- Strikes spanning ATM ± 5% (₹15,105 to ₹16,695); maturities of 30, 60, and 90 days.
- Risk-free rate assumed at 6.5% (approx. RBI repo rate).

Sample (60-day maturity):

| Strike | Call Price | Put Price |
|---|---|---|
| 15,105 (ITM) | ₹1,054.75 | ₹99.21 |
| 15,900 (ATM) | ₹512.82 | ₹343.83 |
| 16,695 (OTM) | ₹192.85 | ₹810.41 |

### Part C — Greeks & Volatility Surface
- Computed **Delta, Gamma, Vega, Theta, Rho** for all 30 options analytically.
- Built a 3D implied volatility surface visualization (strike × maturity × IV) incorporating a volatility smile.
- ATM 60-day call: Delta 0.576, Gamma 0.00037, Vega 25.24, Theta -5.02/day.

### Part D — Portfolio Construction & Delta–Gamma Hedging
Constructed a 60-day option strategy (long ITM call, short OTM call, short OTM put):

| Strike | Type | Position |
|---|---|---|
| 15,582 | Call | Long (+1) |
| 16,218 | Call | Short (-1) |
| 15,105 | Put | Short (-1) |

**Unhedged portfolio Greeks:** Value ₹246.70 · Delta 0.399 · Gamma -0.00028 · Vega -19.08 · Theta +1.54

- Applied **delta hedging** via underlying stock offset, then **gamma hedging** using an auxiliary ATM call to neutralize convexity risk.
- Simulated portfolio P&L under ±1% and ±2% price shocks — hedged portfolio P&L stayed within ±₹14 vs. unhedged swings of up to ₹141.

### Part E — Value-at-Risk (VaR)
Computed 1-day VaR using both **parametric (variance-covariance)** and **historical simulation** methods, at 95% and 99% confidence:

| Confidence | Parametric VaR (Unhedged) | Parametric VaR (Hedged) | Historical VaR (Unhedged) | Historical VaR (Hedged) |
|---|---|---|---|---|
| 95% | ₹110.29 | ~₹0 | ₹113.38 | ~₹0 |
| 99% | ₹155.99 | ~₹0 | ₹157.60 | ~₹0 |

**Key takeaway:** Delta–gamma hedging reduced 1-day VaR by ~100%, effectively neutralizing directional and convexity risk on the options book.

---

## Repository Structure
```
├── fram_maruti_portfolio.py       # Full analysis script (Parts A–E)
├── FRAM_Maruti_Assignment_Results.xlsx   # Exported results (7 sheets)
├── part_a_stock_analysis.png      # Price & returns chart
├── part_c_volatility_surface.png  # 3D IV surface
├── part_d_hedging_pnl.png         # Hedged vs unhedged P&L
└── README.md
```

## How to Run
```bash
pip install numpy pandas yfinance matplotlib seaborn scipy openpyxl
python fram_maruti_portfolio.py
```
Outputs three charts (PNG) and one consolidated Excel workbook with all intermediate tables.

---

## Skills Demonstrated
Options pricing theory (Black–Scholes–Merton) · Greeks & risk sensitivities · Delta/gamma hedging strategy · Value-at-Risk (parametric & historical) · Quantitative Python (NumPy/SciPy/Pandas) · Financial data engineering · Risk visualization
