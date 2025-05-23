# Monte Carlo Stock Forecasting & Economic Adjustment Models

This repository contains a suite of Python scripts designed to simulate and evaluate stock price behavior using Monte Carlo methods, as well as to explore the impact of macroeconomic factors on short-term price rebounds. The core objective is to build and refine predictive financial models grounded in stochastic simulation, error optimization, and real-world indicator integration.

---

## Repository Contents

### 1. `MonteCarlov1.py`

Basic Monte Carlo simulation of stock price trajectories for AAPL based on historical log returns.

* **Method:** Geometric Brownian Motion (GBM)
* **Simulation Horizon:** 252 days
* **Features:**

  * 1000 stochastic price paths
  * Expected price and confidence intervals
  * Plot of projected price paths
* **Use Case:** Introductory model for examining variance in future stock behavior

---

### 2. `MCParamOptimization.py`

Extends the baseline simulation by testing and optimizing key model parameters, including:

* Number of simulations (`M`)
* Historical window length (`H`)

**Features:**

* Error analysis against actual price data using MAE
* Runtime benchmarking and computational cost trade-off analysis
* Linear regression on execution time
* Plots: MAE vs. M, MAE vs. H, runtime vs. parameters

**Use Case:** Inform optimal balance between model accuracy and efficiency for deployment

---

### 3. `MCReboundAdjusted.py` (Work-in-Progress)

Initial framework for modeling stock rebounds using macroeconomic indicators and volatility signals.

* **Data Sources:**

  * Yahoo Finance (`AAPL`, `^VIX`)
  * Federal Reserve Economic Data (FRED): `GDP`, `FEDFUNDS`, `CPIAUCSL`, `UNRATE`
* **Feature Engineering:**

  * Lag effects
  * Rolling averages
  * Interaction terms
  * Daily resampling and normalization
* **Output:** Cleaned and aligned dataset with engineered features

**Status:** In development; predictive rebound logic pending integration
**Use Case:** Eventually aims to adjust Monte Carlo outputs using macroeconomic rebound signals

---

## Technologies Used

* Python 3
* NumPy, pandas, yfinance, matplotlib
* scikit-learn
* statsmodels / scipy
* pandas-datareader

---

## How to Use

Ensure all dependencies are installed:

```bash
pip install numpy pandas yfinance matplotlib scikit-learn pandas-datareader
```

Run the core Monte Carlo simulation:

```bash
python MonteCarlov1.py
```

Run the optimization module:

```bash
python MCParamOptimization.py
```

Run the rebound data prep module (WIP, no forecasts yet):

```bash
python MCReboundAdjusted.py
```

---

## Author

**Quinton Peters**
B.S.E. Candidate, Risk, Data, and Financial Engineering
Duke University
