Breakeven Inflation Decomposition

This project decomposes breakeven inflation (BEI) into three components:

Expected Inflation – based on survey data

Liquidity Premium – from TIPS and breakeven volatility

Inflation Risk Premium – residual component

The framework helps understand whether changes in market-implied inflation are driven by fundamentals or technical factors.

📊 Motivation

Market breakeven inflation is often treated as a direct measure of expected inflation, but it also contains risk and liquidity premiums. By separating these components, investors and policymakers can:

Identify true inflation expectations

Understand market stress episodes

Better interpret breakeven inflation movements during crises

🧱 Data

Data is sourced from FRED:

Data Series	Description
DGS5, DGS10	5Y and 10Y Treasury yields
DFII5, DFII10	5Y and 10Y TIPS yields
CPIAUCSL	CPI index (used to compute YoY inflation)
Survey Data	University of Michigan 5Y inflation expectations

Additional volatility proxies are computed using rolling 20-day standard deviations of TIPS yields and breakevens.

⚙️ Methodology

Data Preprocessing

Download data from FRED using pandas_datareader

Align series to a single DataFrame

Compute 5Y and 5Y5Y breakevens:

bei_5y = nominal_yield_5y - tips_yield_5y
bei_5y5y = (10 * nominal_yield_10y - 5 * nominal_yield_5y) / 5 - (10 * tips_yield_10y - 5 * tips_yield_5y) / 5


Expected Inflation

Regress 5Y breakeven on survey inflation:

OLS(bei_5y ~ survey_5y)


Fitted values represent expected inflation

Residual contains risk and liquidity effects

Liquidity Premium

Compute rolling volatility:

tips_vol = tips_yield.rolling(20).std()
bei_vol = bei_5y.rolling(20).std()


Regress BEI on volatilities:

OLS(bei_5y ~ tips_vol + bei_vol)


Captures liquidity-driven fluctuations

Inflation Risk Premium

Residual after removing expected inflation and liquidity premium:

inflation_risk_premium = bei_5y - expected_inflation - liquidity_premium

📈 Results
Expected Inflation

Smooth and closely tracks survey expectations

Moves slowly across cycles

Captures market consensus

Liquidity Premium

Spikes during crisis periods (e.g., 2008, 2020)

Reflects illiquidity and market stress

Inflation Risk Premium

Highly cyclical

Widens when inflation uncertainty rises

Captures risk compensation demanded by investors

📑 Model Diagnostics

Regression summaries show expected inflation is almost perfectly explained by survey data

Liquidity model explains around 10% of variability — expected, as market stress is noisy

Residuals (risk premium) provide insight into uncertainty and tail-risk compensation

🛠️ Requirements

Python 3.10+

Libraries:

pip install pandas numpy pandas_datareader statsmodels matplotlib seaborn

🔗 References

FRED Economic Data

University of Michigan Inflation Survey

J. Gürkaynak et al., “The TIPS Yield Curve and Inflation Compensation”

🚀 How to Run

Clone the repo:

git clone https://github.com/yourusername/breakeven-inflation.git
cd breakeven-inflation


Run the script:

python breakeven_decomposition.py


The script outputs:

Regression summaries

Plots of expected inflation, liquidity premium, and inflation risk premium

⚡ Key Takeaways

Breakeven inflation ≠ pure expectation

Liquidity and risk premiums are significant during stressed periods

Decomposition improves understanding of market pricing and monetary expectations
