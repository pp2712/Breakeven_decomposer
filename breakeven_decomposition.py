import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas_datareader import data as pdr

# -----------------------------
# 1. PARAMETERS
# -----------------------------
START_DATE = "2004-01-01"
ROLLING_VOL_WINDOW = 20

# -----------------------------
# 2. DATA INGESTION (FRED)
# -----------------------------
def load_fred_series():
    series = {
        "DGS5": "nom_5y",
        "DGS10": "nom_10y",
        "DFII5": "tips_5y",
        "DFII10": "tips_10y",
        "CPIAUCSL": "cpi",
        "MICH": "survey_5y"
    }

    df = pd.concat(
        [pdr.DataReader(code, "fred", START_DATE) for code in series],
        axis=1
    )
    df.columns = series.values()
    return df


# -----------------------------
# 3. PREPROCESSING
# -----------------------------
def preprocess_data(df):
    # CPI YoY inflation
    df["cpi_yoy"] = df["cpi"].pct_change(12) * 100

    # Breakevens
    df["bei_5y"] = df["nom_5y"] - df["tips_5y"]
    df["bei_10y"] = df["nom_10y"] - df["tips_10y"]

    # 5y5y Forward Breakeven
    df["bei_5y5y"] = (
        (df["nom_10y"] * 10 - df["nom_5y"] * 5) / 5
        - (df["tips_10y"] * 10 - df["tips_5y"] * 5) / 5
    )

    return df.dropna()


# -----------------------------
# 4. EXPECTED INFLATION MODEL
# -----------------------------
def estimate_expected_inflation(df):
    """
    Survey-anchored regression:
    BEI ~ Survey inflation expectations
    """
    X = sm.add_constant(df["survey_5y"])
    y = df["bei_5y"]

    model = sm.OLS(y, X, missing="drop").fit()
    df["expected_inflation"] = model.predict(X)

    return df, model


# -----------------------------
# 5. LIQUIDITY PREMIUM PROXY
# -----------------------------
def estimate_liquidity_premium(df):
    """
    Liquidity proxy:
    - TIPS yield volatility
    - Breakeven volatility
    """

    df["tips_vol"] = df["tips_5y"].rolling(20).std()
    df["bei_vol"] = df["bei_5y"].rolling(20).std()

    X = sm.add_constant(df[["tips_vol", "bei_vol"]])
    y = df["bei_5y"]

    model = sm.OLS(y, X, missing="drop").fit()
    df["liquidity_premium"] = model.predict(X)

    return df, model


# -----------------------------
# 6. INFLATION RISK PREMIUM
# -----------------------------
def compute_inflation_risk_premium(df):
    df["inflation_risk_premium"] = (
        df["bei_5y"]
        - df["expected_inflation"]
        - df["liquidity_premium"]
    )
    return df


# -----------------------------
# 7. VISUALIZATION
# -----------------------------
def plot_decomposition(df):
    plt.figure(figsize=(12, 6))

    plt.stackplot(
        df.index,
        df["expected_inflation"],
        df["inflation_risk_premium"],
        df["liquidity_premium"],
        labels=[
            "Expected Inflation",
            "Inflation Risk Premium",
            "Liquidity Premium"
        ],
        alpha=0.8
    )

    plt.plot(df.index, df["bei_5y"], color="black", lw=1.2, label="5Y Breakeven")

    plt.legend(loc="upper left")
    plt.title("5Y Breakeven Inflation Decomposition")
    plt.ylabel("Percent")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 8. MAIN EXECUTION
# -----------------------------
def main():
    df = load_fred_series()
    df = preprocess_data(df)

    df, exp_model = estimate_expected_inflation(df)
    df, liq_model = estimate_liquidity_premium(df)
    df = compute_inflation_risk_premium(df)

    plot_decomposition(df)

    print("\nExpected Inflation Model Summary:")
    print(exp_model.summary())

    print("\nLiquidity Premium Model Summary:")
    print(liq_model.summary())


if __name__ == "__main__":
    main()
