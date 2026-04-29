"""
Corporate Credit Spread Analyzer
=================================
Collects and cleans historical corporate bond spread data across investment-grade
and high-yield issuers, builds regression models linking credit spreads to leverage
ratios, interest coverage, and macroeconomic indicators, visualizes spread compression
and widening cycles, and evaluates model accuracy in predicting spread direction changes.

Tools: Python, Pandas, Statsmodels, Scikit-learn, Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings, os

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. DATA COLLECTION & CLEANING
#    Simulate realistic historical corporate bond spread data
#    modeled after FRED (ICE BofA indices) and TRACE-style data
# ============================================================

def generate_spread_data(n_months=180):
    """Generate ~15 years of monthly credit spread data with realistic dynamics."""
    dates = pd.date_range("2010-01-01", periods=n_months, freq="MS")

    # Macro indicators
    fed_funds = np.cumsum(np.random.normal(0, 0.05, n_months)) + 1.5
    fed_funds = np.clip(fed_funds, 0.0, 5.5)

    gdp_growth = 2.0 + np.cumsum(np.random.normal(0, 0.15, n_months))
    gdp_growth = np.clip(gdp_growth, -3.0, 6.0)

    unemployment = 5.0 + np.cumsum(np.random.normal(0, 0.1, n_months))
    unemployment = np.clip(unemployment, 3.0, 12.0)

    vix = 18 + np.cumsum(np.random.normal(0, 1.0, n_months))
    vix = np.clip(vix, 10, 50)

    # Corporate fundamentals (sector-level aggregates)
    leverage_ratio = 2.5 + np.cumsum(np.random.normal(0, 0.03, n_months))
    leverage_ratio = np.clip(leverage_ratio, 1.5, 5.0)

    interest_coverage = 6.0 + np.cumsum(np.random.normal(0, 0.05, n_months))
    interest_coverage = np.clip(interest_coverage, 2.0, 12.0)

    default_rate = 1.5 + np.cumsum(np.random.normal(0, 0.05, n_months))
    default_rate = np.clip(default_rate, 0.2, 8.0)

    # Credit spreads driven by fundamentals + noise
    ig_spread = (
        80
        + 25 * (leverage_ratio - 2.5)
        - 8 * (interest_coverage - 6.0)
        + 3 * (vix - 18)
        + 5 * (unemployment - 5.0)
        - 4 * (gdp_growth - 2.0)
        + np.random.normal(0, 10, n_months)
    )
    ig_spread = np.clip(ig_spread, 40, 350)

    hy_spread = (
        400
        + 80 * (leverage_ratio - 2.5)
        - 20 * (interest_coverage - 6.0)
        + 12 * (vix - 18)
        + 15 * (unemployment - 5.0)
        - 10 * (gdp_growth - 2.0)
        + 30 * (default_rate - 1.5)
        + np.random.normal(0, 40, n_months)
    )
    hy_spread = np.clip(hy_spread, 200, 1200)

    df = pd.DataFrame({
        "date": dates,
        "ig_spread_bps": ig_spread,
        "hy_spread_bps": hy_spread,
        "leverage_ratio": leverage_ratio,
        "interest_coverage": interest_coverage,
        "default_rate_pct": default_rate,
        "fed_funds_rate": fed_funds,
        "gdp_growth_pct": gdp_growth,
        "unemployment_pct": unemployment,
        "vix": vix,
    })
    df.set_index("date", inplace=True)
    return df


def clean_data(df):
    """Clean and preprocess the spread data."""
    # Remove any NaN rows
    df = df.dropna()

    # Winsorize extreme values at 1st/99th percentile
    for col in ["ig_spread_bps", "hy_spread_bps"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # Compute derived features
    df["spread_ratio"] = df["hy_spread_bps"] / df["ig_spread_bps"]
    df["ig_spread_chg"] = df["ig_spread_bps"].diff()
    df["hy_spread_chg"] = df["hy_spread_bps"].diff()
    df["ig_spread_ma6"] = df["ig_spread_bps"].rolling(6).mean()
    df["hy_spread_ma6"] = df["hy_spread_bps"].rolling(6).mean()

    # Direction labels (1 = widening, 0 = tightening)
    df["ig_direction"] = (df["ig_spread_chg"] > 0).astype(int)
    df["hy_direction"] = (df["hy_spread_chg"] > 0).astype(int)

    df = df.dropna()
    return df


# ============================================================
# 2. REGRESSION MODEL
#    Link credit spreads to fundamentals & macro indicators
# ============================================================

def build_regression_model(df):
    """OLS regression: credit spreads ~ leverage, coverage, macro."""
    features = [
        "leverage_ratio", "interest_coverage", "default_rate_pct",
        "fed_funds_rate", "gdp_growth_pct", "unemployment_pct", "vix"
    ]

    print("=" * 65)
    print("REGRESSION ANALYSIS: Credit Spreads vs. Fundamentals")
    print("=" * 65)

    results = {}
    for target, label in [("ig_spread_bps", "Investment-Grade"), ("hy_spread_bps", "High-Yield")]:
        X = sm.add_constant(df[features])
        y = df[target]
        model = sm.OLS(y, X).fit(cov_type="HC1")  # robust SEs

        print(f"\n--- {label} Spread Model ---")
        print(f"R-squared:      {model.rsquared:.4f}")
        print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic:    {model.fvalue:.2f} (p={model.f_pvalue:.2e})")
        print(f"\nCoefficients:")
        for feat in features:
            coef = model.params[feat]
            pval = model.pvalues[feat]
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {feat:25s}  {coef:8.3f}  (p={pval:.4f}) {sig}")

        results[target] = model

    return results


# ============================================================
# 3. VISUALIZATION
#    Spread compression/widening cycles & mispricing detection
# ============================================================

def create_visualizations(df):
    """Generate all charts and save to output directory."""

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Corporate Credit Spread Analyzer", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: Historical Spreads ---
    ax = axes[0, 0]
    ax.plot(df.index, df["ig_spread_bps"], label="IG Spread", color="#2166ac", linewidth=1.2)
    ax.plot(df.index, df["ig_spread_ma6"], label="6M MA", color="#2166ac", linewidth=2, alpha=0.5, linestyle="--")
    ax.fill_between(df.index, 0, df["ig_spread_bps"], alpha=0.1, color="#2166ac")
    ax.set_title("Investment-Grade Credit Spreads (bps)")
    ax.set_ylabel("Spread (bps)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[0, 1]
    ax.plot(df.index, df["hy_spread_bps"], label="HY Spread", color="#b2182b", linewidth=1.2)
    ax.plot(df.index, df["hy_spread_ma6"], label="6M MA", color="#b2182b", linewidth=2, alpha=0.5, linestyle="--")
    ax.fill_between(df.index, 0, df["hy_spread_bps"], alpha=0.1, color="#b2182b")
    ax.set_title("High-Yield Credit Spreads (bps)")
    ax.set_ylabel("Spread (bps)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Panel 2: Spread Ratio & Compression/Widening ---
    ax = axes[1, 0]
    ratio = df["spread_ratio"]
    median_ratio = ratio.median()
    ax.plot(df.index, ratio, color="#4a1486", linewidth=1.2)
    ax.axhline(median_ratio, color="gray", linestyle="--", label=f"Median = {median_ratio:.2f}")
    ax.fill_between(df.index, ratio, median_ratio,
                    where=(ratio > median_ratio), alpha=0.3, color="#d73027", label="Wide (HY stressed)")
    ax.fill_between(df.index, ratio, median_ratio,
                    where=(ratio < median_ratio), alpha=0.3, color="#1a9850", label="Compressed")
    ax.set_title("HY/IG Spread Ratio — Compression & Widening Cycles")
    ax.set_ylabel("Ratio")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Panel 3: Mispricing Detection ---
    ax = axes[1, 1]
    # Fit simple model to get "fair value" spread
    features = ["leverage_ratio", "interest_coverage", "vix", "unemployment_pct", "gdp_growth_pct"]
    X = df[features].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=1.0).fit(X_s, df["hy_spread_bps"].values)
    fair_value = model.predict(X_s)
    residual = df["hy_spread_bps"].values - fair_value
    residual_std = residual / residual.std()

    ax.bar(df.index, residual_std, width=25, color=np.where(residual_std > 1, "#d73027",
           np.where(residual_std < -1, "#1a9850", "#bababa")), alpha=0.7)
    ax.axhline(1, color="#d73027", linestyle="--", alpha=0.5, label="Overpriced risk (>1σ)")
    ax.axhline(-1, color="#1a9850", linestyle="--", alpha=0.5, label="Underpriced risk (<-1σ)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("HY Spread Mispricing vs. Fundamentals (Std. Residuals)")
    ax.set_ylabel("Std. Residual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Panel 4: Fundamentals ---
    ax = axes[2, 0]
    ax2 = ax.twinx()
    ax.plot(df.index, df["leverage_ratio"], color="#e66101", label="Leverage Ratio", linewidth=1.5)
    ax2.plot(df.index, df["interest_coverage"], color="#5e3c99", label="Interest Coverage", linewidth=1.5)
    ax.set_ylabel("Leverage Ratio (Debt/EBITDA)", color="#e66101")
    ax2.set_ylabel("Interest Coverage (x)", color="#5e3c99")
    ax.set_title("Corporate Fundamentals Over Time")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Panel 5: Macro Environment ---
    ax = axes[2, 1]
    ax.plot(df.index, df["vix"], label="VIX", color="#d73027", alpha=0.7)
    ax.plot(df.index, df["fed_funds_rate"] * 5, label="Fed Funds (×5)", color="#4575b4", alpha=0.7)
    ax.plot(df.index, df["unemployment_pct"] * 3, label="Unemployment (×3)", color="#91bfdb", alpha=0.7)
    ax.set_title("Macro Indicators")
    ax.set_ylabel("Level (scaled)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "credit_spread_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCharts saved to: {path}")
    return path


# ============================================================
# 4. DIRECTION PREDICTION MODEL
#    Predict spread widening vs. tightening
# ============================================================

def build_direction_model(df):
    """Train a classifier to predict spread direction changes."""
    features = [
        "leverage_ratio", "interest_coverage", "default_rate_pct",
        "fed_funds_rate", "gdp_growth_pct", "unemployment_pct", "vix",
        "ig_spread_bps", "spread_ratio"
    ]

    # Add lagged features
    for lag in [1, 3]:
        df[f"hy_spread_lag{lag}"] = df["hy_spread_bps"].shift(lag)
        df[f"hy_chg_lag{lag}"] = df["hy_spread_chg"].shift(lag)
        features += [f"hy_spread_lag{lag}", f"hy_chg_lag{lag}"]

    df = df.dropna()
    X = df[features]
    y = df["hy_direction"]

    # Time-series aware split (no shuffle)
    split_idx = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print("DIRECTION PREDICTION: HY Spread Widening vs. Tightening")
    print("=" * 65)
    print(f"\nTrain size: {len(X_train)} months | Test size: {len(X_test)} months")
    print(f"Out-of-sample directional accuracy: {acc:.1%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Tightening", "Widening"]))

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("Top 5 Features by Importance:")
    for feat, imp in importances.head(5).items():
        print(f"  {feat:25s}  {imp:.4f}")

    return clf, acc


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("  CORPORATE CREDIT SPREAD ANALYZER")
    print("  Python | Pandas | Statsmodels | Scikit-learn | Matplotlib")
    print("=" * 65)

    # Step 1: Generate and clean data
    print("\n[1/4] Generating historical credit spread data (15 years, monthly)...")
    df = generate_spread_data(n_months=180)
    df = clean_data(df)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    # Save cleaned data
    csv_path = os.path.join(OUTPUT_DIR, "credit_spread_data.csv")
    df.to_csv(csv_path)
    print(f"  Data saved to: {csv_path}")

    # Step 2: Regression analysis
    print("\n[2/4] Building regression models...")
    reg_results = build_regression_model(df)

    # Step 3: Visualization
    print("\n[3/4] Creating visualizations...")
    chart_path = create_visualizations(df)

    # Step 4: Direction prediction
    print("\n[4/4] Training direction prediction model...")
    clf, accuracy = build_direction_model(df)

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  IG Spread — Mean: {df['ig_spread_bps'].mean():.0f} bps, "
          f"Min: {df['ig_spread_bps'].min():.0f}, Max: {df['ig_spread_bps'].max():.0f}")
    print(f"  HY Spread — Mean: {df['hy_spread_bps'].mean():.0f} bps, "
          f"Min: {df['hy_spread_bps'].min():.0f}, Max: {df['hy_spread_bps'].max():.0f}")
    print(f"  IG Model R²: {reg_results['ig_spread_bps'].rsquared:.4f}")
    print(f"  HY Model R²: {reg_results['hy_spread_bps'].rsquared:.4f}")
    print(f"  Direction Accuracy: {accuracy:.1%}")
    print(f"\n  Outputs: {OUTPUT_DIR}/")
    print(f"    - credit_spread_data.csv")
    print(f"    - credit_spread_analysis.png")
    print("=" * 65)


if __name__ == "__main__":
    main()