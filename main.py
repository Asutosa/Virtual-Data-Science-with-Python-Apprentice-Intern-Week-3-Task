"""
Week 3: Statistical Analysis and Hypothesis Testing
Run this file from the folder containing Advertising.csv.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore", category=FutureWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Advertising.csv"
OUTPUT_DIR = BASE_DIR / "week3_outputs"
CHART_DIR = OUTPUT_DIR / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)
ALPHA = 0.05
sns.set_theme(style="whitegrid")

# 1. Load and prepare the data
if not DATA_PATH.exists():
    raise FileNotFoundError("Keep Advertising.csv in the same folder as main.py.")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
required = ["TV", "Radio", "Newspaper", "Sales"]
missing_columns = set(required) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
df = df[required].apply(pd.to_numeric, errors="coerce")
original_rows = len(df)
df = df.dropna().reset_index(drop=True)
removed_rows = original_rows - len(df)
if len(df) < 4:
    raise ValueError("Not enough complete rows for analysis.")

# 2. Descriptive statistics
summary = df.describe().T
summary.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

# 3. Main hypothesis: Pearson correlation between TV spend and sales
# H0: population correlation is zero. H1: population correlation is not zero.
pearson_result = stats.pearsonr(df["TV"], df["Sales"])
pearson_r, pearson_p = float(pearson_result.statistic), float(pearson_result.pvalue)
pearson_ci = pearson_result.confidence_interval(confidence_level=0.95)

# 4. Supporting test: Welch's t-test for low vs high TV spending
median_tv = df["TV"].median()
low_sales = df.loc[df["TV"] <= median_tv, "Sales"]
high_sales = df.loc[df["TV"] > median_tv, "Sales"]
t_result = stats.ttest_ind(high_sales, low_sales, equal_var=False)
t_stat, t_p = float(t_result.statistic), float(t_result.pvalue)
n_high, n_low = len(high_sales), len(low_sales)
mean_difference = float(high_sales.mean() - low_sales.mean())
var_high, var_low = high_sales.var(ddof=1), low_sales.var(ddof=1)
standard_error = np.sqrt(var_high / n_high + var_low / n_low)
welch_df = (var_high / n_high + var_low / n_low) ** 2 / (
    (var_high / n_high) ** 2 / (n_high - 1)
    + (var_low / n_low) ** 2 / (n_low - 1)
)
critical_t = stats.t.ppf(0.975, welch_df)
difference_ci = (
    mean_difference - critical_t * standard_error,
    mean_difference + critical_t * standard_error,
)

# Effect size for the Welch t-test: Cohen's d using pooled SD (descriptive effect size)
pooled_sd = np.sqrt(
    ((n_high - 1) * var_high + (n_low - 1) * var_low) / (n_high + n_low - 2)
)
cohens_d = mean_difference / pooled_sd if pooled_sd > 0 else np.nan

# 5. One-way ANOVA across low, medium and high TV spending groups
df["TV_group"] = pd.qcut(
    df["TV"], q=3, labels=["Low", "Medium", "High"], duplicates="drop"
)
anova_model = ols("Sales ~ C(TV_group)", data=df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
anova_f = float(anova_table.loc["C(TV_group)", "F"])
anova_p = float(anova_table.loc["C(TV_group)", "PR(>F)"])
anova_table.to_csv(OUTPUT_DIR / "anova_results.csv")

# Follow-up comparisons after ANOVA: Tukey HSD controls family-wise error.
tukey = pairwise_tukeyhsd(endog=df["Sales"], groups=df["TV_group"], alpha=ALPHA)
tukey_frame = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0],
)
tukey_frame.to_csv(OUTPUT_DIR / "tukey_hsd_results.csv", index=False)


# 6. Multiple regression using all three advertising channels
X = sm.add_constant(df[["TV", "Radio", "Newspaper"]])
regression = sm.OLS(df["Sales"], X).fit()
with open(OUTPUT_DIR / "regression_summary.txt", "w", encoding="utf-8") as file:
    file.write(regression.summary().as_text())

# Regression diagnostics: residual normality and heteroscedasticity.
residuals = regression.resid
jb_stat, jb_p, residual_skew, residual_kurtosis = jarque_bera(residuals)
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, regression.model.exog)
diagnostics = pd.DataFrame([
    {"Test": "Jarque-Bera residual normality", "Statistic": jb_stat, "P_value": jb_p,
     "Interpretation_at_0.05": "Evidence against normal residuals" if jb_p < ALPHA else "No significant evidence against normal residuals"},
    {"Test": "Breusch-Pagan heteroscedasticity", "Statistic": bp_lm, "P_value": bp_lm_p,
     "Interpretation_at_0.05": "Evidence of non-constant error variance" if bp_lm_p < ALPHA else "No significant evidence of non-constant error variance"},
])
diagnostics.to_csv(OUTPUT_DIR / "regression_diagnostics.csv", index=False)

correlations = df[required].corr()
correlations.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
test_results = pd.DataFrame([
    {"Test": "Pearson correlation: TV vs Sales", "Statistic": pearson_r,
     "P_value": pearson_p, "Alpha": ALPHA,
     "Decision": "Reject H0" if pearson_p < ALPHA else "Fail to reject H0"},
    {"Test": "Welch t-test: high vs low TV spending", "Statistic": t_stat,
     "P_value": t_p, "Alpha": ALPHA, "Cohens_d": cohens_d,
     "Decision": "Reject H0" if t_p < ALPHA else "Fail to reject H0"},
    {"Test": "One-way ANOVA: sales by TV group", "Statistic": anova_f,
     "P_value": anova_p, "Alpha": ALPHA,
     "Decision": "Reject H0" if anova_p < ALPHA else "Fail to reject H0"},
])
test_results.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)

# 7. Charts
chart_paths = {}
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="TV", y="Sales", scatter_kws={"alpha": 0.7})
plt.title("TV Advertising Spending and Sales")
plt.xlabel("TV advertising spending")
plt.ylabel("Sales")
plt.tight_layout()
chart_paths["scatter"] = CHART_DIR / "01_tv_spending_vs_sales.png"
plt.savefig(chart_paths["scatter"], dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], bins=20, kde=True)
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Observations")
plt.tight_layout()
chart_paths["histogram"] = CHART_DIR / "02_sales_distribution.png"
plt.savefig(chart_paths["histogram"], dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="TV_group", y="Sales")
plt.title("Sales Across TV Advertising Spending Groups")
plt.xlabel("TV spending group")
plt.ylabel("Sales")
plt.tight_layout()
chart_paths["boxplot"] = CHART_DIR / "03_sales_by_tv_group.png"
plt.savefig(chart_paths["boxplot"], dpi=160)
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation Between Advertising Channels and Sales")
plt.tight_layout()
chart_paths["heatmap"] = CHART_DIR / "04_correlation_heatmap.png"
plt.savefig(chart_paths["heatmap"], dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=regression.fittedvalues, y=regression.resid, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.title("Regression Residuals vs Fitted Values")
plt.xlabel("Fitted sales")
plt.ylabel("Residual")
plt.tight_layout()
chart_paths["residuals"] = CHART_DIR / "05_regression_residuals.png"
plt.savefig(chart_paths["residuals"], dpi=160)
plt.close()

print("\n--- Week 3 analysis completed ---")
print(f"Rows used: {len(df)} (removed {removed_rows} incomplete rows)")
print(f"Pearson r (TV vs Sales): {pearson_r:.4f}")
print(f"Pearson p-value: {pearson_p:.4g}")
print(f"95% CI for correlation: ({pearson_ci.low:.4f}, {pearson_ci.high:.4f})")
print(f"Welch t-test p-value: {t_p:.4g}")
print(f"Welch t-test Cohen's d: {cohens_d:.4f}")
print(f"ANOVA p-value: {anova_p:.4g}")
print(f"Tukey HSD results saved to: {OUTPUT_DIR / 'tukey_hsd_results.csv'}")
print(f"Regression diagnostics saved to: {OUTPUT_DIR / 'regression_diagnostics.csv'}")
print(f"Jarque-Bera p-value: {jb_p:.4g}")
print(f"Breusch-Pagan p-value: {bp_lm_p:.4g}")
print(f"Regression R-squared: {regression.rsquared:.4f}")
print(f"Charts and CSV analysis outputs saved in: {OUTPUT_DIR}")
print("No DOCX/PDF report was generated; create the report separately after reviewing the results.")
