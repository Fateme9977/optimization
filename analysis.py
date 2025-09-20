import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

def run_statistical_tests(df, model1_col, model2_col, alpha=0.05):
    """
    Performs paired t-test and Wilcoxon signed-rank test on error metrics
    from two different models.

    Args:
        df (pd.DataFrame): DataFrame containing the error metrics for each model.
        model1_col (str): Column name for model 1's errors.
        model2_col (str): Column name for model 2's errors.
        alpha (float): Significance level.
    """
    errors1 = df[model1_col].dropna()
    errors2 = df[model2_col].dropna()

    print(f"--- Running Statistical Comparison between {model1_col} and {model2_col} ---")

    # Paired t-test
    t_stat, p_val_t = stats.ttest_rel(errors1, errors2)
    print(f"Paired t-test: t-statistic={t_stat:.4f}, p-value={p_val_t:.4f}")
    if p_val_t < alpha:
        print("  -> Result is statistically significant (reject H0).")
    else:
        print("  -> Result is not statistically significant (fail to reject H0).")

    # Wilcoxon signed-rank test
    w_stat, p_val_w = stats.wilcoxon(errors1, errors2)
    print(f"\nWilcoxon signed-rank test: W-statistic={w_stat:.4f}, p-value={p_val_w:.4f}")
    if p_val_w < alpha:
        print("  -> Result is statistically significant (reject H0).")
    else:
        print("  -> Result is not statistically significant (fail to reject H0).")

def get_bootstrap_ci(data, n_resamples=1000, alpha=0.05):
    """
    Calculates the bootstrap confidence interval for the mean of a dataset.
    """
    n = len(data)
    resampled_means = np.zeros(n_resamples)
    for i in range(n_resamples):
        resample = np.random.choice(data, size=n, replace=True)
        resampled_means[i] = np.mean(resample)

    lower_bound = np.percentile(resampled_means, (alpha / 2) * 100)
    upper_bound = np.percentile(resampled_means, (1 - alpha / 2) * 100)

    print(f"\nBootstrap 95% CI for the mean: ({lower_bound:.4f}, {upper_bound:.4f})")
    return lower_bound, upper_bound

def run_power_analysis(effect_size=0.1, alpha=0.05, power=0.8):
    """
    Performs a simple power analysis to determine the required sample size.
    """
    print("\n--- Power Analysis ---")
    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative='two-sided'
    )
    print(f"Required sample size to detect effect size={effect_size} with power={power}: {np.ceil(sample_size)}")
    return sample_size

if __name__ == '__main__':
    # --- Example Usage ---
    # Create dummy error data for two models
    np.random.seed(42)
    data = {
        'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=100, freq='h')),
        'baseline_error': np.random.normal(loc=0.5, scale=0.1, size=100),
        'eamtf_error': np.random.normal(loc=0.48, scale=0.12, size=100)
    }
    error_df = pd.DataFrame(data).set_index('timestamp')

    # 1. Run statistical tests
    run_statistical_tests(error_df, 'baseline_error', 'eamtf_error')

    # 2. Get bootstrap CI for one of the error sets
    get_bootstrap_ci(error_df['eamtf_error'])

    # 3. Run power analysis
    # Let's say we want to detect a 1% change in RMSE, and our baseline RMSE is ~0.5
    # This corresponds to a small effect size.
    run_power_analysis(effect_size=0.1)
