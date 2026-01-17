import pytest
import numpy as np
import qconv_param as qcp


def test_logodds_to_d():
    """
    Validation: Does logodds_to_d recover d under the assumptions of the method?
    """
    np.random.seed(42)
    n_samples = 10_000
    diff = 1.0
    scale = 1.0

    # 1. Generate samples
    group1 = np.random.logistic(loc=0, scale=scale, size=n_samples)
    group2 = np.random.logistic(loc=diff, scale=scale, size=n_samples)

    # 2. Calculate observed classification metrics
    threshold = diff / 2
    sens = np.mean(group2 > threshold)
    spec = np.mean(group1 <= threshold)

    # 3. Estimate log-odds and convert to d
    lor = qcp.logodds_from_sens_spec(sens, spec)
    d_from_logodds = qcp.logodds_to_d(lor)

    # 4. Calculate d_direct using the empirical pooled standard deviation
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt((var1 + var2) / 2)

    d_direct = (np.mean(group2) - np.mean(group1)) / pooled_std

    # 5. Check recovery
    assert pytest.approx(d_from_logodds, abs=0.01) == d_direct


def test_auc_to_d():
    """
    Validation: Does auc_to_d recover d under the assumptions of the method?
    """
    np.random.seed(42)
    n1 = 5_000
    n2 = 18_000
    d_true = 0.6

    # 1. Generate normal samples
    group1 = np.random.normal(loc=0, scale=1, size=n1)
    group2 = np.random.normal(loc=d_true, scale=1, size=n2)

    # 2. Calculate Empirical AUC
    from scipy.stats import mannwhitneyu

    stat, _ = mannwhitneyu(group2, group1)
    auc_empirical = stat / (n1 * n2)

    # 3. Convert AUC back to d
    d_recovered = qcp.auc_to_d(auc_empirical)

    # 4. Calculate d_direct using the pooled standard deviation
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    d_direct = (np.mean(group2) - np.mean(group1)) / pooled_std

    # 5. Check recovery
    assert pytest.approx(d_recovered, abs=0.02) == d_direct
    assert pytest.approx(d_direct, abs=0.02) == d_true


def test_cohen_d_to_r_pb_unbalanced():
    """
    Validation: Does cohen_d_to_r_pb correctly recover point-biserial correlation
    under the assumptions of the method?
    """
    np.random.seed(42)
    n1 = 5_000
    n2 = 15_000
    d_true = 0.5

    # 1. Generate data for two groups with d_true separation
    g1 = np.random.normal(0, 1, n1)
    g2 = np.random.normal(d_true, 1, n2)

    # 2. Create the continuous (y) and binary (x) variables
    y = np.concatenate([g1, g2])
    x = np.concatenate([np.zeros(n1), np.ones(n2)])

    # 3. Calculate observed d from the samples
    mean_diff = np.mean(g2) - np.mean(g1)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    # Pooled SD for unequal groups
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d_observed = mean_diff / s_pooled

    # 4. Calculate observed point-biserial correlation directly
    from scipy.stats import pointbiserialr

    r_observed, _ = pointbiserialr(x, y)

    # 5. Convert observed d back to r using the function with group counts
    r_recovered = qcp.cohen_d_to_r_pb(d_observed, p=n1 / (n1 + n2))

    # 6. Check recovery
    assert pytest.approx(r_recovered, abs=0.01) == r_observed
