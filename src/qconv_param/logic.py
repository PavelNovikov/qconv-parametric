import math
import numpy as np
from scipy.stats import norm


def cohen_d_to_r_pb(d, p=0.5):
    """
    Converts Cohen's d to a point-biserial correlation coefficient.

    Args:
        d (float): The standardized mean difference (Cohen's d).
        p (float): The proportion of the total sample in the focal group.
                   Defaults to 0.5 (equal group sizes).

    Returns:
        float: The estimated point-biserial correlation (r_pb).

    Reference:
        McGrath, R. E., & Meyer, G. J. (2006). When effect sizes disagree:
        The case of r and d. Psychological Methods, 11(4), 386.
    """
    h = 1 / (p * (1 - p))

    r_pb = d / math.sqrt(d**2 + h)
    return r_pb


def auc_to_d(auc, s1=None, s2=None, p1=None):
    """
    Converts AUC to Cohen's d.

    Simplifications:
    - If s1 = s2 (equal variances), base rates cancel out
    - If p1 = 0.5 (balanced classes), variance differences cancel out
    - Either condition reduces to d = sqrt(2) * z

    Args:
        auc (float): The Area Under the Curve (0.5 to 1.0).
        s1 (float, optional): Standard deviation of the first class.
        s2 (float, optional): Standard deviation of the second class.
        p1 (float, optional): Proportion (base rate) of the first class (0 to 1).

    Returns:
        float: The derived Cohen's d value.

    Reference:
        Ruscio, J. (2008). A probability-based measure of effect size:
        Robustness to base rates and other factors. Psychological Methods, 13(1), 19.
    """

    # Validate that the AUC lies within its meaningful range.
    if not (0.5 <= auc <= 1.0):
        raise ValueError("AUC must be between 0.5 and 1.0")

    # Convert AUC to a standard normal z-score.
    z_score = norm.ppf(auc)

    # If no contextual parameters are provided, assume a fully balanced case
    # with equal variances and equal base rates.
    if s1 is None and s2 is None and p1 is None:
        return np.sqrt(2) * z_score

    # If the class standard deviations are equal, the base rates cancel out
    # and the simplified conversion applies.
    if s1 is not None and s2 is not None and np.isclose(s1, s2):
        return np.sqrt(2) * z_score

    # If the class base rates are balanced, variance differences cancel out
    # and the simplified conversion applies.
    if p1 is not None and np.isclose(p1, 0.5):
        return np.sqrt(2) * z_score

    # If no simplification applies, all contextual parameters are required
    # to avoid making implicit assumptions.
    if s1 is None or s2 is None or p1 is None:
        raise ValueError(
            "Full parametric conversion requires s1, s2, and p1 "
            "unless an exact simplification condition is met."
        )

    # Compute Cohen's d using the full parametric formula.
    p2 = 1 - p1
    numerator = s1**2 + s2**2
    denominator = (p1 * s1**2) + (p2 * s2**2)

    return np.sqrt(numerator / denominator) * z_score


def logodds_to_d(logodds):
    """
    Converts LOg-Odds Ratio to Cohen's d.

    Args:
        logodds (float): The Log-Odds Ratio.

    Returns:
        float: The derived Cohen's d value.

    Reference:
        Hasselblad, V., & Hedges, L. V. (1995). Meta-analysis of screening
        and diagnostic tests. Psychological Bulletin, 117(1), 167.
    """
    factor = math.sqrt(3) / math.pi
    return logodds * factor


def logodds_from_sens_spec(sensitivity, specificity):
    """
    Calculates the Log-Odds Ratio from Sensitivity and Specificity.

    Formula: ln((Sens * Spec) / ((1 - Sens) * (1 - Spec)))
    """
    # Using clip to avoid division by zero or log(0)
    eps = 1e-9
    sens = np.clip(sensitivity, eps, 1 - eps)
    spec = np.clip(specificity, eps, 1 - eps)

    odds_ratio = (sens * spec) / ((1 - sens) * (1 - spec))
    return math.log(odds_ratio)


def logodds_from_ppv_npv(ppv, npv):
    """
    Calculates the Log-Odds Ratio from Positive Predictive Value (PPV)
    and Negative Predictive Value (NPV).

    Formula: ln((PPV * NPV) / ((1 - PPV) * (1 - NPV)))
    """
    eps = 1e-9
    p = np.clip(ppv, eps, 1 - eps)
    n = np.clip(npv, eps, 1 - eps)

    odds_ratio = (p * n) / ((1 - p) * (1 - n))
    return math.log(odds_ratio)
