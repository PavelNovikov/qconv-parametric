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
    z_score = norm.ppf(auc)

    params = [s1, s2, p1]
    any_provided = any(x is not None for x in params)
    all_provided = all(x is not None for x in params)

    # Balanced Case
    # Assumes p1=0.5, p2=0.5, and s1=s2. Scaling factor simplifies to sqrt(2).
    if not any_provided:
        return np.sqrt(2) * z_score

    # Parametric Case
    # Requires all context arguments to avoid hidden default assumptions.
    if not all_provided:
        raise ValueError(
            "Parametric derivation requires all context arguments: s1, s2, and p1. "
            "For the balanced case, leave all arguments as None."
        )

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
