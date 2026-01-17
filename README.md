# QConv-Parametric

This project provides an independent implementation of the statistical conversion toolbox used in **Azucar, Marengo, and Settanni (2018)**. It enables the conversion of classification performance metrics into Pearson correlation coefficients ($r$).

## Project Context

The logic implemented here follows the methodology described in:
> Azucar, D., Marengo, D., & Settanni, M. (2018). Predicting the Big 5 personality traits from digital footprints on social media: A meta-analysis. *Personality and Individual Differences*, 124, 150–159.

### Methodology Excerpt
> Area Under the Receiver Operating Characteristic curve (AUROC) statistics were first converted to Cohen's $d$ and then converted from Cohen's $d$ to $r$. When studies provided specificity and sensitivity values or positive predictive values (PPV) and negative predictive values (NPV), or when studies provided sufficient information to compute these statistics, we used this information to compute odds ratios, then converted odds ratios into Cohen's $d$, and finally converted Cohen's $d$ into correlations.

---

## Conversion Logic & Formulas

### 1. AUROC to Cohen's $d$

**Function:** `auc_to_d(auc, s1=None, s2=None, p1=None)`  

**Formula:**

$$
d = \sqrt{\frac{s_1^2 + s_2^2}{p_1 s_1^2 + p_2 s_2^2}} \cdot \Phi^{-1}(\text{AUC})
$$

* **Simplified Case:** When $s_1 = s_2$ and $p_1 = p_2 = 0.5$, the formula simplifies to $d = \sqrt{2} \cdot \Phi^{-1}(\text{AUC})$.
* **Assumptions:** The underlying distributions of the two groups follow Gaussian distributions with equal variances.
* **See:** Ruscio, J. (2008). A probability-based measure of effect size: Robustness to base rates and other factors. *Psychological Methods*, 13(1), 19.

### 2. Log-Odds Ratio to Cohen's $d$

**Function:** `logodds_to_d(logodds)`  

**Formula:**

$$
d = \frac{LOR \cdot \sqrt{3}}{\pi}
$$

* **Assumptions:** The underlying distributions of the two groups follow Logistic distributions with equal variances.
* **See:** Hasselblad, V., & Hedges, L. V. (1995). Meta-analysis of screening and diagnostic tests. *Psychological Bulletin*, 117(1), 167.

### 3. Cohen's $d$ to point-biserial correlation $r_{pb}$

**Function:** `cohen_d_to_r_pb(d, p=0.5)`  

**Formula:**

$$
r_{pb} = \frac{d}{\sqrt{d^2 + \frac{1}{p_1 p_2}}}
$$

* **Assumptions:** Normality of the underlying latent distributions for the two groups.
* **See:** McGrath, R. E., & Meyer, G. J. (2006). When effect sizes disagree: The case of *r* and *d*. *Psychological Methods*, 11(4), 386.

---

### 4. Sensitivity & Specificity to Log-Odds Ratio

**Function:** `logodds_from_sens_spec(sensitivity, specificity)`

### 5. PPV & NPV to Log-Odds Ratio

**Function:** `logodds_from_ppv_npv(ppv, npv)`

---

## Installation

```bash
pip install git+[https://github.com/PavelNovikov/qconv-parametric.git](https://github.com/PavelNovikov/qconv-parametric.git)