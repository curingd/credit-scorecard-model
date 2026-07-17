# Credit Scorecard Model

A retail credit scorecard built in Python from ~1.35 million Lending Club US consumer loans, enriched with vintaged macroeconomic data, and validated with a full out-of-time (OOT) holdout: score-level Population Stability Index (PSI), characteristic-level distribution analysis, and OOT discrimination testing of the frozen model. Model output is converted into an interactive Excel scorecard with configurable base score and points-to-double-odds (PDO) settings.

## Headline results

| Metric | Unpenalised LR (in-time test) | Ridge, C = 0.0001 (in-time test) | Unpenalised LR (OOT holdout) |
| --- | --- | --- | --- |
| AUC | 0.67 | 0.66 | 0.61 |
| Gini | 0.34 | 0.32 | 0.23 |
| Accuracy | 0.60 | 0.59 | 0.67 |
| Recall | 0.64 | 0.65 | 0.44 |
| Precision | 0.27 | 0.27 | 0.31 |
| F1 | 0.38 | 0.38 | 0.36 |

**PSI (train → OOT, 10 score-decile bins): 0.82** — far above the conventional 0.25 "significant shift" threshold.

The three OOT results tell one coherent story of population drift degrading the model:

- The score distribution shifted markedly toward **lower** predicted default probabilities (PSI 0.82);
- yet the **realised** default rate in the OOT vintage was slightly **higher** (21.8% vs 19.6% in training);
- and rank-ordering power fell (AUC 0.67 → 0.61).

In other words, the model reads the recent vintage as safer while it actually defaults more often — precisely the failure mode that PSI monitoring and periodic outcome analysis exist to catch, and the trigger for redevelopment in a production setting. The headline in-time AUC reflects the deliberate constraint of the dataset: application-time variables only, with no credit bureau or behavioural data, which structurally caps discrimination.

Point-metric comparisons across the two columns should be read with care: the classification threshold is re-estimated by the K–S statistic on each sample, and the class mix differs, so the OOT accuracy uptick reflects a different precision/recall trade rather than improved performance — the threshold-free AUC/Gini row is the clean comparison.

## Validation design

The split architecture separates two distinct failure modes:

1. **Time cut first** — loans are sorted by issue date and the most recent 20% (~270k loans) are set aside as an OOT holdout before any model development.
2. **Random 80/20 split** within the remaining development window (~863k train / ~216k test) measures overfitting.
3. **Model frozen**, then scored on the OOT holdout.
4. **Stability testing:** PSI computed on score deciles (quantile bins from training-set predicted probabilities), with characteristic-level distribution comparisons for each model variable.
5. **OOT discrimination testing:** ROC/AUC, confusion matrix, and classification metrics for the frozen model on the holdout.

The random in-time split and the OOT holdout are complements, not substitutes: the former tests generalisation to unseen borrowers from the same population, the latter tests stability as the population itself shifts. Issue date is used only as the cutpoint and is not a model variable — in an OOT framework it cannot be, since it would act as a proxy for the business-cycle position of the training window.

## Data

- **Loans:** [Lending Club loan dataset for granting models](https://doi.org/10.5281/zenodo.11295916) (Ariza-Garzón, Sanz-Guerrero & Arroyo Gallardo, 2024) — 1,347,681 loans, filtered to completed or defaulted loans only, so the OOT holdout is not contaminated by unresolved outcomes.
- **Unemployment rate:** ALFRED vintaged UNRATE series, lagged one month to respect the BLS publication schedule — the model only sees the figure that was actually available at loan origination.
- **Credit spread:** Moody's Baa corporate bond spread over 10-year Treasuries (FRED, BAA10Y), monthly averaged. Tested and **dropped at variable selection** (IV < 0.02).

Unfortunately, many of the datasets are too large to store in this project: only `alfred_moodys_baa_10y_spread.csv` and `alfred_unrate.csv` are currently present in the `Datasets` folder, but these two files are sufficient to generate the intermediate datasets using the files in the `Code` folder once the LC loans dataset is downloaded from the link above.

## Methodology

- **Variable selection** by information value from weight-of-evidence (WoE) scores. Continuous variables were binned at a range of quantile-bin counts and discrete variables tested at a range of minimum category sizes to choose stable binning parameters (10 bins; minimum category size 20, with rare categories merged to "Other"). Variables with IV < 0.02 dropped: `credit_spread`, `emp_length`, `experience_c`, `addr_state`.
- **WoE transformation** of all retained variables, with bin boundaries and category–WoE maps derived from the training set only and applied unchanged to the test and OOT sets. Categories unseen in training are mapped to "Other".
- **Multicollinearity check** via variance inflation factors (max implied R² = 0.43).
- **Models:** unpenalised logistic regression with balanced class weights, benchmarked against a heavily penalised ridge specification (C = 0.0001) to confirm that reducing effective model complexity does not improve out-of-sample performance.
- **Classification threshold** chosen by the Kolmogorov–Smirnov statistic (maximising TPR − FPR), prioritising recall over precision — in a scorecard context, a missed default is more costly than a false alarm.
- **Scorecard conversion:** coefficients × WoE mapped to points in an interactive Excel workbook with user-selectable base score, base odds, and PDO, plus dropdowns for categorical characteristics.

## Repository structure

```
├── Analysis/     Full write-up (docx + pdf): methodology, results, OOT and PSI investigation
├── Code/         data_processing, credit_scorecard, X_test_psi_analysis (.py + .ipynb)
├── Datasets/     Source and derived CSVs (loans + ALFRED/FRED macro series)
├── Images/       Charts: IV selection, ROC curves (in-time and OOT), confusion matrices, PSI distributions
└── Scorecard/    Interactive Excel scorecard + bin labels/scores CSVs
```

## Reproducing the results

Requires Python 3 with `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `seaborn`, and `matplotlib`.

1. `Code/data_processing.py` — merges the loan data with the lagged ALFRED unemployment series and monthly-averaged FRED credit spread.
2. `Code/credit_scorecard.py` — performs the OOT/random split, WoE/IV variable selection, model fitting and validation, scorecard export, score-level PSI analysis, and OOT discrimination testing.
3. `Code/X_test_psi_analysis.py` — characteristic-level distribution comparisons between the training set and OOT holdout.

Each script is also provided as a Jupyter notebook with outputs preserved.

## Key findings

- **Population drift degrades the model, and the monitoring framework catches it.** The recent vintage scores materially lower for default risk (PSI 0.82) while defaulting slightly more often, and discrimination weakens (AUC 0.67 → 0.61). Characteristic-level analysis attributes the score shift to upward drift in income and FICO distributions, the treatment of unseen `title` categories (mapped to "Other", which carries a negative score contribution), partially offset by a falling unemployment rate. In production this combination — score distribution and outcomes moving in opposite directions — would trigger model redevelopment.
- **The unemployment coefficient is negative** — counterintuitive at first sight. The plausible mechanism is timing: unemployment is measured at origination, and loans originated near a cyclical unemployment peak are repaid into an improving economy (and vice versa).
- **Ridge penalisation degrades performance across the board** without zeroing out variables, indicating all retained characteristics carry genuine predictive signal at this specification.

## References

Full Harvard-style references are in `Analysis/credit_scorecard_analysis.pdf`, including Siddiqi (2006) *Credit Risk Scorecards*, the Zenodo dataset citation, and the ALFRED/FRED data sources.

---

© David Curington 2026
