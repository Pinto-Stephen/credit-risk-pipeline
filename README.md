# Credit Default Risk Pipeline — PD / LGD / EAD + Scorecard

An end-to-end, regulator-style retail credit-risk modelling pipeline built on the public
**Lending Club** loan book (2007–2018). It produces the three Basel risk parameters —
**PD** (Probability of Default), **LGD** (Loss Given Default), **EAD** (Exposure at Default) —
combines them into **Expected Loss (ECL = PD × LGD × EAD)**, and delivers a banker-readable
**scorecard** plus a **Streamlit underwriting app** and a **one-page Model Validation Report**.

---

## 1. Business problem

A lender must decide, at the point of application, whether to **approve / decline / refer**
a loan, and must hold regulatory capital and IFRS-9 provisions against expected losses. That
requires answering three questions for every borrower:

| Parameter | Question | This repo |
|-----------|----------|-----------|
| **PD**  | How likely is default in the performance window? | Logistic-regression WoE **scorecard** (champion) + LightGBM **challenger** |
| **LGD** | If they default, what fraction of exposure is unrecoverable? | LightGBM regressor, `1 − recoveries / loan_amnt` |
| **EAD** | If they default, what balance is outstanding? | LightGBM regressor, `loan_amnt − total_rec_prncp` |

**Expected Loss** for provisioning = `PD × LGD × EAD`.

### Target definition (the most important modelling decision)
- **Good (target = 0):** `Fully Paid`
- **Bad  (target = 1):** `Charged Off`, `Default`
- **Excluded:** `Current`, `In Grace Period`, `Late (...)` — their economic outcome is not yet
  known; including them would bias the performance window.

Performance cohort after filtering: **1,345,350 loans**, baseline default rate **≈ 20%**.

---

## 2. Repository structure

```
credit-risk-pipeline/
├── 01_eda.ipynb                      # Ingest, target cohort, imputation, temporal dev/OOT split
├── 02_feature_engineering.ipynb      # WoE / IV binning, leakage screening, monotonicity audit
├── 03_traditonal_model_scorecard.ipynb  # Champion: statsmodels logit → points scorecard
├── 04_ml_challenger.ipynb            # Challenger: LightGBM + SHAP explainability
├── 05_lgd_ead_modelling.ipynb        # LGD & EAD regressors on the defaulted population
├── 06_psi_monitoring.ipynb           # Out-of-time PSI drift monitoring
├── app.py                            # Streamlit decision engine (PD / LGD / EAD / ECL)
├── requirements.txt
├── README.md
├── Model_Validation_Report.pdf       # One-page MRM-style sign-off (generated, see §8)
└── data/
    ├── raw/        accepted_2007_to_2018Q4.csv.gz   # (git-ignored — download from Kaggle)
    └── processed/  cohorts, engineered features, model_artifacts/   (CSVs git-ignored)
```

**Data source:** Lending Club *accepted* loans, Kaggle dataset
`wordsforthewise/lending-club` → `accepted_2007_to_2018Q4.csv.gz`. Place it in `data/raw/`.

---

## 3. How to run

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Reproduce the pipeline end-to-end (notebooks are ordered and chained by artifacts):
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_feature_engineering.ipynb
#   ... 03, 04, 05, 06 in order

# Launch the underwriting app (uses the small artifacts in data/processed/, which ARE committed):
streamlit run app.py
```

The Streamlit app loads only the lightweight artifacts (`champion_scorecard.pkl` ≈ 11 KB after
`remove_data()`, `lgd_ead_engines.pkl`, WoE mappings, bin edges, scaling params), so it runs on a
fresh clone **without** re-running the heavy notebooks.

---

## 4. Data dictionary (modelled features)

The champion uses **21 application-time features** (everything known at the decision point).
Bureau/behavioural attributes are point-in-time at origination.

| Feature | Meaning |
|---|---|
| `loan_amnt` | Requested loan amount ($) |
| `term` | 36 or 60 months |
| `int_rate` | Lending Club assigned APR (%) |
| `grade`, `sub_grade` | LC internal risk rating (A1…G5) |
| `home_ownership` | RENT / MORTGAGE / OWN / OTHER |
| `annual_inc` | Borrower annual income ($) |
| `verification_status` | Income verification level |
| `dti` | Debt-to-income ratio (%) |
| `fico_range_low` | FICO score (lower bound) |
| `tot_cur_bal`, `avg_cur_bal` | Total / average current balance across accounts |
| `total_rev_hi_lim` | Total revolving high credit limit |
| `acc_open_past_24mths` | Accounts opened in last 24 months |
| `bc_open_to_buy`, `bc_util` | Bankcard available credit / utilisation |
| `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl` | Account-age / recency signals |
| `mort_acc` | Number of mortgage accounts |
| `num_actv_rev_tl` | Active revolving trade lines |

### Excluded — **target leakage** (used only to build LGD/EAD targets, never as PD inputs)
| Field | Why excluded |
|---|---|
| `total_rec_prncp` | Cumulative principal repaid *to date* — a post-origination outcome. Raw IV ≈ **2.18** ⇒ textbook leakage. |
| `recoveries` | Post charge-off recovery cash flow — only exists *after* default. |

### Excluded — **redundancy / multicollinearity**
| Field | Why excluded |
|---|---|
| `funded_amnt` | Equals `loan_amnt` on ~all records. |
| `installment` | Deterministic function of `loan_amnt`, `int_rate`, `term` (and weak IV). |
| `fico_range_high` | Always `fico_range_low + 4`. |

---

## 5. Feature-engineering log — WoE / IV

Every feature is coarse-binned (quantile bins for numerics, native categories for strings) and
transformed to **Weight of Evidence**: `WoE = ln(%goods / %bads)`. Predictive power is screened by
**Information Value**: `IV = Σ (%goods − %bads) × WoE`. Features with **IV < 0.02 are dropped**.

| IV band | Interpretation | Action |
|---|---|---|
| < 0.02 | unpredictive | drop |
| 0.02 – 0.10 | weak | keep if stable |
| 0.10 – 0.30 | medium | keep |
| 0.30 – 0.50 | strong | keep |
| > 0.50 | suspiciously high | **audit for leakage** |

**Retained features and IV (post leakage-removal):**

| Feature | IV | Band |
|---|---|---|
| `sub_grade` | 0.502 | strong (LC's own rating — audited, legitimate) |
| `grade` | 0.468 | strong |
| `int_rate` | 0.423 | strong |
| `term` | 0.197 | medium |
| `fico_range_low` | 0.113 | medium |
| `acc_open_past_24mths` | 0.077 | weak |
| `dti` | 0.073 | weak |
| `verification_status` | 0.053 | weak |
| ... 13 more weak-but-stable features | 0.02–0.05 | weak |

The full per-bin **WoE → Coefficient → Points** table is exported to
`data/processed/model_artifacts/scorecard_point_table.csv` (133 rows).

### Scorecard scaling
Standard points arithmetic: **PDO = 20** (points to double the odds), **factor = 20/ln 2 ≈ 28.85**,
**offset** anchored at 600 points for 50:1 good:bad odds. **Base score = 528.**
`score = base + Σ (bin points)`, where each bin's points `= −factor × β × WoE`, so a **higher
score = safer**. Example: `sub_grade A1 = +42 pts`, `G5 = −32 pts`; `dti < 10.6% = +5 pts`,
`dti > 25.6% = −5 pts`.

---

## 6. Model card

### Champion — Logistic-Regression WoE Scorecard (the production model)
- **Why champion:** fully explainable, monotonic, auditable coefficients, trivially deployable as a
  points table. This is what regulators (Basel A-IRB / IFRS-9) accept.
- **Train/test:** 783,797 / 335,914 (70/30, stratified, dev cohort only).

| Metric | Champion | Challenger (LightGBM) | Lift |
|---|---|---|---|
| **AUC** | 0.7145 | 0.721 | +0.006 |
| **Gini** | 0.4291 | 0.4420 | **+0.0129** |
| **KS** | 31.07% | 32.06% | +0.99 pt |

- **Deploy gate KS ≥ 30: PASS.**
- **Decision:** the challenger adds only ~1 Gini point. That marginal lift does **not** justify
  sacrificing the scorecard's explainability, so the **logistic scorecard is retained as champion**;
  the GBDT is kept as a benchmark and challenger.

> **Before vs after leakage removal:** the original build reported Champion KS 80.5 / Challenger KS
> 99.6 because `total_rec_prncp` and `recoveries` had leaked in. Removing them brought the model to
> the realistic figures above — a deliberate, documented correction.

### Challenger — LightGBM + SHAP
Ingests raw (un-binned) features; SHAP supplies model-agnostic, locally-faithful attribution used to
auto-generate **ECOA/FCRA adverse-action reason codes**. Top global drivers: `sub_grade`, `term`,
`int_rate`, `dti`, `acc_open_past_24mths`.

### LGD / EAD — LightGBM regressors (defaulted population, n = 220,556)
| Model | Target | MAE | R² |
|---|---|---|---|
| **LGD** | `1 − recoveries/loan_amnt` (clipped [0,1]) | 0.0627 | 0.015 |
| **EAD** | `loan_amnt − total_rec_prncp` (clipped) | $2,394.80 | 0.793 |

LGD R² is low **by nature** — recoveries on unsecured consumer loans are near-zero, so LGD clusters
tightly around **0.92** with little explainable variance (a well-known industry phenomenon). EAD is
highly predictable (R² 0.79) as it is anchored by loan size.

### Known limitations (disclosed)
- `grade`/`sub_grade`/`int_rate` are mutually collinear (they encode the same LC pricing signal);
  `grade`'s coefficient is insignificant (p = 0.27) and `int_rate` shows a mild sign flip. A future
  iteration would keep one of the three. They are retained here for transparency.
- LGD/EAD use point regression, not a Tweedie/Beta GLM (LightGBM has no native beta objective).

---

## 7. Monitoring plan (PSI)

**Population Stability Index** between the development cohort (≤2016) and a genuine
**out-of-time** sample (2017–2018, 225,639 loans): `PSI = Σ (A% − E%) × ln(A%/E%)`.
Thresholds: **< 0.10 stable · 0.10–0.25 monitor · > 0.25 retrain**.

| Result | Value | Status |
|---|---|---|
| **Output-score PSI** (dev PD vs OOT PD) | **0.0059** | **STABLE** |
| Worst feature (`bc_util`) | 0.154 | MINOR SHIFT |
| Second (`bc_open_to_buy`) | 0.104 | MINOR SHIFT |
| All 19 other model features | < 0.05 | STABLE |

No feature shows critical drift; the score distribution is stable out-of-time. Full table:
`data/processed/model_artifacts/psi_monitoring_report.csv`.

**Operating cadence:** recompute output + feature PSI monthly. **Triggers:** any feature PSI > 0.25,
or output PSI > 0.25, or KS on fresh vintages < 30 → escalate to recalibration / redevelopment.

---

## 8. Streamlit decision engine (`app.py`)

Input borrower attributes → outputs **scorecard points, PD, LGD, EAD, ECL**, an **approve / manual-
review / decline** decision (PD cut-offs 8% / 15%), an **IFRS-9 stage** proxy, and **adverse-action
reason codes** (the features that subtract the most points from the score). Run with
`streamlit run app.py`.

---

## 9. Concepts referenced (for interview defence)

**Basel III / A-IRB** (bank models PD, LGD, EAD) vs **F-IRB** (PD only). **WoE / IV** feature
transform and screening. **KS / Gini / AUC** discrimination metrics (`Gini = 2·AUC − 1`).
**PSI** stability monitoring. **SHAP** for regulator-acceptable attribution. **IFRS-9 ECL** staging
(Stage 1 = 12-month ECL, Stage 2/3 = lifetime). See `Model_Validation_Report.pdf` for the MRM-style
sign-off.
