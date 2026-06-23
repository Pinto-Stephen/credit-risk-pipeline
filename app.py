import os
import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Credit Risk Decision Engine", layout="wide")
st.title("🏦 Basel III / IFRS 9 Credit Decision Engine")
st.caption("Champion scorecard (logistic regression + WoE) · ECL = PD × LGD × EAD")

PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "model_artifacts")


# ── Artifact loading ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, "champion_scorecard.pkl"), "rb") as f:
        pd_model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "lgd_ead_engines.pkl"), "rb") as f:
        loss_engines = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "scaling_params.json"), "r") as f:
        scaling = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "bin_edges.pkl"), "rb") as f:
        bin_edges = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "woe_mappings.pkl"), "rb") as f:
        woe_mappings = pickle.load(f)
    return pd_model, loss_engines, scaling, bin_edges, woe_mappings


try:
    pd_model, loss_engines, scaling, bin_edges, woe_mappings = load_artifacts()
    lgd_model    = loss_engines['lgd']
    ead_model    = loss_engines['ead']
    ead_features = loss_engines['features']   # exact feature list saved in notebook 05
except Exception as e:
    st.error(f"Artifacts missing. Run notebooks 02 → 05 first.\n\n`{e}`")
    st.stop()


# ── Inference pipeline ────────────────────────────────────────────────────────
def apply_woe_pipeline(raw: dict) -> dict:
    """
    Bin each raw feature value using training-time cut-points (bin_edges),
    then look up the WoE value from woe_mappings.
    Returns {woe_col: woe_value} for all features the model knows about.
    Missing bins default to 0.0 (neutral WoE).
    """
    woe_row = {}
    for col, val in raw.items():
        bin_col = f"bin_{col}"
        woe_col = f"{bin_col}_WoE"
        if bin_col not in woe_mappings:
            continue
        if col in bin_edges:                           # numeric: apply saved cut-points
            try:
                label = str(pd.cut([val], bins=bin_edges[col], include_lowest=True)[0])
            except Exception:
                label = "nan"
        else:                                          # categorical: label IS the raw value
            label = str(val)
        woe_row[woe_col] = woe_mappings[bin_col].get(label, 0.0)
    return woe_row


def score_applicant(woe_row: dict) -> tuple[float, int]:
    """Run WoE vector through champion logistic regression. Returns (PD, scorecard_points)."""
    model_cols = [c for c in pd_model.params.index if c != "const"]
    X = pd.DataFrame([[woe_row.get(c, 0.0) for c in model_cols]], columns=model_cols)
    X = sm.add_constant(X, has_constant="add")
    pd_pred  = float(pd_model.predict(X).iloc[0])
    log_odds = np.log((1 - pd_pred) / pd_pred)
    points   = int(scaling["factor"] * log_odds + scaling["offset"])
    return pd_pred, points


def adverse_reasons(woe_row: dict, n: int = 3) -> list[tuple[str, float]]:
    """
    Adverse action reasons for ECOA / FCRA compliance.

    Each feature's contribution to the SCORE is -(WoE × coefficient × factor): the
    minus sign matches the scorecard convention (the logit models log-odds of default,
    while the score reads "higher = safer", see notebook 03). The n features that
    subtract the MOST points — i.e. drive the applicant toward decline — are returned.
    """
    factor  = scaling["factor"]
    impacts = []
    for col in pd_model.params.index:
        if col == "const":
            continue
        score_impact = -woe_row.get(col, 0.0) * float(pd_model.params[col]) * factor
        label = col.replace("bin_", "").replace("_WoE", "").replace("_", " ")
        impacts.append((label, score_impact))
    impacts.sort(key=lambda x: x[1])   # most negative score impact = most adverse
    return impacts[:n]


# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Applicant Profile")

loan_amnt  = st.sidebar.number_input("Loan Amount ($)",        500,    40000,  15000,  500)
term       = st.sidebar.selectbox("Term",                      [" 36 months", " 60 months"])
grade      = st.sidebar.selectbox("Credit Grade",              ["A","B","C","D","E","F","G"])
int_rate   = st.sidebar.slider("Interest Rate (%)",            5.0,  30.0,  12.5, 0.25)
annual_inc = st.sidebar.number_input("Annual Income ($)",      10000, 500000, 75000, 1000)
dti        = st.sidebar.slider("Debt-to-Income Ratio (%)",     0.0,  50.0,  18.0,  0.5)
fico       = st.sidebar.slider("FICO Score",                   500,   850,   710)
home_own   = st.sidebar.selectbox("Home Ownership",            ["RENT","MORTGAGE","OWN","OTHER"])
revol_util = st.sidebar.slider("Revolving Utilisation (%)",    0.0,  100.0,  45.0,  1.0)

if st.sidebar.button("Run Credit Underwriting", type="primary", use_container_width=True):

    term_months = int(term.strip().split()[0])
    r = int_rate / 1200
    installment = (loan_amnt * r / (1 - (1 + r) ** -term_months)) if r > 0 else loan_amnt / term_months

    raw = {
        "loan_amnt"    : loan_amnt,
        "funded_amnt"  : loan_amnt,
        "term"         : term,
        "int_rate"     : int_rate,
        "installment"  : installment,
        "grade"        : grade,
        "annual_inc"   : annual_inc,
        "dti"          : dti,
        "fico_range_low": fico,
        "home_ownership": home_own,
        "revol_util"   : revol_util,
    }

    # ── 1. PD via champion scorecard ─────────────────────────────────────────
    woe_row      = apply_woe_pipeline(raw)
    computed_pd, score = score_applicant(woe_row)

    # ── 2. LGD and EAD ───────────────────────────────────────────────────────
    ead_vals = []
    for f in ead_features:
        if f == "term":
            ead_vals.append(float(term_months))
        else:
            ead_vals.append(float(raw.get(f, 0.0)))

    ead_df       = pd.DataFrame([ead_vals], columns=ead_features)
    computed_lgd = float(np.clip(lgd_model.predict(ead_df)[0], 0.0, 1.0))
    computed_ead = float(np.clip(ead_model.predict(ead_df)[0], 0.0, loan_amnt))
    ecl          = computed_pd * computed_lgd * computed_ead

    # ── 3. Decision ───────────────────────────────────────────────────────────
    if computed_pd < 0.08:
        decision = "🟢 APPROVE"
    elif computed_pd < 0.15:
        decision = "🟡 MANUAL REVIEW"
    else:
        decision = "🔴 DECLINE"

    # ── 4. IFRS 9 staging ────────────────────────────────────────────────────
    # Proper IFRS 9 SICR requires comparing current PD to origination PD.
    ifrs9 = "Stage 1 — 12-month ECL" if computed_pd < 0.10 else "Stage 2 — Lifetime ECL"

    # ── 5. Output ─────────────────────────────────────────────────────────────
    st.subheader(f"Decision: {decision}")
    st.progress(float(np.clip(1.0 - computed_pd, 0.0, 1.0)), text="Creditworthiness Index")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scorecard Points", f"{score:,}")
    c2.metric("PD",               f"{computed_pd:.2%}")
    c3.metric("LGD",              f"{computed_lgd:.2%}")
    c4.metric("EAD",              f"${computed_ead:,.0f}")
    c5.metric("ECL",              f"${ecl:,.2f}")

    st.caption(
        f"**IFRS 9:** {ifrs9} · Staging is an absolute-PD proxy; "
        "production systems compare current PD to origination PD (SICR)."
    )

    # ── 6. Adverse action notice (Manual Review + Decline only) ───────────────
    if computed_pd >= 0.08:
        st.divider()
        st.subheader("📋 Adverse Action Notice  ·  ECOA / FCRA")
        st.caption(
            "Reason codes identify the features contributing most negatively to this "
            "applicant's scorecard. Point impacts are derived from WoE × coefficient × scaling factor."
        )
        for i, (feature, pts) in enumerate(adverse_reasons(woe_row, n=3), 1):
            st.write(f"**Reason {i}:** `{feature}` — score impact: **{pts:+.1f} pts**")