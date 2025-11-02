import streamlit as st, numpy as np, joblib, json
from features import extract_features, FEATURE_NAMES

st.set_page_config(page_title="Human-or-AI Probability", layout="centered")
st.title("Human-or-AI Probability (MVP)")

# --- Load artifacts
scaler = joblib.load("scaler.joblib")
model = joblib.load("model.joblib")
with open("feature_list.json") as f:
    feat_names = json.load(f)

# --- Helpers
def predict_proba_safe(mdl, Xs):
    """Return P(human) even if the model lacks predict_proba (e.g., raw linear)."""
    if hasattr(mdl, "predict_proba"):
        p = mdl.predict_proba(Xs)
        if p.ndim == 2:
            return p[:, 1]
    # Fallback: use decision_function or linear score → squashed to [0,1]
    if hasattr(mdl, "decision_function"):
        s = mdl.decision_function(Xs)
        s = np.asarray(s).reshape(-1)
        # Logistic squashing (approx)
        return 1.0 / (1.0 + np.exp(-s))
    # Last resort: predict → map {0,1} to {0.0,1.0}
    preds = mdl.predict(Xs)
    return preds.astype(float)

def get_linear_coefs(mdl):
    """
    Return (coef, intercept) when available.
    Works for:
      - CalibratedClassifierCV with linear base_estimator
      - Plain LogisticRegression / LinearSVC (coef_) etc.
    Otherwise return (None, None).
    """
    # Calibrated wrapper case
    if hasattr(mdl, "base_estimator_") and hasattr(mdl.base_estimator_, "coef_"):
        return mdl.base_estimator_.coef_[0], getattr(mdl.base_estimator_, "intercept_", np.array([0.0]))[0]
    # Plain linear model (e.g., LogisticRegression)
    if hasattr(mdl, "coef_"):
        return mdl.coef_[0], getattr(mdl, "intercept_", np.array([0.0]))[0]
    return None, None

txt = st.text_area("Paste 50–250 words:", height=200, placeholder="Try something like: 'hello world' ...")
if st.button("Analyze") and txt.strip():
    x = extract_features(txt)
    xs = scaler.transform([x])
    p_human = float(predict_proba_safe(model, xs)[0])

    st.metric("P(human)", f"{p_human*100:.1f}%")
    st.caption("Probabilistic estimate. Use with human review. Calibrated if enough data was available.")

    # Simple feature contributions for linear models (interpretive; not SHAP)
    coef, intercept = get_linear_coefs(model)
    if coef is not None:
        contrib = np.abs(coef * xs[0])  # magnitude-based
        top_idx = np.argsort(contrib)[-3:][::-1]
        st.subheader("Top contributing features")
        for i in top_idx:
            st.write(f"• **{feat_names[i]}** (scaled {xs[0][i]:+.2f}, weight {coef[i]:+.2f})")
    else:
        st.info("Feature contribution view is unavailable for this model type. "
                "Add more data and retrain to enable a calibrated linear base for explanations.")
