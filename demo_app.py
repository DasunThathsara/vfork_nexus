import streamlit as st, numpy as np, joblib, json
from features import extract_features, FEATURE_NAMES

st.set_page_config(page_title="Human-or-AI Probability", layout="centered")
st.title("Human-or-AI Probability (MVP)")

scaler = joblib.load("scaler.joblib")
model = joblib.load("model.joblib")
with open("feature_list.json") as f:
    feat_names = json.load(f)

def proba_human(mdl, Xs):
    if hasattr(mdl, "predict_proba"):
        P = mdl.predict_proba(Xs)
        if hasattr(mdl, "classes_"):
            classes = np.array(mdl.classes_)
            if 1 in classes:
                col = int(np.where(classes == 1)[0][0])
                return P[:, col]
        return P[:, -1]
    if hasattr(mdl, "decision_function"):
        s = mdl.decision_function(Xs).reshape(-1)
        return 1.0 / (1.0 + np.exp(-s))
    return mdl.predict(Xs).astype(float)

def get_linear_coefs(mdl):
    if hasattr(mdl, "base_estimator_") and hasattr(mdl.base_estimator_, "coef_"):
        return mdl.base_estimator_.coef_[0], getattr(mdl.base_estimator_, "intercept_", np.array([0.0]))[0]
    if hasattr(mdl, "coef_"):
        return mdl.coef_[0], getattr(mdl, "intercept_", np.array([0.0]))[0]
    return None, None

txt = st.text_area("Paste ~50–250 words:", height=200,
                   placeholder="Longer text gives a more reliable probability.")
if len(txt.split()) < 30:
    st.warning("Very short text (<30 words). Probability may be unreliable.")

if st.button("Analyze") and txt.strip():
    x = extract_features(txt)
    xs = scaler.transform([x])
    p_human = float(proba_human(model, xs)[0])
    st.metric("P(human)", f"{p_human*100:.1f}%")
    st.caption("Calibrated if the dataset allowed. Use with human review.")

    coef, _ = get_linear_coefs(model)
    if coef is not None:
        contrib = np.abs(coef * xs[0])
        top_idx = np.argsort(contrib)[-3:][::-1]
        st.subheader("Top contributing features")
        for i in top_idx:
            st.write(f"• **{FEATURE_NAMES[i]}** (scaled {xs[0][i]:+.2f}, weight {coef[i]:+.2f})")
    else:
        st.info("Feature contributions not available for this model type.")
