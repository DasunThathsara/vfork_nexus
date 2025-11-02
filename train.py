import pandas as pd, numpy as np, json, joblib, sys, math
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from features import extract_features, FEATURE_NAMES

def safe_metrics(y_true, probs):
    try:
        roc = roc_auc_score(y_true, probs)
    except Exception as e:
        print(f"[WARN] ROC-AUC unavailable: {e}"); roc = float("nan")
    try:
        acc = accuracy_score(y_true, (probs>=0.5).astype(int))
    except Exception as e:
        print(f"[WARN] Accuracy unavailable: {e}"); acc = float("nan")
    try:
        brier = brier_score_loss(y_true, probs)
    except Exception as e:
        print(f"[WARN] Brier unavailable: {e}"); brier = float("nan")
    return roc, acc, brier

# --- Load ---
df = pd.read_csv("data/text_samples.csv")  # columns: text,label
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["human","ai"])].copy()
if df.empty:
    print("[ERROR] No valid rows. Ensure labels are 'human' or 'ai'.")
    sys.exit(1)

X = np.vstack([extract_features(t) for t in df["text"].astype(str)])
y = (df["label"] == "human").astype(int).values

cnt = Counter(y)
n0, n1 = cnt.get(0,0), cnt.get(1,0)
n = len(y)
print(f"[INFO] Samples: total={n}, ai={n0}, human={n1}")

# --- Tiny-data mode decision ---
min_class = min(n0, n1)
tiny_mode = (n < 6) or (min_class < 3)

scaler = StandardScaler().fit(X)

if tiny_mode:
    print("[WARN] Tiny-data mode: training on ALL samples, no calibration, no reliable metrics.")
    base = LogisticRegression(max_iter=200, class_weight="balanced")
    base.fit(scaler.transform(X), y)

    # Save artifacts
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(base, "model.joblib")  # this is the final model (no CalibratedCV)
    with open("feature_list.json","w") as f: json.dump(FEATURE_NAMES, f)
    print("Saved scaler.joblib, model.joblib, feature_list.json")
    print("Tip: Add ≥10–20 samples per class to enable split, calibration, and metrics.")
    sys.exit(0)

# --- Normal path (enough data to split) ---
test_size = 0.2
strat = y if (math.floor(n0*test_size) >= 1 and math.floor(n1*test_size) >= 1) else None
if strat is None:
    print("[INFO] Proceeding without stratification (still enough data to split).")

Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)

Xs_tr, Xs_v = scaler.transform(Xtr), scaler.transform(Xv)
base = LogisticRegression(max_iter=200, class_weight="balanced")

# Choose CV folds safely for calibration
cv_folds = min(3, np.bincount(ytr).min())
if cv_folds >= 2:
    try:
        clf = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds).fit(Xs_tr, ytr)
    except Exception as e:
        print(f"[WARN] Isotonic failed ({e}), trying sigmoid with cv={cv_folds}.")
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv_folds).fit(Xs_tr, ytr)
else:
    print("[WARN] Not enough per-class samples in train split for calibration. Using raw logistic probabilities.")
    clf = base.fit(Xs_tr, ytr)

probs = clf.predict_proba(Xs_v)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(Xs_v)
if probs.ndim == 1:
    probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)  # fallback normalization

roc, acc, brier = safe_metrics(yv, probs)
print(f"ROC-AUC: {roc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Brier: {brier:.4f}")

joblib.dump(scaler, "scaler.joblib")
joblib.dump(clf, "model.joblib")
with open("feature_list.json","w") as f: json.dump(FEATURE_NAMES, f)
print("Saved scaler.joblib, model.joblib, feature_list.json")
