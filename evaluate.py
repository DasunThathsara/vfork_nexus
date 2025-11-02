# evaluate.py
import argparse, os, json, math, joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from features import extract_features, FEATURE_NAMES

def load_csv(csv_path: str, min_words: int = 0):
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "text" not in cols:
        raise ValueError("CSV must contain a 'text' column.")
    tcol = cols["text"]
    df[tcol] = df[tcol].astype(str)

    # Schema A: text,label in {human, ai}
    if "label" in cols:
        labcol = cols["label"]
        df[labcol] = df[labcol].astype(str).str.strip().str.lower()
        mask = df[labcol].isin(["human", "ai"])
        if not mask.all():
            bad = df.loc[~mask, labcol].unique().tolist()
            raise ValueError(f"Invalid labels {bad}. Expected 'human' or 'ai'.")
        y = (df[labcol] == "human").astype(int).values  # 1=human, 0=ai

    # Schema B (Kaggle): text,generated with 1=AI, 0=human
    elif "generated" in cols:
        gcol = cols["generated"]
        y_ai = pd.to_numeric(df[gcol], errors="coerce").fillna(0).clip(0,1).astype(int).values
        y = (1 - y_ai)  # convert to 1=human, 0=ai
    else:
        raise ValueError("CSV must contain either 'label' or 'generated' columns.")

    if min_words > 0:
        w = df[tcol].str.split().str.len().fillna(0).astype(int)
        keep = (w >= min_words)
        df = df[keep].copy()
        y = y[keep.values]

    texts = df[tcol].tolist()
    X = np.vstack([extract_features(t) for t in texts])
    return X, y

def can_stratify(n0, n1, test_size):
    if min(n0, n1) < 2: return False
    return (math.floor(n0*test_size) >= 1) and (math.floor(n1*test_size) >= 1)

def proba_human(model, Xs):
    # Always return probability for class=1 (human)
    if hasattr(model, "predict_proba"):
        P = model.predict_proba(Xs)
        if hasattr(model, "classes_"):
            import numpy as np
            col = int(np.where(np.array(model.classes_) == 1)[0][0])
            return P[:, col]
        return P[:, -1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(Xs).reshape(-1)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(Xs).astype(float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to the same CSV used for training (Kaggle or custom).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction (default 0.2 = 20%).")
    ap.add_argument("--min_words", type=int, default=30, help="Ignore very short texts (<min_words).")
    ap.add_argument("--save_preds", default="", help="Optional path to save predictions CSV.")
    args = ap.parse_args()

    # Load artifacts
    if not os.path.exists("scaler.joblib") or not os.path.exists("model.joblib"):
        raise FileNotFoundError("Missing scaler.joblib or model.joblib. Train first.")
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("model.joblib")

    # Load data
    X, y = load_csv(args.csv, min_words=args.min_words)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    print(f"[INFO] Loaded rows: {len(y)}  (ai={n0}, human={n1})")

    # Recreate the same 80/20 split logic used in training
    strat = y if can_stratify(n0, n1, args.test_size) else None
    if strat is None:
        print("[INFO] Evaluating with non-stratified split (still deterministic).")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=strat
    )

    Xs_te = scaler.transform(X_te)
    probs = proba_human(model, Xs_te)
    preds = (probs >= 0.5).astype(int)

    # Metrics
    try:
        roc = roc_auc_score(y_te, probs)
    except Exception as e:
        print(f"[WARN] ROC-AUC unavailable: {e}"); roc = float("nan")
    acc = accuracy_score(y_te, preds)
    try:
        brier = brier_score_loss(y_te, probs)
    except Exception as e:
        print(f"[WARN] Brier unavailable: {e}"); brier = float("nan")

    print("\n=== Test Results (20%) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"Brier    : {brier:.4f}")

    cm = confusion_matrix(y_te, preds, labels=[0,1])
    print("\nConfusion Matrix [rows=true, cols=pred] (0=AI, 1=human):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_te, preds, target_names=["AI (0)","Human (1)"]))

    # Optional: save predictions
    if args.save_preds:
        out = pd.DataFrame({
            "y_true": y_te,
            "p_human": probs,
            "y_pred": preds
        })
        out.to_csv(args.save_preds, index=False)
        print(f"[INFO] Saved predictions to {args.save_preds}")

if __name__ == "__main__":
    main()
