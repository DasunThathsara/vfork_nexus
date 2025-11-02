import sys, os, json, math, joblib, argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
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

    # Schema B: text,generated with 1=AI, 0=human (Kaggle)
    elif "generated" in cols:
        gcol = cols["generated"]
        y_ai = pd.to_numeric(df[gcol], errors="coerce").fillna(0).clip(0,1).astype(int).values
        y = (1 - y_ai)  # convert to 1=human, 0=ai
    else:
        raise ValueError("CSV must contain either 'label' (human/ai) or 'generated' (1=AI, 0=human).")

    if min_words > 0:
        w = df[tcol].str.split().str.len().fillna(0).astype(int)
        keep = (w >= min_words)
        df = df[keep].copy()
        y = y[keep.values]

    texts = df[tcol].tolist()
    X = np.vstack([extract_features(t) for t in texts])
    return y, X

def downsample_balance(X, y, per_class=None, seed=42):
    if not per_class: return X, y
    rs = np.random.RandomState(seed)
    idx_h = np.where(y == 1)[0]
    idx_a = np.where(y == 0)[0]
    take_h = rs.choice(idx_h, size=min(per_class, len(idx_h)), replace=False)
    take_a = rs.choice(idx_a, size=min(per_class, len(idx_a)), replace=False)
    idx = np.concatenate([take_h, take_a])
    rs.shuffle(idx)
    return X[idx], y[idx]

def can_stratify(n0, n1, test_size):
    if min(n0, n1) < 2: return False
    return (math.floor(n0*test_size) >= 1) and (math.floor(n1*test_size) >= 1)

def safe_metrics(y_true, probs):
    def _try(name, fn, *a, **k):
        try: return fn(*a, **k)
        except Exception as e:
            print(f"[WARN] {name} unavailable: {e}"); return float("nan")
    roc = _try("ROC-AUC", roc_auc_score, y_true, probs)
    acc = _try("Accuracy", accuracy_score, y_true, (probs >= 0.5).astype(int))
    brier = _try("Brier", brier_score_loss, y_true, probs)
    return roc, acc, brier

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to CSV (Kaggle or custom).")
    ap.add_argument("--sample", type=int, default=60000, help="Total rows to sample for speed (0=all).")
    ap.add_argument("--per_class", type=int, default=25000, help="Downsample cap per class (0=no cap).")
    ap.add_argument("--min_words", type=int, default=30, help="Drop very short texts (<min_words).")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    print(f"[INFO] Loading {args.csv}")
    y_all, X_all = load_csv(args.csv, min_words=args.min_words)
    n0, n1 = int((y_all == 0).sum()), int((y_all == 1).sum())
    print(f"[INFO] After filtering: total={len(y_all)}, ai={n0}, human={n1}")

    # Optional global subsample
    if args.sample and args.sample > 0 and args.sample < len(y_all):
        rs = np.random.RandomState(42)
        take = rs.choice(np.arange(len(y_all)), size=args.sample, replace=False)
        X_all, y_all = X_all[take], y_all[take]
        n0, n1 = int((y_all == 0).sum()), int((y_all == 1).sum())
        print(f"[INFO] Subsampled to {len(y_all)} rows (ai={n0}, human={n1})")

    # Optional per-class balancing cap
    if args.per_class and args.per_class > 0:
        X_all, y_all = downsample_balance(X_all, y_all, per_class=args.per_class)
        n0, n1 = int((y_all == 0).sum()), int((y_all == 1).sum())
        print(f"[INFO] Downsampled (per_class cap) -> ai={n0}, human={n1}")

    # Tiny-data guard
    if len(y_all) < 12 or min(n0, n1) < 5:
        print("[WARN] Tiny-data mode: training on ALL samples, no split/calibration.")
        scaler = joblib.load("scaler.joblib") if os.path.exists("scaler.joblib") else None
        if scaler is None: scaler = StandardScaler().fit(X_all)
        else: print("[INFO] Reusing existing scaler.joblib")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_all)
        base = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(X_all), y_all)
        joblib.dump(scaler, "scaler.joblib"); joblib.dump(base, "model.joblib")
        with open("feature_list.json","w") as f: json.dump(FEATURE_NAMES, f)
        sys.exit(0)

    # Normal path
    from sklearn.preprocessing import StandardScaler
    strat = y_all if can_stratify(n0, n1, args.test_size) else None
    if strat is None: print("[INFO] Split without stratification.")
    X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=args.test_size,
                                              random_state=42, stratify=strat)
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_va = scaler.transform(X_tr), scaler.transform(X_va)

    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    min_class_tr = np.bincount(y_tr).min()
    cv_folds = max(2, min(5, int(min_class_tr)))
    try:
        print(f"[INFO] Calibrating (isotonic, cv={cv_folds})")
        clf = CalibratedClassifierCV(base, method="isotonic", cv=cv_folds).fit(Xs_tr, y_tr)
    except Exception as e:
        print(f"[WARN] Isotonic failed ({e}). Trying sigmoid...")
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv_folds).fit(Xs_tr, y_tr)

    # Select human (class=1) probability explicitly
    probs = clf.predict_proba(Xs_va)[:, list(clf.classes_).index(1)]
    roc, acc, brier = safe_metrics(y_va, probs)
    print(f"ROC-AUC: {roc:.4f}\nAccuracy: {acc:.4f}\nBrier: {brier:.4f}")

    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(clf, "model.joblib")
    with open("feature_list.json","w") as f: json.dump(FEATURE_NAMES, f)
    print("Saved scaler.joblib, model.joblib, feature_list.json")
