"""
Cross-domain probe: FinAgent-style regime-conditioning on LOB mid-price prediction.

Idea: FinAgent's Reflection module conditions trading decisions on a textual
"market regime" inferred by an LLM. We test the *numerical* core of that idea
on HFT LOB data: does conditioning a predictor on a cheap volatility regime
feature add skill over a regime-blind baseline? If not, the cost of an LLM
regime estimator in FinAgent is hard to justify for HFT-scale data.

Data: valid.parquet from the Wunder Fund LOB challenge
      (1.44M rows, 12 price levels, 16 volume levels, targets t0/t1)
Predictor: Ridge regression on standardised LOB features.
Comparison: (A) regime-blind baseline; (B) regime-conditioned mixture.
Metrics: Pearson rho, MAE, per-regime rho.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

OUT = Path(r"D:/DeFi/AIRI_2026_report")
OUT.mkdir(parents=True, exist_ok=True)

DATA = Path(r"D:/DeFi/Wunder Fund/Claude/datasets/valid.parquet")
N_SAMPLE = 200_000   # subset for tractable experiment
RANDOM_STATE = 42

print(f"[1/6] Loading {DATA.name} ...")
t0 = time.time()
df = pd.read_parquet(DATA)
print(f"      shape={df.shape} loaded in {time.time()-t0:.1f}s")

price_cols  = [f"p{i}" for i in range(12)]
volume_cols = [f"v{i}" for i in range(12)]
dp_cols     = [c for c in df.columns if c.startswith("dp")]
dv_cols     = [c for c in df.columns if c.startswith("dv")]
feat_cols   = price_cols + volume_cols + dp_cols + dv_cols
target_col  = "t0"

df = df[df["need_prediction"] == 1].reset_index(drop=True)
df = df.dropna(subset=[target_col]).reset_index(drop=True)
print(f"[2/6] After need_prediction==1 filter: {len(df)} rows")

# Chronological 70/30 split inside the validation set (no leakage across seq).
df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)
if len(df) > N_SAMPLE:
    df = df.iloc[:N_SAMPLE].copy()

split = int(0.7 * len(df))
train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
print(f"      train={len(train)}, test={len(test)}")

# --- Regime feature: rolling volatility of best-bid/ask spread proxy.
# We use rolling std of the first price level over a 100-tick window as a
# cheap surrogate for "market state". This mirrors what FinAgent does
# textually ("volatile" / "trending" / "ranging") but in O(N) numpy.
def add_regime(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["mid_proxy"] = (out["p0"] + out["p1"]) / 2.0
    out["regime_vol"] = (
        out.groupby("seq_ix")["mid_proxy"]
           .transform(lambda x: x.rolling(100, min_periods=10).std())
           .fillna(0.0)
    )
    return out

print("[3/6] Computing regime feature ...")
train = add_regime(train)
test  = add_regime(test)

# Median split into "calm" vs "volatile" (regime label, computed on TRAIN ONLY
# to avoid leakage of test-set statistics into training-time regime labels).
thr = train["regime_vol"].median()
train["regime"] = (train["regime_vol"] > thr).astype(int)
test["regime"]  = (test["regime_vol"]  > thr).astype(int)
print(f"      regime threshold (train median) = {thr:.6g}")
print(f"      train regime balance: {train['regime'].mean():.3f} volatile")
print(f"      test  regime balance: {test['regime'].mean():.3f} volatile")

X_train = train[feat_cols].values
X_test  = test [feat_cols].values
y_train = train[target_col].values
y_test  = test [target_col].values

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- (A) Regime-blind baseline.
print("[4/6] Training regime-blind Ridge ...")
base = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_train_s, y_train)
yhat_base = base.predict(X_test_s)

# --- (B) Regime-conditioned mixture: two specialists, one per regime.
print("[5/6] Training regime-conditioned mixture (2 specialists) ...")
masks_tr = [train["regime"] == k for k in (0, 1)]
masks_te = [test ["regime"] == k for k in (0, 1)]
specialists = []
yhat_mix = np.zeros_like(y_test, dtype=float)
for k in (0, 1):
    m_tr, m_te = masks_tr[k].values, masks_te[k].values
    if m_tr.sum() < 100 or m_te.sum() < 10:
        yhat_mix[m_te] = base.predict(X_test_s[m_te])
        continue
    mdl = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_train_s[m_tr], y_train[m_tr])
    specialists.append(mdl)
    yhat_mix[m_te] = mdl.predict(X_test_s[m_te])

# --- Metrics
def report(name: str, y: np.ndarray, yhat: np.ndarray) -> dict:
    rho, _ = pearsonr(y, yhat)
    mae = float(np.mean(np.abs(y - yhat)))
    return {"model": name, "pearson_rho": float(rho), "mae": mae, "n": int(len(y))}

results = []
results.append(report("baseline_ridge", y_test, yhat_base))
results.append(report("regime_mixture_ridge", y_test, yhat_mix))
# Per-regime breakdown for the baseline (shows whether different regimes are
# intrinsically harder, which is what would *motivate* conditioning).
for k, name in [(0, "baseline_on_calm"), (1, "baseline_on_volatile")]:
    m = masks_te[k].values
    if m.sum() > 10:
        results.append(report(name, y_test[m], yhat_base[m]))
# Per-regime breakdown for the mixture.
for k, name in [(0, "mixture_on_calm"), (1, "mixture_on_volatile")]:
    m = masks_te[k].values
    if m.sum() > 10:
        results.append(report(name, y_test[m], yhat_mix[m]))

# --- Latency probe (per-decision wall time).
print("[6/6] Latency probe ...")
x_one = X_test_s[:1]
t0 = time.perf_counter()
for _ in range(1000):
    base.predict(x_one)
ridge_us = (time.perf_counter() - t0) / 1000 * 1e6
latency = {
    "ridge_us_per_decision": ridge_us,
    "finagent_typical_s_per_decision_reported": 2.0,  # order-of-magnitude
    "lob_hft_budget_us": 100.0,
}

summary = {"results": results, "latency": latency, "threshold": float(thr)}
out_path = OUT / "metrics.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n===== RESULTS =====")
for r in results:
    print(f"  {r['model']:28s}  rho={r['pearson_rho']:+.4f}  MAE={r['mae']:.4f}  n={r['n']}")
print("\n===== LATENCY =====")
print(f"  Ridge:    {ridge_us:.2f} µs/decision")
print(f"  FinAgent: ~{latency['finagent_typical_s_per_decision_reported']*1e6:.0f} µs/decision (LLM call)")
print(f"  HFT budget: {latency['lob_hft_budget_us']:.0f} µs/decision")
print(f"\nSaved: {out_path}")
