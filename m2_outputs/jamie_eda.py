"""
M2 — Phase 1 & 2: Liquidity EDA and Stock Classification
Jamie's deliverable

What this does:
  1. Explores BAS and order volume distributions across all 127 stocks
  2. Classifies each stock as liquid, mixed or illiquid
  3. Flags the most extreme time_ids within each stock for the models to use
  4. Outputs jamie_liquidity.csv — the shared input every other script depends on

Classification logic:
  A stock is LIQUID if:
    - Median BAS is in the bottom 50% across all stocks  (tight spread)
    - Median order volume is in the top 50%              (deep book)
  Stocks that conflict (tight spread but low volume, or vice versa) are
  classified by BAS alone as the primary signal.

Outputs:
  jamie_liquidity.csv         — stock-level regime labels (shared input)
  jamie_timeid_flags.csv      — per time_id liquidity scores for model filtering
  jamie_eda_summary.png       — overview plots
  jamie_eda_per_stock.png     — BAS vs volume scatter coloured by regime
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

def label_bucket_liquidity(frame, bas_q33, bas_q66, act_q33, act_q66):
    def _label(r):
        if r["bas"] <= bas_q33 and r["log_activity"] >= act_q66: return "liquid"
        if r["bas"] >= bas_q66 and r["log_activity"] <= act_q33: return "illiquid"
        return "mixed"
    frame["bucket_liquidity"] = frame.apply(_label, axis=1)
    return frame

df_full = df_full.rename(columns={
    "WAP_mean": "wap", "BidAskSpread_mean": "bas", "volatility": "rv"})
bas_q33_1 = train1["bas"].quantile(0.33); bas_q66_1 = train1["bas"].quantile(0.66)
act_q33_1 = train1["log_activity"].quantile(0.33); act_q66_1 = train1["log_activity"].quantile(0.66)
train1 = label_bucket_liquidity(train1, bas_q33_1, bas_q66_1, act_q33_1, act_q66_1)
test1  = label_bucket_liquidity(test1,  bas_q33_1, bas_q66_1, act_q33_1, act_q66_1)

# ═══════════════════════════════════════════════════════════════════════
# ██  Liquidity Classification  ████████████████
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "█" * 70)
print("Liquidity Classification  (all stocks)")
print("█" * 70)

train_all, _, _, _, _ = train_val_split(df_full)
print(f"  Engineering features on full train ({len(train_all):,} rows) …")
train_all_feat = add_features(train_all)

bas_q33_g = train_all_feat["bas"].quantile(0.33); bas_q66_g = train_all_feat["bas"].quantile(0.66)
act_q33_g = train_all_feat["log_activity"].quantile(0.33); act_q66_g = train_all_feat["log_activity"].quantile(0.66)
print(f"  Global BAS thresholds  : Low < {bas_q33_g:.6f}  ≤  Med < {bas_q66_g:.6f}  ≤  High")
print(f"  Global Activity thresh : Low < {act_q33_g:.4f}   ≤  Med < {act_q66_g:.4f}   ≤  High")

train_all_feat = label_bucket_liquidity(train_all_feat, bas_q33_g, bas_q66_g, act_q33_g, act_q66_g)

stock_liq = (
    train_all_feat.groupby("stock_id").agg(
        median_bas          = ("bas",              "median"),
        median_log_activity = ("log_activity",     "median"),
        median_rv           = ("rv",               "median"),
        mean_rv             = ("rv",               "mean"),
        rv_std              = ("rv",               "std"),
        liquid_pct          = ("bucket_liquidity", lambda x: (x == "liquid").mean()),
        illiquid_pct        = ("bucket_liquidity", lambda x: (x == "illiquid").mean()),
        mixed_pct           = ("bucket_liquidity", lambda x: (x == "mixed").mean()),
        n_buckets           = ("rv",               "count"),
    ).reset_index()
)

def _regime(row):
    if row["liquid_pct"]   >= 0.40: return "liquid"
    if row["illiquid_pct"] >= 0.40: return "illiquid"
    return "mixed"
def _model(regime): return "EGARCH-X" if regime == "liquid" else "WLS / HAR-RV"

stock_liq["stock_regime"]      = stock_liq.apply(_regime, axis=1)
stock_liq["recommended_model"] = stock_liq["stock_regime"].apply(_model)
rc = stock_liq["stock_regime"].value_counts()
print("\n  Stock-level regime counts:")
for r, c in rc.items():
    print(f"    {r:<10}: {c:>4} stocks ({100*c/len(stock_liq):.1f}%)")

def sample_stocks(stock_liq_df, regime, n, seed):
    pool = stock_liq_df[stock_liq_df["stock_regime"] == regime]["stock_id"].tolist()
    n    = min(n, len(pool))
    return sorted(np.random.default_rng(seed).choice(pool, size=n, replace=False).tolist())

liquid_stocks   = sample_stocks(stock_liq, "liquid",   N_DEMO_PER_REGIME, RANDOM_SEED)
mixed_stocks    = sample_stocks(stock_liq, "mixed",    N_DEMO_PER_REGIME, RANDOM_SEED + 1)
illiquid_stocks = sample_stocks(stock_liq, "illiquid", N_DEMO_PER_REGIME, RANDOM_SEED + 2)
demo_stocks     = liquid_stocks + mixed_stocks + illiquid_stocks

print(f"\n  ── Demo Stock Selection (random, seed={RANDOM_SEED}) ──")
print(f"  20 Random Liquid   stock_ids : {liquid_stocks}")
print(f"  20 Random Mixed    stock_ids : {mixed_stocks}")
print(f"  20 Random Illiquid stock_ids : {illiquid_stocks}")
print(f"  Total demo stocks: {len(demo_stocks)}")

for label, sids in [("LIQUID", liquid_stocks), ("MIXED", mixed_stocks), ("ILLIQUID", illiquid_stocks)]:
    tbl = stock_liq[stock_liq["stock_id"].isin(sids)].sort_values("stock_id")
    print(f"\n  ── {label} (n={len(sids)}) ──")
    print(tbl[["stock_id","liquid_pct","illiquid_pct","mixed_pct",
               "median_bas","median_log_activity","recommended_model"]].to_string(index=False))

stock_liq.to_csv(OUT + "m2_stock_liquidity_profile.csv", index=False)
print(f"\n  Saved: m2_stock_liquidity_profile.csv")

fig, axes = plt.subplots(1, 3, figsize=(22, 7)); fig.patch.set_facecolor(C_BG)
fig.suptitle("Phase 2 — Per-Stock Liquidity Classification  (all stocks)\n"
             "Each dot = one stock  |  Teal = liquid  |  Gold = mixed  |  Red = illiquid",
             fontsize=12, fontweight="bold", color=C_BLUE)
for regime, color in STOCK_COLORS.items():
    sub = stock_liq[stock_liq["stock_regime"] == regime]
    axes[0].scatter(sub["median_bas"], sub["median_log_activity"], color=color, alpha=0.85,
                    s=60, edgecolors="white", linewidths=0.5, label=f"{regime} (n={len(sub)})", zorder=3)
for sids, color in [(liquid_stocks, C_TEAL), (mixed_stocks, C_GOLD), (illiquid_stocks, C_RED)]:
    for sid in sids[:5]:
        row = stock_liq[stock_liq["stock_id"] == sid].iloc[0]
        axes[0].annotate(f"S{int(sid)}", xy=(row["median_bas"], row["median_log_activity"]),
                         xytext=(2,2), textcoords="offset points", fontsize=6, color=color)
axes[0].set_title("All Stocks: Median BAS vs Activity"); axes[0].legend(fontsize=9)
axes[0].set_xlabel("Median Bid-Ask Spread"); axes[0].set_ylabel("Median log(Activity)")
liq_labels_p2 = ["liquid", "mixed", "illiquid"]
axes[1].bar(liq_labels_p2, [rc.get(l,0) for l in liq_labels_p2],
            color=[STOCK_COLORS[l] for l in liq_labels_p2], alpha=0.85, edgecolor="none")
axes[1].set_title("Stock Regime Distribution"); axes[1].set_ylabel("Number of stocks")
for i, l in enumerate(liq_labels_p2):
    cnt = rc.get(l, 0)
    axes[1].text(i, cnt + len(stock_liq)*0.01, f"{100*cnt/len(stock_liq):.1f}%",
                 ha="center", fontsize=9)
ranked = stock_liq.sort_values("liquid_pct", ascending=False).reset_index(drop=True)
bar_c  = [C_TEAL if r["stock_id"] in liquid_stocks else
           C_GOLD if r["stock_id"] in mixed_stocks  else
           C_RED  if r["stock_id"] in illiquid_stocks else C_LGREY
           for _, r in ranked.iterrows()]
axes[2].bar(range(len(ranked)), ranked["liquid_pct"].values, color=bar_c, alpha=0.85, edgecolor="none", width=1.0)
axes[2].set_title("Liquid % per Stock (ranked)"); axes[2].set_xlabel("Stock rank"); axes[2].set_ylabel("liquid_pct")
axes[2].legend(handles=[mpatches.Patch(color=C_TEAL,  label="20 random liquid"),
                         mpatches.Patch(color=C_GOLD,  label="20 random mixed"),
                         mpatches.Patch(color=C_RED,   label="20 random illiquid"),
                         mpatches.Patch(color=C_LGREY, label="Other stocks")], fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "phase2_all_stock_liquidity.png", dpi=150, bbox_inches="tight"); plt.show()
print("  Saved: phase2_all_stock_liquidity.png  ✓")
