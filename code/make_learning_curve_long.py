#!/usr/bin/env python3
"""生成 docs/learning_curve_long.png（v23 长学习曲线：四 family 统一步长 5、42→104 满量程）。
与 learning_curve_dense.png 并列：dense 版 w 截断于平台处（67），本版全程展示——
h 为真平台（~0.60 平到 104），w 在 ~82 后掉到 ~0.60 低位走平（深槽区间分布偏移的证据）。
横轴 = 校准数据集总数（含 8 条 fixed8 测试；实际训练样本数 = 值 − 8）。
数据：runs_stageC_v20_dense_curve/summary_v20.csv（v20 + v23 补档）。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_HERE, "runs_stageC_v20_dense_curve", "summary_v20.csv")
OUT = os.path.join(_HERE, "..", "docs", "learning_curve_long.png")

PANELS = [  # (family, 显示名, 原稿值, 颜色)
    ("zmin", "z", 0.94, "#0072B2"),
    ("h1",   "h", 0.92, "#E69F00"),
    ("d1",   "d", 0.73, "#009E73"),
    ("w",    "w", 0.82, "#CC79A7"),
]

df = pd.read_csv(CSV)
df = df[df.exp.str.startswith("v20_")].copy()
df["fam"] = df.exp.str.extract(r"v20_([a-z0-9]+)_n")[0]
df["n"] = df.exp.str.extract(r"_n(\d+)")[0].astype(int)

fig, axes = plt.subplots(1, 4, figsize=(20, 5.3), sharey=True)
for ax, (fam, label, ref, color) in zip(axes, PANELS):
    g = df[df.fam == fam].groupby("n").r2.agg(["mean", "std"]).sort_index()
    x = g.index.to_numpy()
    y = g["mean"].to_numpy()
    s = g["std"].fillna(0).to_numpy()
    ax.errorbar(x, y, yerr=s, fmt="o-", color=color, capsize=3, lw=1.6, ms=4.5,
                ecolor=color, elinewidth=1.2)
    ax.axhline(ref, color="gray", ls="--", lw=1.2)
    ax.set_title(f"{label}  (batch-order expansion)", fontsize=13)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("test R²", fontsize=12)
fig.supxlabel("Calibration set size (recipes, incl. 8 held-out test)", fontsize=13)
fig.suptitle("Learning curves (full range): test R² vs calibration set size "
             "(mean±std, 3 seeds, y-axis 0–1)", fontsize=14)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(OUT, dpi=160)
print(f"saved {OUT}")
for fam, label, ref, color in PANELS:
    g = df[df.fam == fam].groupby("n").r2.agg(["mean", "std"]).sort_index()
    pts = " → ".join(f"{n}: {r['mean']:.2f}±{r['std']:.2f}" for n, r in g.iterrows())
    print(f"{label}: {pts}")
