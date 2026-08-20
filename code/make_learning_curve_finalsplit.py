#!/usr/bin/env python3
"""生成 docs/learning_curve_finalsplit.png（v24 混合口径长学习曲线，SI 用）。
各面板在**各 family 最终测试集**上评估：z/w = fixed8（复用 v20/v23 数据），
h = ss184、d = ss177（v24 新数据）；训练侧均为 109 池批次增样。
虚线为最终报告值（0.926/0.903/0.792/0.826）——d 在标签 83（其最终数据集 compat83）处
锚定最终值；h 显示扩充批次在其最终测试集上直接造成退化。
横轴 = 校准数据集总数（含固定留出；实际训练样本数 = 值 − 8（z/w）或 −12（h/d））。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV20 = os.path.join(_HERE, "runs_stageC_v20_dense_curve", "summary_v20.csv")
CSV24 = os.path.join(_HERE, "runs_stageC_v24_finalsplit_curve", "summary_v24.csv")
OUT = os.path.join(_HERE, "..", "docs", "learning_curve_finalsplit.png")

PANELS = [  # (family, 显示名, 最终值, 颜色, 数据源, 测试集说明)
    ("zmin", "z", 0.926, "#0072B2", "v20", "fixed8 test"),
    ("h1",   "h", 0.903, "#E69F00", "v24", "ss184 test (final)"),
    ("d1",   "d", 0.792, "#009E73", "v24", "ss177 test (final)"),
    ("w",    "w", 0.826, "#CC79A7", "v20", "fixed8 test"),
]

dfs = {}
for tag, path in (("v20", CSV20), ("v24", CSV24)):
    d = pd.read_csv(path)
    d = d[d.exp.str.startswith(("v20_", "v24_"))].copy()
    d["fam"] = d.exp.str.extract(r"v2[04]_([a-z0-9]+)_n")[0]
    d["n"] = d.exp.str.extract(r"_n(\d+)")[0].astype(int)
    dfs[tag] = d

fig, axes = plt.subplots(1, 4, figsize=(20, 5.3), sharey=True)
for ax, (fam, label, final, color, src, testname) in zip(axes, PANELS):
    g = dfs[src][dfs[src].fam == fam].groupby("n").r2.agg(["mean", "std"]).sort_index()
    x = g.index.to_numpy()
    y = g["mean"].to_numpy()
    s = g["std"].fillna(0).to_numpy()
    ax.errorbar(x, y, yerr=s, fmt="o-", color=color, capsize=3, lw=1.6, ms=4.5,
                ecolor=color, elinewidth=1.2)
    ax.axhline(final, color="gray", ls="--", lw=1.2)
    ax.annotate(f"final {final:.3f}", xy=(x[-1], final), xytext=(0, 6),
                textcoords="offset points", ha="right", fontsize=9, color="gray")
    ax.set_title(f"{label}  ({testname})", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("test R²", fontsize=12)
fig.supxlabel("Calibration set size (recipes, incl. held-out test/val)", fontsize=13)
fig.suptitle("Learning curves on each target's final test split (mean±std, 3 seeds, y-axis 0–1)",
             fontsize=14)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(OUT, dpi=160)
print(f"saved {OUT}")
