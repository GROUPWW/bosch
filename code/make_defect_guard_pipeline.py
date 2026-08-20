#!/usr/bin/env python3
"""缺陷护栏 + 反筛流程图（SI 用）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=200)
ax.set_xlim(0, 115); ax.set_ylim(0, 46); ax.axis("off")

C_REC   = "#DDEBF7"  # 蓝：数据/配方
C_MODEL = "#E2EFDA"  # 绿：模型
C_GUARD = "#FCE4D6"  # 橙：护栏
C_WARN  = "#F8CBAD"  # 深橙：风险/排除
C_TXT   = "#1a1a1a"

def box(x, y, w, h, text, fc, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec="#555", lw=1.0))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=C_TXT, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, label="", color="#333", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=color, lw=1.3, linestyle=ls))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 1.6, label, ha="center", fontsize=7.6, color=color)

# 上排：反筛主流程（先验护栏）
box(1,  28, 15, 9, "Recipe candidates\n(sampling in process window)", C_REC)
box(20, 28, 15, 9, "Morphology prediction\n(StageC surrogate)", C_MODEL)
box(39, 28, 15, 9, "Screening criterion\nd < 10 nm", C_MODEL)
box(58, 28, 16, 9, "PRIOR defect guard\nP(defect) from recipe\n(LOO AUC ≈ 0.72)", C_GUARD, bold=True)
box(80, 34, 15, 7, "Low risk → etch", C_REC)
box(80, 25, 15, 7, "High risk →\nSEM verify first", C_WARN)

# 下排：测量后 QC（后验护栏）→ 校准集
box(20, 6, 16, 9, "Etch + SEM metrology\n(new recipes)", C_REC)
box(42, 6, 22, 9, "POST-metrology defect rules\nR1 off-envelope condition\nR2 width explosion (w@9/w@1>1.3, w@9>1 μm)\nR3 bowing profile (narrow ≥4% + rebound ≥1.12)", C_GUARD, bold=True, fs=7.6)
box(70, 6, 16, 9, "Calibration set\n(104 recipes, defect-free)", C_MODEL)
box(92, 6, 15, 9, "Excluded\n(8 recipes)", C_WARN)
box(70, 17, 16, 6, "retrain / update StageC", C_MODEL, fs=7.6)

arrow(16, 32.5, 20, 32.5)
arrow(35, 32.5, 39, 32.5)
arrow(54, 32.5, 58, 32.5)
arrow(74, 34, 80, 36.5)
arrow(74, 30, 80, 28.5)
arrow(28, 28, 28, 15, "etch &\nmeasure", ls="--")
arrow(36, 10.5, 42, 10.5)
arrow(64, 10.5, 70, 10.5)
arrow(86, 10.5, 92, 10.5, "rule hit")
arrow(78, 15, 78, 17)
arrow(78, 23, 66, 30, "", ls="--", color="#777")
ax.text(71.5, 26.5, "updated model\n(next screening round)", fontsize=7.6, color="#777", ha="center")

ax.text(1, 43, "Defect guard in the screening–calibration loop", fontsize=11, fontweight="bold")
ax.text(1, 1.0, "Rules R1–R3: 5/5 defective recipes caught, 0/34 false positives on the current metrology pool.",
        fontsize=7.8, color="#555")

fig.tight_layout()
out = os.path.join(_HERE := os.path.dirname(os.path.abspath(__file__)), "..", "docs", "defect_guard_pipeline.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", os.path.abspath(out))
