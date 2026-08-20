#!/usr/bin/env python3
"""缺陷护栏算法架构图（SI 用）：先验分类器算法流 + 后验规则流。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=200)
ax.set_xlim(0, 115); ax.set_ylim(0, 56); ax.axis("off")

C_IN  = "#DDEBF7"; C_FEAT = "#E2EFDA"; C_MODEL = "#FFF2CC"
C_OUT = "#FCE4D6"; C_RULE = "#F8CBAD"; C_TXT = "#1a1a1a"

def box(x, y, w, h, text, fc, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=fc, ec="#555", lw=1.0))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=C_TXT, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, label="", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color="#333", lw=1.3, linestyle=ls))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 1.5, label, ha="center", fontsize=7.6, color="#333")

# ===== 上排：先验分类器（算法流）=====
ax.text(2, 51.5, "(a) Prior defect classifier (pre-experiment)", fontsize=10, fontweight="bold")
box(2,  40, 17, 8, "Input features\nrecipe parameters (7-dim)\n+ StageA IEDF descriptors (7-dim)", C_IN)
box(25, 40, 15, 8, "Feature vector x ∈ R¹⁴\n(z-score normalized)", C_FEAT)
box(46, 40, 19, 8, "Logistic regression\nz = wᵀx + b,  P = σ(z)\n(baseline: GradientBoosting)", C_MODEL, bold=True)
box(71, 40, 16, 8, "P(defect) ∈ [0,1]\nthreshold τ", C_OUT)
box(93, 40, 19, 8, "high-risk flag →\nSEM verification\nbefore etching", C_OUT)
arrow(19, 44, 25, 44); arrow(40, 44, 46, 44); arrow(65, 44, 71, 44); arrow(87, 44, 93, 44)
ax.text(46, 36.5, "Training: 39 measured recipes (5 defective / 34 normal), leave-one-out CV;  LOO AUC = 0.735 (reported as-is)",
        fontsize=7.8, color="#555")

# ===== 下排：后验规则（测量后）=====
ax.text(2, 27.5, "(b) Post-metrology rule guard", fontsize=10, fontweight="bold")
box(2,  12, 20, 12, "Measured morphology\nz / h / d / w\n@ cycles 3, 5, 9", C_IN)
box(28, 15.5, 24, 5.5, "R1  operating point outside\ncalibrated envelope", C_RULE, fs=7.4)
box(28, 9.5,  24, 5.5, "R2  width explosion:\nw@9/w@1 > 1.3 and w@9 > 1000 nm", C_RULE, fs=7.4)
box(28, 3.5,  24, 5.5, "R3  bowing profile: mid narrowing ≥4%\n+ end rebound ≥1.12", C_RULE, fs=7.4)
box(58, 12, 16, 9, "any rule hit?\n(OR)", C_MODEL, bold=True)
box(80, 16, 16, 6, "pass → calibration set", C_FEAT)
box(80, 5,  16, 6, "hit → excluded (8 recipes)", C_OUT)
arrow(22, 18, 28, 18); arrow(22, 14, 28, 12.3); arrow(22, 12.5, 28, 6.3)
arrow(52, 18, 58, 16.5); arrow(52, 12.3, 58, 16.5); arrow(52, 6.3, 58, 14)
arrow(74, 18.5, 80, 19); arrow(74, 14.5, 80, 8)
ax.text(58, 1.0, "Current pool: 5/5 defective caught, 0/34 false positives (descriptive rules; to be re-validated on future batches)",
        fontsize=7.8, color="#555")

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "defect_guard_architecture.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", os.path.abspath(out))
