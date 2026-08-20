# -*- coding: utf-8 -*-
"""
评估 stageA 7-head 模型在 Bosch 实测 recipe 上的 Phys7 描述符预测。

由于 Bosch 实测没有真值 IEDF/Phys7，本脚本做分布诊断：
1. 在仿真集上复算 stageA 各 head 的 R²（验证与论文一致）。
2. 用 stageA heads 预测 Bosch 各变体的 7 维 Phys7。
3. 比较 Bosch 预测分布 vs 仿真训练集分布，标记 z-score > 3 的 OOD 样本。
"""
from __future__ import annotations
import os, sys, argparse, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import torch
import physio_util as pu
import stageB_train_morph_on_phys7_pycharm as sb
import stageB_util as su


PHYS7_COLS = [
    "logGamma_SF6_tot",
    "pF_SF6",
    "spread_SF6",
    "qskew_SF6",
    "logGamma_C4F8_tot",
    "rho_C4F8",
    "spread_C4F8",
]


def load_case_phys7(case_excel: str):
    df = pd.read_excel(case_excel, sheet_name="Sheet1")
    recipe_cols = [c for c in df.columns if c not in PHYS7_COLS and c not in ["input", "output"]]
    # 实际 recipe 列为前 7 个数值列：APC, source_RF, LF_RF, SF6, C4F8, DEP time, etch time
    recipe_cols = ["APC（E2步骤）", "source_RF（E2步骤）", "LF_RF（E2步骤）",
                   "SF6（E2步骤）", "C4F8（DEP步骤）", "DEP time", "etch time"]
    X = df[recipe_cols].to_numpy(dtype=np.float32)
    Y = df[PHYS7_COLS].to_numpy(dtype=np.float32)
    return X, Y, recipe_cols


def eval_stageA_on_simulation(heads_root: str, X: np.ndarray, Y: np.ndarray, device: str):
    provider = sb.StageAEnsemblePhys7Provider(
        heads_root=heads_root,
        device=device,
        recipe_cols_in=None,
        expect_k=7,
    )
    pred = provider.infer(X, phys7_mode="full", use_cache=True).astype(np.float32)
    rows = []
    for i, col in enumerate(PHYS7_COLS):
        yt = Y[:, i]
        yp = pred[:, i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt_m = yt[mask]
        yp_m = yp[mask]
        r2 = float(1 - np.sum((yt_m - yp_m)**2) / np.sum((yt_m - yt_m.mean())**2))
        mae = float(np.mean(np.abs(yt_m - yp_m)))
        rows.append({"descriptor": col, "r2": r2, "mae": mae, "n": int(mask.sum())})
    return rows, pred


def predict_stageA_on_bosch(heads_root: str, bosch_xlsx: str, device: str):
    df = pd.read_excel(bosch_xlsx)
    recipe_cols = ["APC（E2步骤）", "source_RF（E2步骤）", "LF_RF（E2步骤）",
                   "SF6（E2步骤）", "C4F8（DEP步骤）", "DEP time", "etch time"]
    X = df[recipe_cols].to_numpy(dtype=np.float32)
    recipe_names = df["配方名"].astype(str).tolist()

    provider = sb.StageAEnsemblePhys7Provider(
        heads_root=heads_root,
        device=device,
        recipe_cols_in=None,
        expect_k=7,
    )
    pred = provider.infer(X, phys7_mode="full", use_cache=True).astype(np.float32)
    return recipe_names, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_excel", default=os.path.join(_HERE, "case_with_phys7.xlsx"))
    ap.add_argument("--heads_root", default=os.path.join(_HERE, "runs_stageA_phys7", "best_by_test"))
    ap.add_argument("--bosch_variants", default="A,B,C,D")
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "runs_stageA_eval_bosch"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # StageA 推理在 MPS 上数值不稳定，强制用 CPU
    device = "cpu"

    # 1) 仿真集上复算 stageA R²
    X_sim, Y_sim, _ = load_case_phys7(args.case_excel)
    sim_rows, _ = eval_stageA_on_simulation(args.heads_root, X_sim, Y_sim, device)
    df_sim = pd.DataFrame(sim_rows)
    df_sim.to_csv(os.path.join(args.out_dir, "stageA_simulation_r2.csv"), index=False)
    print("\n[StageA on simulation dataset]")
    print(df_sim.to_string(index=False))

    # 2) Bosch 各变体预测
    variants = [v.strip() for v in args.bosch_variants.split(",") if v.strip()]
    all_bosch_stats = []
    sim_mean = Y_sim.mean(axis=0)
    sim_std = Y_sim.std(axis=0)

    for variant in variants:
        bosch_xlsx = os.path.join(_HERE, f"Bosch_38_{variant}.xlsx")
        if not os.path.exists(bosch_xlsx):
            continue
        recipe_names, pred = predict_stageA_on_bosch(args.heads_root, bosch_xlsx, device)
        pred_df = pd.DataFrame(pred, columns=PHYS7_COLS)
        pred_df["recipe_name"] = recipe_names
        pred_df.to_csv(os.path.join(args.out_dir, f"stageA_pred_Bosch_38_{variant}.csv"), index=False)

        # distribution stats
        mean = pred.mean(axis=0)
        std = pred.std(axis=0)
        all_bosch_stats.append({
            "variant": variant,
            **{f"mean_{c}": mean[i] for i, c in enumerate(PHYS7_COLS)},
            **{f"std_{c}": std[i] for i, c in enumerate(PHYS7_COLS)},
        })

        # OOD samples (z-score wrt simulation)
        z = np.abs((pred - sim_mean) / np.maximum(sim_std, 1e-12))
        max_z = z.max(axis=1)
        ood_idx = np.where(max_z > 3.0)[0]
        if len(ood_idx) > 0:
            print(f"\n[Variant {variant}] OOD recipes (max z-score > 3 wrt simulation):")
            for idx in ood_idx:
                print(f"  {recipe_names[idx]}: max_z={max_z[idx]:.2f}")
        else:
            print(f"\n[Variant {variant}] No OOD recipes (max z-score <= 3)")

    df_stats = pd.DataFrame(all_bosch_stats)
    df_stats.to_csv(os.path.join(args.out_dir, "stageA_bosch_distribution_stats.csv"), index=False)
    print("\n[Bosch predicted descriptor distribution stats]")
    print(df_stats.to_string(index=False))

    # 3) Save simulation distribution for reference
    ref = pd.DataFrame({
        "descriptor": PHYS7_COLS,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
    })
    ref.to_csv(os.path.join(args.out_dir, "stageA_simulation_distribution_ref.csv"), index=False)
    print(f"\nSaved all results to {args.out_dir}")


if __name__ == "__main__":
    main()
