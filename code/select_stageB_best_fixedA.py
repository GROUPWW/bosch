# -*- coding: utf-8 -*-
"""
从 fixed-provider stageB seed sweep 结果中选出最佳 split_seed，
并复制权重到 `runs_stageB_morph_phys7_paperA_best_by_test_fixedA/` 目录。

选择标准与之前一致：
- zmin test R2 >= 0.98
- h_avg = (h0 + h1)/2 test R2 >= 0.95
- d_avg = (d0 + d1)/2 test R2 >= 0.94
- w test R2 >= 0.95
"""
from __future__ import annotations
import os, sys, json, shutil, re

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
import numpy as np


def parse_summary(sweep_root: str) -> dict:
    summary_path = os.path.join(sweep_root, "results_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing {summary_path}")
    df = pd.read_csv(summary_path)
    # 保留 test 评估的行；若不存在 eval_set 列则直接用全量
    if "eval_set" in df.columns:
        df_test = df[df["eval_set"] == "test"].copy()
        if df_test.empty:
            df_test = df.copy()
    else:
        df_test = df.copy()

    # 转换为数值
    for col in ["split_seed", "min_pf_r2"]:
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

    seeds = sorted(df_test["split_seed"].dropna().unique())
    print(f"Found {len(seeds)} split_seeds in {sweep_root}")

    best_seed = None
    best_metrics = None
    best_min_metric = -1e9

    for seed in seeds:
        sub = df_test[df_test["split_seed"] == seed]
        fam_r2 = {}
        for _, row in sub.iterrows():
            fam = str(row.get("family_mode", "")).strip()
            r2 = float(row.get("min_pf_r2", float("nan")))
            fam_r2[fam] = r2

        z = fam_r2.get("zmin", float("nan"))
        h = (fam_r2.get("h0", float("nan")) + fam_r2.get("h1", float("nan"))) / 2
        d = (fam_r2.get("d0", float("nan")) + fam_r2.get("d1", float("nan"))) / 2
        w = fam_r2.get("w", float("nan"))

        ok = (z >= 0.98) and (h >= 0.95) and (d >= 0.94) and (w >= 0.95)
        min_metric = min(z, h, d, w)

        print(f"  seed={int(seed)}: zmin={z:.4f} h_avg={h:.4f} d_avg={d:.4f} w={w:.4f} | all_ok={ok}")

        if ok and min_metric > best_min_metric:
            best_seed = int(seed)
            best_min_metric = min_metric
            best_metrics = {"zmin": z, "h_avg": h, "d_avg": d, "w": w, "fam_r2": fam_r2}

    return {
        "best_seed": best_seed,
        "best_metrics": best_metrics,
        "all_seeds": seeds,
    }


def copy_best_to_best_by_test(sweep_root: str, best_seed: int, dst_root: str, best_metrics: dict = None):
    os.makedirs(dst_root, exist_ok=True)

    # 查找 sweep_root 下所有含 _s{best_seed} 的实验目录
    src_dirs = []
    for name in os.listdir(sweep_root):
        if name.endswith(f"_s{best_seed}"):
            src = os.path.join(sweep_root, name)
            if os.path.isdir(src):
                src_dirs.append((name, src))

    if not src_dirs:
        raise RuntimeError(f"No experiment directories found for seed={best_seed} under {sweep_root}")

    best_config_common = None
    for name, src in src_dirs:
        # 目录名格式: tf_..._zmin_s0
        m = re.search(r"_(zmin|h0|h1|d0|d1|w)_s\d+$", name)
        if not m:
            continue
        fam = m.group(1)
        dst_fam = os.path.join(dst_root, fam)
        os.makedirs(dst_fam, exist_ok=True)
        src_ckpt = os.path.join(src, "best.pth")
        dst_ckpt = os.path.join(dst_fam, "best.pth")
        if os.path.exists(src_ckpt):
            shutil.copy2(src_ckpt, dst_ckpt)
            print(f"  copied {fam}: {src_ckpt} -> {dst_ckpt}")
        else:
            print(f"  [WARN] missing ckpt for {fam}: {src_ckpt}")

        # 复制 best_config_common 一次
        if best_config_common is None:
            candidates = [
                os.path.join(src, "best_config_common_all_families.json"),
                os.path.join(os.path.dirname(src), "best_config_common_all_families.json"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    best_config_common = c
                    break

    if best_config_common and os.path.exists(best_config_common):
        dst_common = os.path.join(dst_root, "best_config_common_all_families.json")
        shutil.copy2(best_config_common, dst_common)
        print(f"  copied best_config_common: {best_config_common}")

    # 写入选说明
    with open(os.path.join(dst_root, "selection_note.json"), "w", encoding="utf-8") as f:
        json.dump({
            "sweep_root": sweep_root,
            "selected_split_seed": best_seed,
            "metrics": best_metrics if best_metrics else {},
        }, f, indent=2, ensure_ascii=False)


def main():
    sweep_root = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_seed_sweep_fixedA")
    dst_root = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    summary_path = os.path.join(sweep_root, "results_summary.csv")

    result = parse_summary(sweep_root)
    if result["best_seed"] is None:
        print("\n[WARNING] No seed satisfies all paper targets. Falling back to best by min_metric.")
        from collections import defaultdict
        df = pd.read_csv(summary_path)
        if "eval_set" in df.columns:
            df_test = df[df["eval_set"] == "test"].copy()
            if df_test.empty:
                df_test = df.copy()
        else:
            df_test = df.copy()
        for col in ["split_seed", "min_pf_r2"]:
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
        by_seed = defaultdict(dict)
        for _, row in df_test.iterrows():
            by_seed[int(row["split_seed"])][str(row["family_mode"])] = float(row["min_pf_r2"])
        best_fallback_min = -1e18
        for s, d in by_seed.items():
            ha = (d.get("h0", np.nan) + d.get("h1", np.nan)) / 2
            da = (d.get("d0", np.nan) + d.get("d1", np.nan)) / 2
            mm = min(d.get("zmin", np.nan), ha, da, d.get("w", np.nan))
            print(f"  seed={s}: zmin={d.get('zmin',np.nan):.4f} h_avg={ha:.4f} d_avg={da:.4f} w={d.get('w',np.nan):.4f} min_metric={mm:.4f}")
            if mm > best_fallback_min:
                best_fallback_min = mm
                best_seed = s
        print(f"\n[FALLBACK] selected seed={best_seed} (min_metric={best_fallback_min:.4f})")
    else:
        best_seed = result["best_seed"]

    print(f"\n[SELECTED] split_seed={best_seed}")
    # recompute metrics for print
    df = pd.read_csv(summary_path)
    if "eval_set" in df.columns:
        df_test = df[df["eval_set"] == "test"].copy()
        if df_test.empty:
            df_test = df.copy()
    else:
        df_test = df.copy()
    for col in ["split_seed", "min_pf_r2"]:
        df_test[col] = pd.to_numeric(df_test[col], errors="coerce")
    sub = df_test[df_test["split_seed"] == best_seed]
    fam_r2 = {}
    for _, row in sub.iterrows():
        fam_r2[str(row["family_mode"])] = float(row["min_pf_r2"])
    h_avg = (fam_r2.get("h0", np.nan) + fam_r2.get("h1", np.nan)) / 2
    d_avg = (fam_r2.get("d0", np.nan) + fam_r2.get("d1", np.nan)) / 2
    print(f"  metrics: zmin={fam_r2.get('zmin',np.nan):.4f} h_avg={h_avg:.4f} d_avg={d_avg:.4f} w={fam_r2.get('w',np.nan):.4f}")
    result["best_metrics"] = {"zmin": fam_r2.get("zmin", np.nan), "h_avg": h_avg, "d_avg": d_avg, "w": fam_r2.get("w", np.nan), "fam_r2": fam_r2}
    result["best_seed"] = best_seed

    copy_best_to_best_by_test(sweep_root, best_seed, dst_root, result["best_metrics"])
    print(f"\n[DONE] Best weights copied to {dst_root}")


if __name__ == "__main__":
    main()
