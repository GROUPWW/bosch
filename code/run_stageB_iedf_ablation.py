#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a-iv 消融：learned IEDF 潜变量 (pca7 / ae7) vs 物理描述符 (phys7_true)。
训练集 = case_with_phys7_156.xlsx（156 个有完整 IEDF 的 case）。

3 臂 × 3 train_seeds(42/123/7) × 6 family，paper-aligned 超参，
split_seed=3 固定（与 v9/verify 口径一致）。

接线方式（无需改 harness）：
  - phys7 臂：phys_source="phys7_true"，build_morph_dataset_phys7 直接从 df 读 PHYS7_NAMES 列；
  - pca7/ae7 臂：prepare_shared_cache 之后，把 (N,7) 潜变量按 case_id 对齐注入
    shared_phys7_seq_cache["pca7"] / ["ae7"]，run_one_experiment 以 phys_source="pca7"/"ae7"
    命中缓存（stageB_train_morph_on_phys7_pycharm.py:663-673），
    build_morph_dataset_phys7 内部 broadcast_phys7_to_T 自动沿 T 复制成 (N,7,T)
    （stageB_util.py:1287 + stageB_util.py:457）。

用法：
  python run_stageB_iedf_ablation.py                 # 全部训练 + 聚合
  python run_stageB_iedf_ablation.py --aggregate_only # 只从 results_summary.csv 重新聚合
"""
import os, sys, argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stageB_train_morph_on_phys7_pycharm import (
    Cfg, prepare_shared_cache, run_one_experiment, set_seed, _ensure_dir,
    append_summary_row,
)
from stageB_util import FAMILIES, _norm_case_id

RUNS_ROOT = os.path.join(_HERE, "runs_stageB_iedf_ablation")

# 两臂共同训练集（156 行）
CASE_XLSX_156 = os.path.join(_HERE, "case_with_phys7_156.xlsx")

# 潜变量文件：key = phys_source 名；value = (xlsx, 7 个特征列)
LATENT_SOURCES = {
    "pca7": (os.path.join(_HERE, "case_with_pca7_156.xlsx"), [f"pca_{i+1}" for i in range(7)]),
    "ae7":  (os.path.join(_HERE, "case_with_ae7_156.xlsx"),  [f"ae_{i+1}" for i in range(7)]),
}

# 3 臂：phys7 = 真值物理描述符（从 df 列读），pca7/ae7 = learned IEDF 潜变量（缓存注入）
ARMS = {
    "phys7": "phys7_true",
    "pca7": "pca7",
    "ae7": "ae7",
}

TRAIN_SEEDS = [42, 123, 7]
SPLIT_SEED = 3

HP_OVERRIDE = {
    "lr": 1e-3, "weight_decay": 1e-4, "tf_dropout": 0.1,
    "tf_d_model": 256, "tf_layers": 2,
    "loss_type": "huber", "huber_beta": 0.1,
    "epochs": 200, "early_patience": 30, "test_eval_every": 5,
}
HP_TAG_BASE = "lr0p001_wd0p0001_do0p1_dm256_L2_hb0p1"


def build_latent_cache(case_ids_cache) -> dict:
    """按清洗后的 case_ids 顺序，从潜变量 xlsx 取 (N,7) 数组。"""
    out = {}
    for key, (fname, cols) in LATENT_SOURCES.items():
        dfl = pd.read_excel(fname)
        dfl["_cid"] = dfl["input"].astype(str).map(_norm_case_id)
        lut = dfl.set_index("_cid")[cols]
        missing = [c for c in case_ids_cache if c not in lut.index]
        if missing:
            raise KeyError(f"{os.path.basename(fname)} 缺 case: {missing[:5]} (共{len(missing)})")
        arr = lut.loc[list(case_ids_cache)].to_numpy(dtype=np.float32)
        assert arr.shape == (len(case_ids_cache), 7), f"{key} shape={arr.shape}"
        assert np.isfinite(arr).all(), f"{key} 含 NaN/inf"
        out[key] = arr
    return out


def aggregate(runs_root: str = RUNS_ROOT):
    """per-family × 3 臂 test R2 mean±std -> summary_table.csv + 打印。"""
    csv_path = os.path.join(runs_root, "results_summary.csv")
    if not os.path.exists(csv_path):
        print(f"[AGG] 未找到 {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df["arm"] = df["phys_source"].map({v: k for k, v in ARMS.items()}).fillna(df["phys_source"])
    df["test_r2"] = pd.to_numeric(df["overall_r2"], errors="coerce")

    long_rows = []
    for (arm, fam), g in df.groupby(["arm", "family_mode"]):
        long_rows.append({
            "arm": arm, "family": fam, "n_seeds": int(g["test_r2"].notna().sum()),
            "test_r2_mean": float(g["test_r2"].mean()),
            "test_r2_std": float(g["test_r2"].std(ddof=1)) if len(g) > 1 else 0.0,
        })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(os.path.join(runs_root, "summary_long.csv"), index=False)

    arm_order = [a for a in ARMS if a in set(long_df["arm"])]
    fam_order = [f for f in FAMILIES if f in set(long_df["family"])]
    table = pd.DataFrame(index=fam_order, columns=arm_order, dtype=str)
    for _, r in long_df.iterrows():
        table.loc[r["family"], r["arm"]] = f"{r['test_r2_mean']:.4f}±{r['test_r2_std']:.4f}"
    table.index.name = "family"
    table.to_csv(os.path.join(runs_root, "summary_table.csv"))

    print("\n===== a-iv 消融汇总：test R2 mean±std (3 train seeds, split_seed=3) =====")
    print(table.to_string())
    print(f"\n[OK] 写出: {os.path.join(runs_root, 'summary_table.csv')}")
    print(f"[OK] 写出: {os.path.join(runs_root, 'summary_long.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--runs_root", default=RUNS_ROOT)
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate(args.runs_root)
        return

    runs_root = args.runs_root
    _ensure_dir(runs_root)
    arms = {k: v for k, v in ARMS.items() if k in [a.strip() for a in args.arms.split(",")]}

    # ---- 数据口径：156 子集；不需要 StageA provider（phys7 臂用真值列）----
    Cfg.excel_path = CASE_XLSX_156
    Cfg.phys_sources = ["none"]
    Cfg.seed = TRAIN_SEEDS[0]
    set_seed(Cfg.seed)

    (df_cache, recipe_cols_cache, recipe_raw_cache,
     stageA_provider_cache, targets_full, mask_full,
     phys7_seq_cache, case_ids_cache,
     bad_row, before_N, kept_idx, clipped_stats) = prepare_shared_cache()

    print(f"[a-iv] cleaned N = {len(df_cache)} (before={before_N})")

    # ---- 注入 pca7 / ae7 潜变量缓存（按清洗后 case 顺序对齐）----
    latent_cache = build_latent_cache(case_ids_cache)
    for key, arr in latent_cache.items():
        phys7_seq_cache[key] = arr
        print(f"[a-iv] inject phys7_seq_cache['{key}'] shape={arr.shape}")

    fam_list = list(FAMILIES)
    total = len(arms) * len(TRAIN_SEEDS) * len(fam_list)
    fieldnames = None
    job = 0
    failures = []

    for arm, ps in arms.items():
        for seed in TRAIN_SEEDS:
            for fam in fam_list:
                job += 1
                hp = dict(HP_OVERRIDE)
                hp["train_seed"] = seed
                hp_tag = f"{HP_TAG_BASE}_ts{seed}"
                print(f"\n[a-iv {job}/{total}] arm={arm} phys_source={ps} train_seed={seed} family={fam}", flush=True)
                try:
                    r = run_one_experiment(
                        model_type="transformer",
                        phys_source=ps,
                        recipe_aug_mode="time",
                        phys7_mode="full",
                        root_out=runs_root,
                        split_seed=SPLIT_SEED,
                        job_idx=job,
                        job_total=total,
                        target_family=fam,
                        shared_df=df_cache,
                        shared_recipe_cols=recipe_cols_cache,
                        shared_recipe_raw=recipe_raw_cache,
                        shared_targets_full=targets_full,
                        shared_mask_full=mask_full,
                        shared_phys7_seq_cache=phys7_seq_cache,
                        shared_stageA_provider=stageA_provider_cache,
                        hp_override=hp,
                        hp_tag=hp_tag,
                    )
                    if fieldnames is None:
                        fieldnames = list(r.keys())
                    append_summary_row(r, runs_root, fieldnames=fieldnames)
                    print(f"[a-iv RESULT] {arm:6s} ts{seed:<4d} {fam:5s} test R2 = {r['overall_r2']:.4f}", flush=True)
                except Exception as e:
                    failures.append((arm, seed, fam, repr(e)))
                    print(f"[a-iv FAIL] arm={arm} seed={seed} fam={fam}: {e}", flush=True)

    if failures:
        print(f"\n[a-iv] {len(failures)} 个 run 失败：")
        for f_ in failures:
            print("  ", f_)

    aggregate(runs_root)


if __name__ == "__main__":
    main()
