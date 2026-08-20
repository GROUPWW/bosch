#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9 · R2-3b 模型主干消融：GRU / MLP（+后续 LSTM/TCN/AR-MLP）

paper-aligned 同预算（lr1e-3, wd1e-4, do0.1, dm256, L2, huber0.1, 200ep, split_seed=3, train_seed=42）。
Transformer 对照已有：best_config_common_all_families.json 的 per_family_r2
（zmin 0.9805 / h0 0.9672 / h1 0.9381 / d0 0.9665 / d1 0.9403 / w 0.9710）。

用法：python run_v9_backbone_ablation.py --model_types gru,mlp
"""
import os, sys, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stageB_train_morph_on_phys7_pycharm import (
    Cfg, prepare_shared_cache, run_one_experiment, set_seed, _ensure_dir,
    append_summary_row,
)
from stageB_util import FAMILIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_types", default="gru,mlp")
    ap.add_argument("--phys_source", default="stageA_pred")
    ap.add_argument("--runs_root", default=os.path.join(_HERE, "runs_stageB_backbone_ablation_v9"))
    args = ap.parse_args()
    model_types = [m.strip() for m in args.model_types.split(",")]

    runs_root = args.runs_root
    _ensure_dir(runs_root)

    hp_override = {
        "lr": 1e-3, "weight_decay": 1e-4, "tf_dropout": 0.1,
        "tf_d_model": 256, "tf_layers": 2,
        "loss_type": "huber", "huber_beta": 0.1,
        "epochs": 200, "early_patience": 30, "test_eval_every": 5,
    }
    hp_tag = "lr0p001_wd0p0001_do0p1_dm256_L2_hb0p1"
    split_seed, train_seed = 3, 42

    Cfg.stageA_heads_root = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")
    Cfg.seed = train_seed
    set_seed(Cfg.seed)

    (df_cache, recipe_cols_cache, recipe_raw_cache,
     stageA_provider_cache, targets_full, mask_full,
     phys7_seq_cache, case_ids_cache,
     bad_row, before_N, kept_idx, clipped_stats) = prepare_shared_cache()

    fam_list = list(FAMILIES)
    total = len(fam_list) * len(model_types)
    fieldnames = None
    job = 0
    for mt in model_types:
        print(f"\n========== model_type={mt} ==========", flush=True)
        for fam in fam_list:
            job += 1
            print(f"\n[v9 {job}/{total}] model={mt} family={fam}", flush=True)
            r = run_one_experiment(
                model_type=mt,
                phys_source=args.phys_source,
                recipe_aug_mode="time",
                phys7_mode="full",
                root_out=runs_root,
                split_seed=split_seed,
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
                hp_override=hp_override,
                hp_tag=hp_tag,
            )
            if fieldnames is None:
                fieldnames = list(r.keys())
            append_summary_row(r, runs_root, fieldnames=fieldnames)
            pf = r.get("per_family_r2", {})
            print(f"[v9 RESULT] {mt:5s} {fam:5s} test R2 = {pf.get(fam, float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
