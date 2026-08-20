#!/usr/bin/env python3
"""v6: w 达标配置（head_ln lr3e-4 1000ep 新口径）稳定性验证 ×3 seeds + 权重落盘（STAGEC_SAVE_CKPT=1）。"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su

FIXED_TEST_RECIPES = ["B17", "B12", "B36", "B70", "B26", "B27", "B30", "B45"]
SEEDS = [2026, 42, 123]


def main():
    fam = "w"
    out_dir = os.path.join(_HERE, "runs_stageC_w359_archive")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    raw = sc.build_stageC_raw(device=device,
                              new_excel=os.path.join(_HERE, "Bosch_38_B.xlsx"),
                              height_family="h1", recipe_aug_mode="time",
                              stageA_heads_root=stageA)
    recipe_ids = raw["recipe_ids"]
    test_idx = [i for i, rid in enumerate(recipe_ids) if rid in FIXED_TEST_RECIPES]
    remaining = [i for i in range(len(recipe_ids)) if i not in test_idx]
    sp = {"train_idx": remaining, "val_idx": list(remaining), "test_idx": test_idx,
          "ignored_idx": [], "seed": 2026, "drop_tag": "w359arch"}

    summary = os.path.join(out_dir, "summary_w359arch.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd
    for seed in SEEDS:
        run_args = argparse.Namespace(
            out_dir=out_dir, stageA_heads_root=stageA,
            recipe_aug_mode="time", model_type="transformer", batch=4,
            patience=99999, num_workers=0, seed=seed, seed_repeats=1,
            test_ratio=0.2, val_ratio=0.1, min_test_points=1)
        exp = sc.ExpCfg(
            name=f"w359arch_head_ln_lr3e-04_ep1000_rs{seed}",
            init="stageB_best", finetune_mode="head_ln", phys7_mode=stageB_phys_mode,
            lr=3e-4, wd=1e-4, l2sp=0.0, backbone_lr_ratio=0.01,
            loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=True)
        dk = sc._load_done_keys(summary)
        sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                        experiments=[exp], stageB_best=stageB_best, args=run_args,
                        device=device, summary_all=summary, header_fields=hf,
                        done_keys=dk, epochs=1000, patience=99999,
                        tag="w359arch", key_recipes=[])
        r = pd.read_csv(summary).iloc[-1]
        print(f"[w359arch RESULT] seed={seed}: R2={r.r2:.4f} best_ep={r.best_epoch}")


if __name__ == "__main__":
    main()
