#!/usr/bin/env python3
"""归档 h1 最佳权重：重跑 v7d 获胜配置（l2sp=0, seed42, ss184, 2000ep）并落盘。"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su


def main():
    fam = "h1"
    out_dir = os.path.join(_HERE, "runs_stageC_archive_h1")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[archive-h1] device={device}")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    raw = sc.build_stageC_raw(device=device,
                              new_excel=os.path.join(_HERE, "Bosch_38_B.xlsx"),
                              height_family="h1", recipe_aug_mode="time",
                              stageA_heads_root=stageA)
    sp = sc.build_single_fixed_dataset(raw=raw, key_recipes=[], families_eval=[fam],
                                       test_ratio=0.2, val_ratio=0.1,
                                       min_test_points=1, split_seed=184)

    exp = sc.ExpCfg(
        name="archive_h1_head_ln_lr1e-04_l2sp0_ep2000_ss184_rs42",
        init="stageB_best", finetune_mode="head_ln", phys7_mode=stageB_phys_mode,
        lr=1e-4, wd=1e-4, l2sp=0.0, backbone_lr_ratio=0.01,
        loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=True)
    run_args = argparse.Namespace(
        out_dir=out_dir, stageA_heads_root=stageA,
        recipe_aug_mode="time", model_type="transformer", batch=4,
        patience=99999, num_workers=0, seed=42, seed_repeats=1,
        test_ratio=0.2, val_ratio=0.1, min_test_points=1)

    summary = os.path.join(out_dir, "summary_h1.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")
    dk = sc._load_done_keys(summary)
    sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                    experiments=[exp], stageB_best=stageB_best, args=run_args,
                    device=device, summary_all=summary, header_fields=hf,
                    done_keys=dk, epochs=2000, patience=99999,
                    tag="arch_h1", key_recipes=[])
    import pandas as pd
    r = pd.read_csv(summary).iloc[-1]
    print(f"[archive-h1 RESULT] R2={r.r2:.4f} best_ep={r.best_epoch}")


if __name__ == "__main__":
    main()
