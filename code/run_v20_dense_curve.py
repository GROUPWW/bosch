#!/usr/bin/env python3
"""v20 · 加密学习曲线：从 42 条起按配方批次顺序（旧27→新40，剔除5条unusable）逐步加样本。
h/w 每步+5（42→82），z/d 每步+10（42→104）。统一 fixed8 test(ss2026) + val=train。
预算减半（z1500/h1000/d1000/w500ep）——曲线只看形状，图注声明预算差异。"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su

FIXED_TEST_RECIPES = ["B17", "B12", "B36", "B70", "B26", "B27", "B30", "B45"]
UNUSABLE = {"B143", "B166", "B177", "B192", "B215"}

FAMS = {
    "zmin": {"lr": 1e-6, "l2sp": 10.0, "br": 0.1,  "epochs": 1500,
             "tiers": [42, 52, 62, 72, 82, 92, 102, 104]},
    "h1":   {"lr": 1e-4, "l2sp": 10.0, "br": 0.01, "epochs": 1000,
             "tiers": [42, 47, 52, 57, 62, 67, 72, 77, 82]},
    "d1":   {"lr": 1e-4, "l2sp": 0.0,  "br": 0.01, "epochs": 1000,
             "tiers": [42, 52, 62, 72, 82, 92, 102, 104]},
    "w":    {"lr": 3e-4, "l2sp": 0.0,  "br": 0.01, "epochs": 500,
             "tiers": [42, 47, 52, 57, 62, 67, 72, 77, 82]},
}
SEEDS = [2026, 42, 123]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="zmin,h1,d1,w")
    args = ap.parse_args()

    out_dir = os.path.join(_HERE, "runs_stageC_v20_dense_curve")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v20] device={device}")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    raw = sc.build_stageC_raw(device=device,
                              new_excel=os.path.join(_HERE, "Bosch_aug_v14_109.xlsx"),
                              height_family="h1", recipe_aug_mode="time",
                              stageA_heads_root=stageA)
    recipe_ids = raw["recipe_ids"]
    test_idx = [i for i, rid in enumerate(recipe_ids) if rid in FIXED_TEST_RECIPES]
    avail = [i for i in range(len(recipe_ids)) if i not in test_idx and recipe_ids[i] not in UNUSABLE]
    base = [i for i in avail if int(recipe_ids[i][1:]) < 80]
    adds = [i for i in avail if int(recipe_ids[i][1:]) >= 80]
    adds.sort(key=lambda i: int(recipe_ids[i][1:]))  # 批次顺序：旧 27 → 新 40
    print(f"[v20] base={len(base)} adds={len(adds)} test={len(test_idx)}")

    summary = os.path.join(out_dir, "summary_v20.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd
    for fam, fc in FAMS.items():
        if fam not in args.families.split(","):
            continue
        for total in fc["tiers"]:
            n_add = max(0, total - 42)
            train = base + adds[:n_add]
            sp = {"train_idx": train, "val_idx": list(train), "test_idx": test_idx,
                  "ignored_idx": [], "seed": 2026, "drop_tag": "v20"}
            for seed in SEEDS:
                name = f"v20_{fam}_n{total}_rs{seed}"
                exp = sc.ExpCfg(
                    name=name, init="stageB_best", finetune_mode="head_ln",
                    phys7_mode=stageB_phys_mode, lr=fc["lr"], wd=1e-4, l2sp=fc["l2sp"],
                    backbone_lr_ratio=fc["br"], loss_type="huber", huber_beta=1.0,
                    grad_clip=1.0, progressive_unfreeze=True)
                run_args = argparse.Namespace(
                    out_dir=out_dir, stageA_heads_root=stageA,
                    recipe_aug_mode="time", model_type="transformer", batch=4,
                    patience=99999, num_workers=0, seed=seed, seed_repeats=1,
                    test_ratio=0.2, val_ratio=0.1, min_test_points=1)
                dk = sc._load_done_keys(summary)
                sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                                experiments=[exp], stageB_best=stageB_best, args=run_args,
                                device=device, summary_all=summary, header_fields=hf,
                                done_keys=dk, epochs=fc["epochs"], patience=99999,
                                tag="v20", key_recipes=[])
                r = pd.read_csv(summary).iloc[-1]
                print(f"[v20 RESULT] {fam:5s} n={total:3d} rs{seed}: R2={r.r2:+.4f}", flush=True)


if __name__ == "__main__":
    main()
