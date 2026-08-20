#!/usr/bin/env python3
"""
v8 · R1-C3 学习曲线 v2（最终口径）

设计（docs/06_reviewer_response_guide.md §1.3）：
- P-fixed：test 固定为原始 42 条中的 8 条（B17/B12/B36/B70/B26/B27/B30/B45），val=train。
- 训练样本档（5 档）：42(Bosch_38_B) / 48(aug_yellow) / 69(aug_all27) / 85(109 中 compat 子集) / 109(全量)
- 每档 × 3 train_seeds {2026,42,123} × 4 family = 60 runs
- 每 family 钻石超参：zmin head_ln lr1e-6 l2sp10 3000ep；h1 head_ln lr1e-4 l2sp10 2000ep；
  d1 head_ln lr1e-4 l2sp0 2000ep；w head_ln lr3e-4 l2sp0 1000ep
"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su

FIXED_TEST_RECIPES = ["B17", "B12", "B36", "B70", "B26", "B27", "B30", "B45"]

TIERS = [
    ("t42",  "Bosch_38_B.xlsx",        None,    set()),
    ("t48",  "Bosch_aug_yellow.xlsx",  None,    set()),
    ("t69",  "Bosch_aug_all27.xlsx",   None,    set()),
    ("t85",  "Bosch_aug_v14_109.xlsx", "compat", {"B143","B166"}),
    ("t109", "Bosch_aug_v14_109.xlsx", None,    {"B143","B166","B177","B192","B215"}),
]

FAM_CFG = {
    "zmin": {"lr": 1e-6, "l2sp": 10.0, "br": 0.1,  "epochs": 3000},
    "h1":   {"lr": 1e-4, "l2sp": 10.0, "br": 0.01, "epochs": 2000},
    "d1":   {"lr": 1e-4, "l2sp": 0.0,  "br": 0.01, "epochs": 2000},
    "w":    {"lr": 3e-4, "l2sp": 0.0,  "br": 0.01, "epochs": 1000},
}
SEEDS = [2026, 42, 123]


def build_split(raw, fam, compat_only, exclude=frozenset()):
    recipe_ids = raw["recipe_ids"]
    test_idx = [i for i, rid in enumerate(recipe_ids) if rid in FIXED_TEST_RECIPES]
    remaining = [i for i in range(len(recipe_ids)) if i not in test_idx]
    remaining = [i for i in remaining if recipe_ids[i] not in exclude]
    if compat_only:
        import pandas as pd
        df_compat = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
        rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
        compat = set(df_compat[rid_compat].astype(str).tolist())
        remaining = [i for i in remaining
                     if (recipe_ids[i].startswith("B") and int(recipe_ids[i][1:]) < 80) or recipe_ids[i] in compat]
    return {"train_idx": remaining, "val_idx": list(remaining), "test_idx": test_idx,
            "ignored_idx": [], "seed": 2026, "drop_tag": "v8lc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="zmin,h1,d1,w")
    ap.add_argument("--tiers", default="t42,t48,t69,t85,t109")
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "runs_stageC_v8_learning_curve"))
    args = ap.parse_args()
    fams = [f.strip() for f in args.families.split(",")]
    tiers = [t for t in TIERS if t[0] in args.tiers.split(",")]

    out_dir = args.out_dir
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v8] device={device} 学习曲线 v2: {len(tiers)}档×{len(SEEDS)}seed×{len(fams)}fam")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    summary = os.path.join(out_dir, "summary_v8.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd
    raws = {}
    for tier_name, file, compat_only, exclude in tiers:
        if file not in raws:
            raws[file] = sc.build_stageC_raw(device=device, new_excel=os.path.join(_HERE, file),
                                             height_family="h1", recipe_aug_mode="time",
                                             stageA_heads_root=stageA)
        raw = raws[file]
        for fam in fams:
            fc = FAM_CFG[fam]
            sp = build_split(raw, fam, compat_only, exclude)
            for seed in SEEDS:
                name = f"v8_{tier_name}_{fam}_rs{seed}"
                exp = sc.ExpCfg(
                    name=name, init="stageB_best", finetune_mode="head_ln",
                    phys7_mode=stageB_phys_mode,
                    lr=fc["lr"], wd=1e-4, l2sp=fc["l2sp"], backbone_lr_ratio=fc["br"],
                    loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=True)
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
                                tag="v8lc", key_recipes=[])
                r = pd.read_csv(summary).iloc[-1]
                print(f"[v8 RESULT] {tier_name:5s} {fam:5s} rs{seed}: R2={r.r2:+.4f} trainN={r.trainN} best_ep={r.best_epoch}", flush=True)


if __name__ == "__main__":
    main()
