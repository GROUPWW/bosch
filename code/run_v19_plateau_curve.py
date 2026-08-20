#!/usr/bin/env python3
"""v19 · 平台期学习曲线：各 family 最终训练池内做同区间子采样，证"再加样本无帮助"。
z: 51条精选 train{12,22,32,43}; h: base42 ss184 train{8,14,20,26,30};
d: compat83 train{15,30,45,60,71}; w: base42 fixed8 train{10,16,22,28,34}。各 3 seeds。"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su
from run_v7_transfer_ablation import fixed8_split
from run_stageC_partitioned_Phase1 import build_arm_split

EXCLUDE = {"B143", "B166"}

FAMS = {
    "zmin": {"file": "Bosch_zmin_select_aug_v2.xlsx", "split": "fixed8", "ss": 2026,
             "lr": 1e-6, "l2sp": 10.0, "br": 0.1, "epochs": 3000,
             "sizes": [12, 22, 32, 43]},
    "h1":   {"file": "Bosch_38_B.xlsx", "split": "ss184", "ss": 184,
             "lr": 1e-4, "l2sp": 10.0, "br": 0.01, "epochs": 2000,
             "sizes": [8, 14, 20, 26, 30]},
    "d1":   {"file": "Bosch_aug_v14_109.xlsx", "split": "arm", "ss": 177,
             "lr": 1e-4, "l2sp": 0.0, "br": 0.01, "epochs": 2000,
             "sizes": [15, 30, 45, 60, 71]},
    "w":    {"file": "Bosch_38_B.xlsx", "split": "fixed8", "ss": 2026,
             "lr": 3e-4, "l2sp": 0.0, "br": 0.01, "epochs": 1000,
             "sizes": [10, 16, 22, 28, 34]},
}
SEEDS = [2026, 42, 123]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="zmin,h1,d1,w")
    args = ap.parse_args()
    fams = [f.strip() for f in args.families.split(",")]

    out_dir = os.path.join(_HERE, "runs_stageC_v19_plateau")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    summary = os.path.join(out_dir, "summary_v19.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd
    raws = {}
    for fam in fams:
        fc = FAMS[fam]
        if fc["file"] not in raws:
            raws[fc["file"]] = sc.build_stageC_raw(device=device, new_excel=os.path.join(_HERE, fc["file"]),
                                                   height_family="h1", recipe_aug_mode="time",
                                                   stageA_heads_root=stageA)
        raw = raws[fc["file"]]
        if fc["split"] == "fixed8":
            sp0 = fixed8_split(raw, fam)
        elif fc["split"] == "ss184":
            sp0 = sc.build_single_fixed_dataset(raw=raw, key_recipes=[], families_eval=[fam],
                                                test_ratio=0.2, val_ratio=0.1,
                                                min_test_points=1, split_seed=184)
        else:
            df_109 = pd.read_excel(os.path.join(_HERE, fc["file"]))
            df_compat = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
            rid_109 = [c for c in df_109.columns if "配方" in str(c)][0]
            rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
            ids_109 = df_109[rid_109].astype(str).tolist()
            ids_compat = set(df_compat[rid_compat].astype(str).tolist())
            extra_idx = [i for i, s in enumerate(ids_109) if s in ids_compat and not (s.startswith("B") and int(s[1:]) < 80)]
            orig_n = sum(1 for s in ids_109 if s.startswith("B") and int(s[1:]) < 80)
            sp0 = build_arm_split(raw, orig_n, extra_idx, fc["ss"], 0.2, 0.1, 1, fam, "compat85")
            rid = raw["recipe_ids"]
            sp0["train_idx"] = [i for i in sp0["train_idx"] if rid[i] not in EXCLUDE]

        full_train = list(sp0["train_idx"])
        val_is_train = (fc["split"] == "fixed8")
        for k in fc["sizes"]:
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + k)
                sub = sorted(rng.choice(full_train, size=min(k, len(full_train)), replace=False).tolist())
                sp = dict(sp0)
                sp["train_idx"] = sub
                if val_is_train:
                    sp["val_idx"] = list(sub)
                name = f"v19_{fam}_n{k}_rs{seed}"
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
                                tag="v19", key_recipes=[])
                r = pd.read_csv(summary).iloc[-1]
                print(f"[v19 RESULT] {fam:5s} n={k:3d} rs{seed}: R2={r.r2:+.4f} best_ep={r.best_epoch}", flush=True)


if __name__ == "__main__":
    main()
