#!/usr/bin/env python3
"""v13 · d1 compat83 复核：compat85 剔除 B143(0°C)/B166(缺陷) 后，钻石配置重跑（对照 0.7802）。"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su
from run_stageC_partitioned_Phase1 import build_arm_split

EXCLUDE = {"B143", "B166"}


def main():
    fam = "d1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "runs_stageC_v13_d1_compat83"))
    args = ap.parse_args()
    out_dir = args.out_dir
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v13] device={device}  d1 compat83 (剔除 {sorted(EXCLUDE)})")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    import pandas as pd
    file = os.path.join(_HERE, "Bosch_aug_v14_109.xlsx")
    raw = sc.build_stageC_raw(device=device, new_excel=file,
                              height_family="h1", recipe_aug_mode="time",
                              stageA_heads_root=stageA)

    df_109 = pd.read_excel(file)
    df_compat = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
    rid_109 = [c for c in df_109.columns if "配方" in str(c)][0]
    rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
    ids_109 = df_109[rid_109].astype(str).tolist()
    ids_compat = set(df_compat[rid_compat].astype(str).tolist())
    extra_idx = [i for i, s in enumerate(ids_109) if s in ids_compat and not (s.startswith("B") and int(s[1:]) < 80)]
    orig_n = sum(1 for s in ids_109 if s.startswith("B") and int(s[1:]) < 80)
    sp = build_arm_split(raw, orig_n, extra_idx, 177, 0.2, 0.1, 1, fam, "compat85")

    recipe_ids = raw["recipe_ids"]
    before = len(sp["train_idx"])
    sp["train_idx"] = [i for i in sp["train_idx"] if recipe_ids[i] not in EXCLUDE]
    print(f"[v13] train {before} -> {len(sp['train_idx'])}")

    exp = sc.ExpCfg(
        name="v13_d1_compat83_head_ln_lr1e-04_ep2000_ss177",
        init="stageB_best", finetune_mode="head_ln", phys7_mode=stageB_phys_mode,
        lr=1e-4, wd=1e-4, l2sp=0.0, backbone_lr_ratio=0.01,
        loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=True)
    run_args = argparse.Namespace(
        out_dir=out_dir, stageA_heads_root=stageA,
        recipe_aug_mode="time", model_type="transformer", batch=4,
        patience=99999, num_workers=0, seed=2026, seed_repeats=1,
        test_ratio=0.2, val_ratio=0.1, min_test_points=1)

    summary = os.path.join(out_dir, "summary_v13.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")
    dk = sc._load_done_keys(summary)
    sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                    experiments=[exp], stageB_best=stageB_best, args=run_args,
                    device=device, summary_all=summary, header_fields=hf,
                    done_keys=dk, epochs=2000, patience=99999,
                    tag="v13", key_recipes=[])
    r = pd.read_csv(summary).iloc[-1]
    print(f"[v13 RESULT] d1 compat83: R2={r.r2:+.4f} (对照 0.7802) trainN={r.trainN} best_ep={r.best_epoch}")


if __name__ == "__main__":
    main()
