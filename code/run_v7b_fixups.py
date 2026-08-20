#!/usr/bin/env python3
"""v7b: 修正 c-i 为真 zero-shot（epochs=-1 绕过 `or 200` 回退）+ L2-SP 钻石配置 on/off。"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su
from run_v7_transfer_ablation import VARIANTS, FAM_CFG, fixed8_split
from run_stageC_partitioned_Phase1 import build_arm_split

FIXED_TEST_RECIPES = ["B17", "B12", "B36", "B70", "B26", "B27", "B30", "B45"]


def main():
    out_dir = os.path.join(_HERE, "runs_stageC_v7b_fixups")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    summary = os.path.join(out_dir, "summary_v7b.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd

    # ---- Part 1: c-i 真 zero-shot（4 family，epochs=-1）----
    for fam, fc in FAM_CFG.items():
        raw = sc.build_stageC_raw(device=device, new_excel=os.path.join(_HERE, fc["file"]),
                                  height_family="h1", recipe_aug_mode="time",
                                  stageA_heads_root=stageA)
        if fc["split"] == "fixed8":
            sp = fixed8_split(raw, fam)
        else:
            df_109 = pd.read_excel(os.path.join(_HERE, fc["file"]))
            df_compat = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
            rid_109 = [c for c in df_109.columns if "配方" in str(c)][0]
            rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
            ids_109 = df_109[rid_109].astype(str).tolist()
            ids_compat = set(df_compat[rid_compat].astype(str).tolist())
            extra_idx = [i for i, s in enumerate(ids_109) if s in ids_compat and not (s.startswith("B") and int(s[1:]) < 80)]
            orig_n = sum(1 for s in ids_109 if s.startswith("B") and int(s[1:]) < 80)
            sp = build_arm_split(raw, orig_n, extra_idx, fc["ss"], 0.2, 0.1, 1, fam, fc["arm"])

        exp = sc.ExpCfg(
            name=f"v7b_c-i_true_zeroshot_{fam}", init="stageB_best", finetune_mode="full",
            phys7_mode=stageB_phys_mode, lr=fc["lr"], wd=1e-4, l2sp=0.0, backbone_lr_ratio=0.1,
            loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=False)
        exp.epochs = -1  # 绕过 `or 200` 回退，真 zero-shot
        run_args = argparse.Namespace(
            out_dir=out_dir, stageA_heads_root=stageA,
            recipe_aug_mode="time", model_type="transformer", batch=4,
            patience=99999, num_workers=0, seed=2026, seed_repeats=1,
            test_ratio=0.2, val_ratio=0.1, min_test_points=1)
        dk = sc._load_done_keys(summary)
        sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                        experiments=[exp], stageB_best=stageB_best, args=run_args,
                        device=device, summary_all=summary, header_fields=hf,
                        done_keys=dk, epochs=-1, patience=99999,
                        tag="v7b", key_recipes=[])
        r = pd.read_csv(summary).iloc[-1]
        print(f"[v7b RESULT] c-i真zero-shot {fam:5s} R2={r.r2:+.4f} best_ep={r.best_epoch}", flush=True)

    # ---- Part 2: L2-SP 钻石配置 on/off（zmin 3000ep trainval + h1 2000ep ss184）----
    L2SP_RUNS = [
        # (fam, file, split_fn_tag, ss, lr, l2sp, br, epochs, name)
        ("zmin", "Bosch_zmin_select_aug.xlsx", "fixed8", 2026, 1e-6, 0.0,  0.1, 3000, "v7b_zmin_l2sp0_ep3000"),
        ("zmin", "Bosch_zmin_select_aug.xlsx", "fixed8", 2026, 1e-6, 10.0, 0.1, 3000, "v7b_zmin_l2sp10_ep3000"),
        ("h1",   "Bosch_38_B.xlsx",            "ss184",  184,  1e-4, 0.0,  0.01, 2000, "v7b_h1_l2sp0_ep2000"),
        ("h1",   "Bosch_38_B.xlsx",            "ss184",  184,  1e-4, 10.0, 0.01, 2000, "v7b_h1_l2sp10_ep2000"),
    ]
    raws = {}
    for fam, file, split_tag, ss, lr, l2sp, br, epochs, name in L2SP_RUNS:
        if file not in raws:
            raws[file] = sc.build_stageC_raw(device=device, new_excel=os.path.join(_HERE, file),
                                             height_family="h1", recipe_aug_mode="time",
                                             stageA_heads_root=stageA)
        raw = raws[file]
        if split_tag == "fixed8":
            sp = fixed8_split(raw, fam)
        else:
            sp = sc.build_single_fixed_dataset(raw=raw, key_recipes=[], families_eval=[fam],
                                               test_ratio=0.2, val_ratio=0.1,
                                               min_test_points=1, split_seed=ss)
        exp = sc.ExpCfg(
            name=name, init="stageB_best", finetune_mode="head_ln",
            phys7_mode=stageB_phys_mode, lr=lr, wd=1e-4, l2sp=l2sp, backbone_lr_ratio=br,
            loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=True)
        run_args = argparse.Namespace(
            out_dir=out_dir, stageA_heads_root=stageA,
            recipe_aug_mode="time", model_type="transformer", batch=4,
            patience=99999, num_workers=0, seed=2026, seed_repeats=1,
            test_ratio=0.2, val_ratio=0.1, min_test_points=1)
        dk = sc._load_done_keys(summary)
        sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                        experiments=[exp], stageB_best=stageB_best, args=run_args,
                        device=device, summary_all=summary, header_fields=hf,
                        done_keys=dk, epochs=epochs, patience=99999,
                        tag="v7b", key_recipes=[])
        r = pd.read_csv(summary).iloc[-1]
        print(f"[v7b RESULT] {name}: R2={r.r2:+.4f} best_ep={r.best_epoch}", flush=True)


if __name__ == "__main__":
    main()
