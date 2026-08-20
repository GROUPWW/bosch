#!/usr/bin/env python3
"""v7d: L2-SP 关键对照 seed 重复——把"无效应"升级为有统计支撑的结论。
d1 高漂移格: full lr3e-4, l2sp {0,10} × seeds {2026,42,123,7}
h1 钻石格:  head_ln lr1e-4 ss184, l2sp {0,10} × seeds {2026,42,123}
"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su
from run_stageC_partitioned_Phase1 import build_arm_split

RUNS = [
    # (fam, file, split_tag, ss, mode, lr, l2sp, br, epochs, seeds)
    ("d1", "Bosch_aug_v14_109.xlsx", "arm_compat85", 177, "full",    3e-4, 0.0,  0.1,  500,  [2026, 42, 123, 7]),
    ("d1", "Bosch_aug_v14_109.xlsx", "arm_compat85", 177, "full",    3e-4, 10.0, 0.1,  500,  [2026, 42, 123, 7]),
    ("h1", "Bosch_38_B.xlsx",        "ss184",        184,  "head_ln", 1e-4, 0.0,  0.01, 2000, [2026, 42, 123]),
    ("h1", "Bosch_38_B.xlsx",        "ss184",        184,  "head_ln", 1e-4, 10.0, 0.01, 2000, [2026, 42, 123]),
]


def main():
    out_dir = os.path.join(_HERE, "runs_stageC_v7d_l2sp_seeds")
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v7d] device={device}  L2-SP 对照 seed 重复")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    summary = os.path.join(out_dir, "summary_v7d.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")

    import pandas as pd
    raws, splits = {}, {}
    for fam, file, split_tag, ss, mode, lr, l2sp, br, epochs, seeds in RUNS:
        if file not in raws:
            raws[file] = sc.build_stageC_raw(device=device, new_excel=os.path.join(_HERE, file),
                                             height_family="h1", recipe_aug_mode="time",
                                             stageA_heads_root=stageA)
        raw = raws[file]
        key = (fam, split_tag)
        if key not in splits:
            if split_tag == "arm_compat85":
                df_109 = pd.read_excel(os.path.join(_HERE, file))
                df_compat = pd.read_excel(os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
                rid_109 = [c for c in df_109.columns if "配方" in str(c)][0]
                rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
                ids_109 = df_109[rid_109].astype(str).tolist()
                ids_compat = set(df_compat[rid_compat].astype(str).tolist())
                extra_idx = [i for i, s in enumerate(ids_109) if s in ids_compat and not (s.startswith("B") and int(s[1:]) < 80)]
                orig_n = sum(1 for s in ids_109 if s.startswith("B") and int(s[1:]) < 80)
                splits[key] = build_arm_split(raw, orig_n, extra_idx, ss, 0.2, 0.1, 1, fam, "compat85")
            else:
                splits[key] = sc.build_single_fixed_dataset(raw=raw, key_recipes=[], families_eval=[fam],
                                                            test_ratio=0.2, val_ratio=0.1,
                                                            min_test_points=1, split_seed=ss)
        sp = splits[key]
        for seed in seeds:
            name = f"v7d_{fam}_{mode}_lr{lr:.0e}_l2sp{l2sp:g}_rs{seed}"
            exp = sc.ExpCfg(
                name=name, init="stageB_best", finetune_mode=mode,
                phys7_mode=stageB_phys_mode, lr=lr, wd=1e-4, l2sp=l2sp, backbone_lr_ratio=br,
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
                            done_keys=dk, epochs=epochs, patience=99999,
                            tag="v7d", key_recipes=[])
            r = pd.read_csv(summary).iloc[-1]
            print(f"[v7d RESULT] {name}: R2={r.r2:+.4f} best_ep={r.best_epoch}", flush=True)

    df = pd.read_csv(summary)
    df["l2sp"] = df.exp.str.extract(r"l2sp([\d.]+)_rs").astype(float)
    df["fam_mode"] = df.exp.str.extract(r"v7d_(\w+_\w+)_")
    print("\n[v7d 统计]")
    for (fm, l2), g in df.groupby(["fam_mode", "l2sp"]):
        print(f"  {fm} l2sp={l2:g}: n={len(g)} mean={g.r2.mean():+.4f} std={g.r2.std():.4f}")


if __name__ == "__main__":
    main()
