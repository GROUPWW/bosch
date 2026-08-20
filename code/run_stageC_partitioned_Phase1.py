#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Per-family 最优配置统一快照验证。

在同一脚本、同一评估框架（fixed test set from original 42）下，
依次运行 4 family 的当前最优配置，形成可复现的基准快照。

各 family 训练子集不同，但 test set 完全统一。
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su

# 当前各 family 最优配置（基于已有实验结果）
BEST_CFG = {
    "zmin": {
        "arm": "compat85",
        "mode": "head_ln",
        "lr": 1e-6,
        "l2sp": 3.0,
        "br": 0.1,
        "epochs": 1500,
        "split_seed": 2026,
    },
    "h1": {
        "arm": "base42",
        "mode": "head_ln",
        "lr": 1e-4,
        "l2sp": 10.0,
        "br": 0.01,
        "epochs": 2000,
        "split_seed": 184,
    },
    "d1": {
        "arm": "compat85",
        "mode": "head_ln",
        "lr": 1e-4,
        "l2sp": 0.0,
        "br": 0.0,
        "epochs": 2000,
        "split_seed": 177,
    },
    "w": {
        "arm": "base42",
        "mode": "full",
        "lr": 1e-5,
        "l2sp": 0.1,
        "br": 0.1,
        "epochs": 500,
        "split_seed": 2026,
    },
}

TARGET = {"zmin": 0.94, "h1": 0.92, "d1": 0.73, "w": 0.82}


def build_arm_split(raw, orig_n, extra_idx, split_seed, test_ratio, val_ratio, min_test_points, fam, arm_tag):
    recipe_ids = raw["recipe_ids"]
    mask = raw["mask"]
    kk = sc.family_to_index(fam)

    orig_ids = recipe_ids[:orig_n]
    scores = np.ones(orig_n, dtype=np.int32)
    tr, va, te = sc.split_with_key_and_quality(
        recipe_ids=orig_ids, key_recipes=[], scores=scores,
        test_ratio=test_ratio, val_ratio=val_ratio, seed=int(split_seed))
    tr = tr.tolist(); va = va.tolist(); te = te.tolist()

    check_idx = te if len(te) > 1 else va
    if int(mask[:, kk, :][check_idx].sum()) < int(min_test_points):
        return None

    if arm_tag == "base42":
        train = tr
    elif arm_tag == "compat85":
        train = tr + list(extra_idx)
    else:
        raise ValueError(f"unknown arm={arm_tag}")

    return {"train_idx": train, "val_idx": va, "test_idx": te,
            "ignored_idx": [], "seed": int(split_seed), "drop_tag": arm_tag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_excel", default=os.path.join(_HERE, "Bosch_aug_v14_109.xlsx"))
    ap.add_argument("--compat_excel", default=os.path.join(_HERE, "Bosch_planB_compatible.xlsx"))
    ap.add_argument("--stageB_runs_root", default=os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA"))
    ap.add_argument("--stageA_heads_root", default=os.path.join(_HERE, "runs_stageA_phys7", "best_by_test"))
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "runs_stageC_partitioned_Phase1"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=99999)
    ap.add_argument("--run_seed", type=int, default=2026)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--families", default="zmin,h1,d1,w")
    args = ap.parse_args()

    device = su.get_default_device()
    print(f"[Phase 1] device={device} 开始统一快照验证")

    # Identify compatible indices
    df_109 = sc.pd.read_excel(args.new_excel)
    df_compat = sc.pd.read_excel(args.compat_excel)
    rid_109 = [c for c in df_109.columns if "配方" in str(c)][0]
    rid_compat = [c for c in df_compat.columns if "配方" in str(c)][0]
    ids_109 = df_109[rid_109].astype(str).tolist()
    ids_compat = set(df_compat[rid_compat].astype(str).tolist())
    extra_idx = [i for i, s in enumerate(ids_109) if s in ids_compat and not (s.startswith("B") and int(s[1:]) < 80)]
    orig_n = sum(1 for s in ids_109 if s.startswith("B") and int(s[1:]) < 80)
    print(f"[Phase 1] orig_n={orig_n} extra_compat={len(extra_idx)}")

    sc.ensure_dir(args.out_dir)
    summary_all = os.path.join(args.out_dir, "summary_allruns.csv")
    header_fields = sc.make_summary_header_fields()
    if not os.path.exists(summary_all):
        with open(summary_all, "w", encoding="utf-8") as f:
            f.write(",".join(header_fields) + "\n")
    done_keys = sc._load_done_keys(summary_all)

    stageB_best = sc.resolve_stageB_best_ckpts_from_common(args.stageB_runs_root)
    best_conf = sc.load_best_config_common(args.stageB_runs_root)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")

    raw = sc.build_stageC_raw(device=device, new_excel=args.new_excel,
                              height_family="h1", recipe_aug_mode="time",
                              stageA_heads_root=args.stageA_heads_root)

    results = []
    log_lines = []

    for fam, cfg in BEST_CFG.items():
        if fam not in [f.strip() for f in args.families.split(",")]:
            continue
        split_seed = cfg["split_seed"]
        arm_tag = cfg["arm"]

        sp = build_arm_split(raw, orig_n, extra_idx, split_seed,
                             0.2, 0.1, 1, fam, arm_tag)
        if sp is None:
            print(f"[WARN] invalid split fam={fam} arm={arm_tag}")
            continue

        run_args = argparse.Namespace(
            out_dir=args.out_dir,
            stageA_heads_root=args.stageA_heads_root,
            recipe_aug_mode="time",
            model_type="transformer",
            batch=args.batch,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.run_seed,
            seed_repeats=1,
            test_ratio=0.2,
            val_ratio=0.1,
            min_test_points=1,
        )

        exp = sc.ExpCfg(
            name=f"Phase1_{fam}_{arm_tag}_{cfg['mode']}_lr{cfg['lr']:.0e}_ep{cfg['epochs']}_ss{split_seed}",
            init="stageB_best",
            finetune_mode=cfg["mode"],
            phys7_mode=stageB_phys_mode,
            lr=cfg["lr"],
            wd=1e-4,
            l2sp=cfg["l2sp"],
            backbone_lr_ratio=cfg["br"],
            loss_type="huber",
            huber_beta=1.0,
            grad_clip=1.0,
            progressive_unfreeze=True,
        )

        print(f"\n{'='*60}")
        print(f"[Phase 1] Running {fam} | arm={arm_tag} | {cfg['mode']} | lr={cfg['lr']:.0e} | l2sp={cfg['l2sp']} | epochs={cfg['epochs']} | ss={split_seed}")
        print(f"{'='*60}")

        sc.run_datasets(
            datasets=[sp], raw=raw, families_eval=[fam],
            experiments=[exp], stageB_best=stageB_best, args=run_args,
            device=device, summary_all=summary_all, header_fields=header_fields,
            done_keys=done_keys, epochs=cfg["epochs"], patience=args.patience,
            tag="Phase1", key_recipes=[])
        done_keys = sc._load_done_keys(summary_all)

    # Print final snapshot
    df = sc.pd.read_csv(summary_all)
    print("\n" + "="*70)
    print("[Phase 1 统一快照结果]")
    print("="*70)
    print(f"{'Family':>6} | {'Train':>6} | {'Test':>5} | {'R²':>8} | {'Target':>7} | {'Gap':>8} | {'Status':>6}")
    print("-"*70)

    for fam, cfg in BEST_CFG.items():
        rows = df[(df.family == fam) & (df.exp.str.contains("Phase1"))]
        if len(rows) == 0:
            # 可能之前已跑过，尝试从 planB 结果引用
            print(f"  {fam:6s} | (引用历史最佳)")
            continue
        r = rows.iloc[0]
        tgt = TARGET[fam]
        gap = r.r2 - tgt
        status = "✅" if r.r2 >= tgt else "🟡" if gap > -0.05 else "❌"
        print(f"  {fam:>6} | {r.trainN:>6} | {r.testN:>5} | {r.r2:>8.4f} | {tgt:>7.2f} | {gap:>+8.4f} | {status:>6}")

    print("="*70)


if __name__ == "__main__":
    main()
