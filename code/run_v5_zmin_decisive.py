#!/usr/bin/env python3
"""
v5 E1: zmin 决定性实验（P-fixed 口径）。

问题：v4_selected 的 0.9596 是 P-pool 口径（test 从 53 条池中划出 11 条，含新配方），
v4fix 的 0.728 是 valN=4 的 best-val 选择伪影（best_ep=247，而 P-pool 同配置峰值在 2360）。
本实验在 P-fixed（固定 8 条原始 hold-out）+ 3000ep 下，用两种 checkpoint 选择规则
重测 zmin 真实上限：

  Arm B (trainval): val_idx = train_idx → best-val 等价 best-train（lr1e-6 单调下降时 ≈ last epoch）
  Arm C (val25):    val_ratio=0.25 → valN≈11，比 v4fix 的 valN=4 噪声小

Arm A（val_ratio=0.1）无需重跑：v4fix 已给出 0.728。
"""
import os, sys, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stageC_paper as sc
import stageB_util as su

FIXED_TEST_RECIPES = ["B17", "B12", "B36", "B70", "B26", "B27", "B30", "B45"]

ARMS = {
    # name -> (val_mode, val_ratio)
    "trainval": ("train", None),   # rule B: best-train ≈ last epoch
    "val25":    ("ratio", 0.25),   # less noisy best-val
}


def build_split(raw, val_mode, val_ratio, seed, fam):
    recipe_ids = raw["recipe_ids"]
    mask = raw["mask"]
    kk = sc.family_to_index(fam)

    test_idx = [i for i, rid in enumerate(recipe_ids) if rid in FIXED_TEST_RECIPES]
    assert len(test_idx) == len(FIXED_TEST_RECIPES)
    remaining = [i for i in range(len(recipe_ids)) if i not in test_idx]

    if val_mode == "train":
        train_idx = remaining
        val_idx = list(remaining)  # val = train → best-val == best-train
    else:
        import random
        random.seed(seed)
        val_n = max(1, int(len(remaining) * val_ratio))
        val_idx = random.sample(remaining, val_n)
        train_idx = [i for i in remaining if i not in val_idx]

    if int(mask[:, kk, :][test_idx].sum()) < 1:
        raise ValueError(f"Empty test set for {fam}")

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            "ignored_idx": [], "seed": seed, "drop_tag": f"v5_{val_mode}"}


def run_arm(arm_name, val_mode, val_ratio, out_dir, device, stageB_best, stageB_phys_mode, stageA_root,
            epochs=3000, seed=2026, file="Bosch_zmin_select_aug.xlsx"):
    fam = "zmin"
    print(f"\n{'='*60}\n[v5-E1] zmin arm={arm_name} val_mode={val_mode} epochs={epochs} seed={seed} file={file}\n{'='*60}")

    raw = sc.build_stageC_raw(device=device,
                              new_excel=os.path.join(_HERE, file),
                              height_family="h1", recipe_aug_mode="time",
                              stageA_heads_root=stageA_root)
    sp = build_split(raw, val_mode, val_ratio, seed, fam)

    run_args = argparse.Namespace(
        out_dir=out_dir, stageA_heads_root=stageA_root,
        recipe_aug_mode="time", model_type="transformer", batch=4,
        patience=99999, num_workers=0, seed=seed, seed_repeats=1,
        test_ratio=0.2, val_ratio=0.1, min_test_points=1)

    exp = sc.ExpCfg(
        name=f"v5e1_zmin_{arm_name}_head_ln_lr1e-06_l2sp10_ep{epochs}_ss2026_rs{seed}",
        init="stageB_best", finetune_mode="head_ln", phys7_mode=stageB_phys_mode,
        lr=1e-6, wd=1e-4, l2sp=10.0, backbone_lr_ratio=0.1,
        loss_type="huber", huber_beta=1.0, grad_clip=1.0, progressive_unfreeze=True)

    summary = os.path.join(out_dir, "summary_v5_e1.csv")
    hf = sc.make_summary_header_fields()
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write(",".join(hf) + "\n")
    dk = sc._load_done_keys(summary)

    sc.run_datasets(datasets=[sp], raw=raw, families_eval=[fam],
                    experiments=[exp], stageB_best=stageB_best, args=run_args,
                    device=device, summary_all=summary, header_fields=hf,
                    done_keys=dk, epochs=epochs, patience=99999,
                    tag="v5e1", key_recipes=[])

    import pandas as pd
    df = pd.read_csv(summary)
    r = df[df.exp.str.contains(arm_name)].iloc[-1]
    print(f"[v5-E1 RESULT] {arm_name}: R2={r.r2:.4f} best_ep={r.best_epoch} "
          f"trainN={r.trainN} valN={r.valN} testN={r.testN}")
    return float(r.r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="trainval,val25")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--file", default="Bosch_zmin_select_aug.xlsx")
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "runs_stageC_v5_decisive"))
    args = ap.parse_args()

    out_dir = args.out_dir
    sc.ensure_dir(out_dir)
    device = su.get_default_device()
    print(f"[v5-E1] device={device}  FIXED TEST: {FIXED_TEST_RECIPES}")

    sroot = os.path.join(_HERE, "runs_stageB_morph_phys7_paperA_best_by_test_fixedA")
    stageB_best = sc.resolve_stageB_best_ckpts_from_common(sroot)
    best_conf = sc.load_best_config_common(sroot)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageA = os.path.join(_HERE, "runs_stageA_phys7", "best_by_test")

    results = {}
    for name in args.arms.split(","):
        vm, vr = ARMS[name.strip()]
        results[name] = run_arm(name, vm, vr, out_dir, device, stageB_best, stageB_phys_mode, stageA,
                                epochs=args.epochs, seed=args.seed, file=args.file)

    print(f"\n{'='*70}\n[v5-E1 汇总] target=0.94 epochs={args.epochs} seed={args.seed}")
    for name, r2 in results.items():
        print(f"  {name:10s} R2={r2:.4f} gap={r2-0.94:+.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
