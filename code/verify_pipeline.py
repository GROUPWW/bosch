# -*- coding: utf-8 -*-
"""
verify_pipeline.py  —  CPU smoke/verification harness for the Bosch pipeline.

Purpose (handover v2): the original trained checkpoints, the raw IEDF tree, and
the tuned best_config were all lost, and every stage script hardcodes Windows
D:\\ paths. This harness does NOT try to reproduce the paper numbers. It verifies
that the *actual author code* (models, dataset builders, losses, transfer path)
runs end-to-end on this machine using only the local data files, on CPU, with
tiny/fast settings.

Strategy:
  - StageA  (recipe -> 7 IEDF descriptors): train phys_model.PhysicsSeqPredictor
    (+ smoke-test MLP/GRU baselines) directly on case_with_phys7.xlsx, using the
    7 precomputed descriptor columns as labels -> needs NO raw IEDF tree.
  - StageB  (recipe + descriptors -> per-cycle morphology): use the real
    stageB_util.build_morph_dataset_phys7 but feed the ground-truth 7 descriptor
    columns via phys7_seq_full -> needs NO stageA head checkpoints. Train
    MorphTransformer (+ smoke-test MorphGRU/MorphMLP).
  - StageC  (transfer to sparse experiment): load Bosch.xlsx via
    physio_util.load_new_excel_as_sparse_morph, then run a few masked fine-tune
    steps of a MorphTransformer on the sparse experimental targets.

Run:  ../.venv/bin/python verify_pipeline.py     (from the code/ directory)
"""
import os, sys, time, json, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                       # so intra-package imports & relative paths resolve
sys.path.insert(0, HERE)

CASE_XLSX = os.path.join(HERE, "case_with_phys7.xlsx")
CASE_SHEET = "Sheet1"
CASE_ID_COL = "input"
BOSCH_XLSX = os.path.join(HERE, "Bosch.xlsx")

RESULTS = {"stageA": {}, "stageB": {}, "stageC": {}, "env": {}}


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    import torch
    import torch.nn as nn
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42); np.random.seed(42)
    dev = "cpu"
    RESULTS["env"] = {"torch": torch.__version__, "device": dev,
                      "cuda": torch.cuda.is_available()}
    print(f"[env] torch={torch.__version__} device={dev} cuda={torch.cuda.is_available()}")

    import stageB_util as su
    su.Cfg.device = "cpu"
    import physio_util as pu
    import phys_model as pm

    # ------------------------------------------------------------------ StageA
    banner("STAGE A  —  recipe(7) -> 7 IEDF physical descriptors")
    try:
        df = pd.read_excel(CASE_XLSX, sheet_name=CASE_SHEET)
        recipe_cols = su._detect_recipe_cols(df.columns.tolist())
        phys_cols = [c for c in su.PHYS7_NAMES if c in df.columns]
        assert len(recipe_cols) == 7, f"expected 7 recipe cols, got {recipe_cols}"
        assert len(phys_cols) == 7, f"expected 7 phys7 cols, got {phys_cols}"
        X = df[recipe_cols].to_numpy(np.float32)
        Y = df[phys_cols].to_numpy(np.float32)
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        X, Y = X[ok], Y[ok]
        N = len(X)
        # standardize
        xm, xs = X.mean(0), X.std(0) + 1e-6
        ym, ys = Y.mean(0), Y.std(0) + 1e-6
        Xn = (X - xm) / xs; Yn = (Y - ym) / ys
        idx = np.random.permutation(N); ntr = int(N * 0.8)
        tr, te = idx[:ntr], idx[ntr:]
        Xt = torch.tensor(Xn); Yt = torch.tensor(Yn)
        tvals = torch.ones(1)  # T=1

        model = pm.PhysicsSeqPredictor(d_model=64, nhead=4, num_layers=2,
                                       dim_ff=128, dropout=0.1, T=1,
                                       in_dim=7, out_dim=7).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        lossf = nn.SmoothL1Loss()
        EP = 60; bs = 64
        for ep in range(EP):
            model.train(); p = np.random.permutation(len(tr))
            for i in range(0, len(tr), bs):
                b = tr[p[i:i + bs]]
                xb = Xt[b]; yb = Yt[b]
                pred = model(xb, tvals)[:, :, 0]          # (B,7)
                loss = lossf(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            yp = model(Xt[te], tvals)[:, :, 0].cpu().numpy()
        yt = Yn[te]
        # per-descriptor R2 (normalized space)
        r2s = []
        for k in range(7):
            ssr = ((yt[:, k] - yp[:, k]) ** 2).sum()
            sst = ((yt[:, k] - yt[:, k].mean()) ** 2).sum() + 1e-9
            r2s.append(float(1 - ssr / sst))
        RESULTS["stageA"] = {"status": "PASS", "N": int(N), "recipe_cols": recipe_cols,
                             "phys_cols": phys_cols, "epochs": EP,
                             "test_R2_per_descriptor": {phys_cols[k]: round(r2s[k], 3) for k in range(7)},
                             "test_R2_mean": round(float(np.mean(r2s)), 3)}
        print(f"[A] N={N} test mean R2={np.mean(r2s):.3f}  per-desc={ {phys_cols[k]:round(r2s[k],2) for k in range(7)} }")
        # smoke-test baselines (1 forward each)
        for name, M in [("mlp", pm.PhysicsMLPBaseline), ("gru", pm.PhysicsGRUBaseline)]:
            try:
                m = M(out_dim=7, T=1); _ = m(Xt[:4], tvals); print(f"[A] baseline {name}: forward OK")
                RESULTS["stageA"].setdefault("baselines_forward_ok", []).append(name)
            except Exception as e:
                print(f"[A] baseline {name}: FAIL {e}")
    except Exception as e:
        RESULTS["stageA"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    # ------------------------------------------------------------------ StageB
    banner("STAGE B  —  recipe + descriptors -> per-cycle morphology (9 cycles)")
    try:
        df = pd.read_excel(CASE_XLSX, sheet_name=CASE_SHEET)
        recipe_cols = su._detect_recipe_cols(df.columns.tolist())
        phys_cols = [c for c in su.PHYS7_NAMES if c in df.columns]
        # drop rows with non-finite recipe/descriptor values (mirrors the row
        # cleaning that the real prepare_shared_cache does) so z-scoring of the
        # descriptor stream cannot emit NaN and poison training.
        fin = (np.isfinite(df[recipe_cols].to_numpy(np.float32)).all(1)
               & np.isfinite(df[phys_cols].to_numpy(np.float32)).all(1))
        df = df[fin].reset_index(drop=True)
        phys7_full = df[phys_cols].to_numpy(np.float32)  # (N,7) ground-truth descriptors
        # bypass lost stageA heads by passing phys7_seq_full
        ds, meta = su.build_morph_dataset_phys7(
            excel_path=CASE_XLSX, sheet_name=CASE_SHEET, case_id_col=CASE_ID_COL,
            target_family=None, phys_source="stagea_pred", recipe_aug_mode="time",
            phys7_mode="full", df=df, recipe_cols=recipe_cols,
            phys7_seq_full=phys7_full,
        )
        N = meta["N"]; Ds = meta["Ds"]; K = meta["K"]; T = meta["T"]
        sp = su.split_dataset_indices(N, seed=0, train_ratio=0.7, val_ratio=0.15)
        tr = np.array(sp["train"]); va = np.array(sp["val"]); te = np.array(sp["test"])
        sx, p7, tg, mk, tm = ds.tensors

        def run_backbone(tag, model, epochs):
            model = model.to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            bs = 64
            for ep in range(epochs):
                model.train(); pp = np.random.permutation(tr)
                for i in range(0, len(tr), bs):
                    b = pp[i:i + bs]
                    pred = model(sx[b], p7[b], tm[b])
                    loss = su.masked_loss(pred, tg[b], mk[b], loss_type="huber", huber_beta=0.1)
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pr = model(sx[te], p7[te], tm[te]).cpu().numpy()
            yt = tg[te].cpu().numpy(); mm = mk[te].cpu().numpy().astype(bool)
            r2 = su.masked_r2_score_np(yt[mm], pr[mm]) if mm.any() else float("nan")
            print(f"[B] {tag}: test masked R2={r2:.3f} (normalized space)")
            return round(float(r2), 3)

        r2_tf = run_backbone("MorphTransformer",
                             su.MorphTransformer(static_dim=Ds, out_dim=K, d_model=128, nhead=4, num_layers=2),
                             epochs=40)
        # smoke-test GRU / MLP backbones (short)
        r2_gru = run_backbone("MorphGRU", su.MorphGRU(static_dim=Ds, out_dim=K, hidden=128, num_layers=1), epochs=10)
        r2_mlp = run_backbone("MorphMLP", su.MorphMLP(static_dim=Ds, out_dim=K, hidden=128, num_layers=3), epochs=10)
        RESULTS["stageB"] = {"status": "PASS", "N": int(N), "Ds": int(Ds), "K": int(K), "T": int(T),
                             "families": meta["families"],
                             "test_R2_transformer_40ep": r2_tf,
                             "test_R2_gru_10ep": r2_gru, "test_R2_mlp_10ep": r2_mlp}
    except Exception as e:
        RESULTS["stageB"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    # ------------------------------------------------------------------ StageC
    banner("STAGE C  —  transfer / calibrate on sparse experimental data (Bosch.xlsx)")
    try:
        recs = pu.load_new_excel_as_sparse_morph(BOSCH_XLSX, height_family="h1")
        n_rec = len(recs)
        # count valid sparse targets
        n_targets = sum(len(r.get("targets", {})) for r in recs)
        # build a static-normalized batch and run a few masked fine-tune steps
        stx = np.array([r["static"] for r in recs], np.float32)
        sm, ss = stx.mean(0), stx.std(0) + 1e-8
        static_b, targ_b, mask_b, _time_raw = pu.build_sparse_batch(
            recs, norm_static_mean=sm, norm_static_std=ss, time_values=su.TIME_VALUES)
        Bc = static_b.shape[0]; Tn = len(su.TIME_LIST)
        # physio_util.build_sparse_batch uses physio_util.TIME_LIST (10 steps,
        # incl. the '9_2' sub-probe); the morph models use 9 cycles. Fold to 9 by
        # keeping the first 9 time columns (author: 9 and 9_2 differ negligibly).
        targ_b = targ_b[:, :, :Tn].contiguous()
        mask_b = mask_b[:, :, :Tn].contiguous()
        time_b = torch.from_numpy(np.tile(su.TIME_VALUES[None, :], (Bc, 1)).astype(np.float32))
        # a fresh morph model (simulating "scratch_full" experiment-only path)
        Ds_c = static_b.shape[1]
        model = su.MorphTransformer(static_dim=Ds_c, out_dim=len(su.FAMILIES), d_model=128, nhead=4, num_layers=2).to(dev)
        # phys7 stream zeroed (no stageA heads) -> shape (B,7,T)
        phys_zero = torch.zeros(Bc, 7, Tn, dtype=torch.float32)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        losses = []
        for ep in range(15):
            model.train()
            pred = model(static_b, phys_zero, time_b)
            loss = su.masked_loss(pred, targ_b, mask_b, loss_type="huber", huber_beta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        RESULTS["stageC"] = {"status": "PASS", "n_experimental_recipes": int(n_rec),
                             "n_sparse_target_points": int(n_targets),
                             "static_dim": int(Ds_c),
                             "finetune_loss_first": round(losses[0], 4),
                             "finetune_loss_last": round(losses[-1], 4),
                             "loss_decreased": bool(losses[-1] < losses[0])}
        print(f"[C] recipes={n_rec} sparse_targets={n_targets} loss {losses[0]:.4f}->{losses[-1]:.4f}")
    except Exception as e:
        RESULTS["stageC"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()

    # ------------------------------------------------------------------ summary
    banner("SUMMARY")
    for s in ["stageA", "stageB", "stageC"]:
        print(f"{s}: {RESULTS[s].get('status','?')}")
    with open(os.path.join(HERE, "verify_results.json"), "w") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    print(f"\n[written] {os.path.join(HERE, 'verify_results.json')}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[done] {time.time()-t0:.1f}s")
