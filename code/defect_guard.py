# -*- coding: utf-8 -*-
"""
defect_guard.py — R1-C2 缺陷护栏：recipe → P(形貌缺陷)

两部分：
  a) 数据 QC 规则（后验）：从已测配方归纳的客观识别规则，输出逐配方判定表 + 混淆矩阵。
     规则（从 40 条已测数据归纳，阈值定稿见 RULES 注释）：
       R1 温度异常：temp_C ≠ -10
       R2 w 爆炸式扩宽：w@9/w@1 > 1.3 且 w@9 > 1000 nm
       R3 bowing 剖面（中段收窄后再扩宽超过开口）：
          w@9 > w@1 且 (w@1 - min(w@3,w@5))/w@1 ≥ 4% 且 w@9/min(w@3,w@5) ≥ 1.12
  b) recipe→P(缺陷) 分类器（先验）：sklearn LogisticRegression / GradientBoosting，
     特征 = 7 维 recipe 参数（对比版再加 stageA 预测的 7 维 phys7 描述符），LOO-CV 评 AUC。

标签口径：
  正例 = B143/B166/B177/B192/B215（作者判定不可用，v14 Excel unusable=1 一致）。
  B113/B114/B116/B144/151 = 非 -10°C 温度异常 → 单独一类，不进训练/评估
    （其中只有 B116 有实测；B117/118/119 无测量且 -20°C，同样单独登记）。
  负例 = 其余已测且 -10°C 的配方（34 条）。

局限：正例仅 5 条，分类器是 proof-of-concept；规则护栏是主防线，分类器是量化补充。

用法：cd code && .venv/bin/python defect_guard.py   （全程 CPU）
"""
import os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

EXCEL_PATH = os.path.join(_HERE, "Bosch_aug_v14_109.xlsx")
CSV_PATH = os.path.join(_HERE, "..", "materials", "新补实验数据记录V2.csv")
OUT_DIR = os.path.join(_HERE, "runs_defect_guard")

RECIPE_COLS = ["APC（E2步骤）", "source_RF（E2步骤）", "LF_RF（E2步骤）",
               "SF6（E2步骤）", "C4F8（DEP步骤）", "DEP time", "etch time"]
RECIPE_SHORT = ["APC", "source_RF", "LF_RF", "SF6", "C4F8", "DEP_time", "etch_time"]

POS_IDS = ["B143", "B166", "B177", "B192", "B215"]
TEMP_ANOM_IDS = ["B113", "B114", "B116", "B144", "B151"]   # 非 -10°C（B117/118/119 也无测量）
UNMEASURED_IDS = ["B117", "B118", "B119"]

# 规则阈值（从 40 条已测数据归纳定稿）
W_RATIO_TH = 1.3      # R2: w@9/w@1
W9_ABS_TH = 1000.0    # R2: w@9 绝对宽度 (nm)
W_DIP_TH = 0.04       # R3: 中段相对开口的收窄幅度
W_REBOUND_TH = 1.12   # R3: 末段相对中段最窄处的再扩宽比

LINES = []  # 收集报告文本


def rep(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


# ---------------------------------------------------------------- data
def load_measured() -> pd.DataFrame:
    """读 V2 CSV（双行表头），返回 40 条有实测的配方 + 实测 w/h/z + temp。"""
    raw = pd.read_csv(CSV_PATH, header=[0, 1])
    raw.columns = ["recipe", "temp_C",
                   "z_pred", "z", "h3_pred", "h3", "h5_pred", "h5", "h9_pred", "h9",
                   "d3_pred", "d3", "d5_pred", "d5", "d9_pred", "d9",
                   "w1_pred", "w1", "w3_pred", "w3", "w5_pred", "w5", "w9_pred", "w9"]
    m = raw[raw["w9"].notna()].copy().reset_index(drop=True)
    m["recipe"] = m["recipe"].astype(str).str.strip()
    return m


def join_recipe_params(m: pd.DataFrame) -> pd.DataFrame:
    """从 v14 Excel 取 7 维 recipe 参数，按配方名 join；交叉核对 unusable 列。"""
    df = pd.read_excel(EXCEL_PATH)
    df["配方名"] = df["配方名"].astype(str).str.strip()
    sub = df[["配方名", "unusable"] + RECIPE_COLS].rename(columns={"配方名": "recipe"})
    out = m.merge(sub, on="recipe", how="left", validate="1:1")
    missing = out[out[RECIPE_COLS[0]].isna()]["recipe"].tolist()
    assert not missing, f"v14 Excel 缺已测配方的 recipe 参数: {missing}"
    # 交叉核对：Excel unusable=1 应恰好是 5 条正例
    xl_pos = sorted(out.loc[out["unusable"] == 1.0, "recipe"].tolist())
    assert xl_pos == sorted(POS_IDS), f"Excel unusable 标记与正例名单不一致: {xl_pos}"
    return out


def assign_labels(m: pd.DataFrame) -> pd.DataFrame:
    def _lab(r):
        if r["recipe"] in POS_IDS:
            return "pos"
        if (r["recipe"] in TEMP_ANOM_IDS) or (float(r["temp_C"]) != -10.0):
            return "temp_anomaly"
        return "neg"
    m["label"] = m.apply(_lab, axis=1)
    return m


# ---------------------------------------------------------------- part a: rules
def apply_rules(m: pd.DataFrame) -> pd.DataFrame:
    w1, w3, w5, w9 = m["w1"], m["w3"], m["w5"], m["w9"]
    wmin = np.minimum(w3, w5)
    m["w_ratio"] = w9 / w1
    m["w_dip"] = (w1 - wmin) / w1
    m["w_rebound"] = w9 / wmin
    m["h_jump"] = np.maximum(np.abs(m["h5"] - m["h3"]) / m["h3"],
                             np.abs(m["h9"] - m["h5"]) / m["h5"])
    m["R1_temp"] = m["temp_C"] != -10
    m["R2_wexplode"] = (m["w_ratio"] > W_RATIO_TH) & (w9 > W9_ABS_TH)
    m["R3_bowing"] = (w9 > w1) & (m["w_dip"] >= W_DIP_TH) & (m["w_rebound"] >= W_REBOUND_TH)
    m["flag"] = m["R1_temp"] | m["R2_wexplode"] | m["R3_bowing"]
    return m


def part_a(m: pd.DataFrame):
    rep("=" * 72)
    rep("Part A · 数据 QC 规则（后验）")
    rep("=" * 72)
    rep(f"规则定稿：R1 temp≠-10°C | R2 w@9/w@1>{W_RATIO_TH} 且 w@9>{W9_ABS_TH:.0f}nm | "
        f"R3 w@9>w@1 且 中段收窄≥{W_DIP_TH:.0%} 且 末段反弹≥{W_REBOUND_TH}")
    rep("")

    eval_set = m[m["label"].isin(["pos", "neg"])].copy()
    n_pos = int((eval_set["label"] == "pos").sum())
    n_neg = int((eval_set["label"] == "neg").sum())

    cols = ["recipe", "temp_C", "label", "w1", "w3", "w5", "w9",
            "w_ratio", "w_dip", "w_rebound", "h_jump",
            "R1_temp", "R2_wexplode", "R3_bowing", "flag"]
    rep("逐配方规则判定表（已测 40 条；temp_anomaly 单独列出，不计入混淆矩阵）：")
    rep(eval_set[cols].sort_values(["flag", "w_ratio"], ascending=[False, False]).to_string(index=False))
    ta = m[m["label"] == "temp_anomaly"]
    if len(ta):
        rep("")
        rep("温度异常类（单独登记，不进训练/评估；被 R1 正确标记为超分布）：")
        rep(ta[cols].to_string(index=False))

    tp = int(((eval_set["label"] == "pos") & eval_set["flag"]).sum())
    fn = n_pos - tp
    fp = int(((eval_set["label"] == "neg") & eval_set["flag"]).sum())
    tn = n_neg - fp
    rep("")
    rep(f"混淆矩阵（n={len(eval_set)} = {n_pos} pos + {n_neg} neg）：")
    rep(f"  TP={tp}  FN={fn}  |  recall    = {tp}/{n_pos} = {tp/n_pos:.3f}")
    rep(f"  FP={fp}  TN={tn}  |  precision = {tp}/{max(1,tp+fp)} = {tp/max(1,tp+fp):.3f}"
        f"   FPR = {fp}/{n_neg} = {fp/n_neg:.3f}")
    fp_ids = eval_set.loc[(eval_set["label"] == "neg") & eval_set["flag"], "recipe"].tolist()
    fn_ids = eval_set.loc[(eval_set["label"] == "pos") & ~eval_set["flag"], "recipe"].tolist()
    rep(f"  误报: {fp_ids if fp_ids else '无'}   漏报: {fn_ids if fn_ids else '无'}")
    for pid in POS_IDS:
        r = eval_set[eval_set["recipe"] == pid].iloc[0]
        fired = [c for c in ["R1_temp", "R2_wexplode", "R3_bowing"] if r[c]]
        rep(f"  正例 {pid}: 捕获规则={fired} (w_ratio={r['w_ratio']:.3f}, "
            f"dip={r['w_dip']:.3f}, rebound={r['w_rebound']:.3f}, h_jump={r['h_jump']:.3f})")
    assert fn == 0, f"规则未捕获全部正例: {fn_ids}"
    rep("")
    rep("[OK] 5 条正例全部被规则捕获")

    eval_set[cols].to_csv(os.path.join(OUT_DIR, "rule_table.csv"), index=False)
    return eval_set


# ---------------------------------------------------------------- part b: classifier
def get_phys7(recipe_raw: np.ndarray) -> np.ndarray:
    """stageA 集成预测的 7 维 phys7 描述符（CPU）。与 build_stageC_raw 同一来源，
    直接调 StageAEnsemblePhys7Provider（build_stageC_raw 内部也是调它）。"""
    import stageB_util as sb
    provider = sb.StageAEnsemblePhys7Provider(
        heads_root=sb.Cfg.stageA_heads_root,
        device="cpu",
        recipe_cols_in=None,
        expect_k=7,
    )
    return provider.infer(recipe_raw, phys7_mode="full", use_cache=True).astype(np.float32)


def loo_probs(X: np.ndarray, y: np.ndarray, make_model) -> np.ndarray:
    from sklearn.model_selection import LeaveOneOut
    n = len(y)
    prob = np.zeros(n, dtype=np.float64)
    for tr_idx, te_idx in LeaveOneOut().split(X):
        model = make_model()
        model.fit(X[tr_idx], y[tr_idx])
        prob[te_idx[0]] = model.predict_proba(X[te_idx])[0, 1]
    return prob


def make_lr():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))


def make_gbm():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2,
                                      random_state=42)


def part_b(eval_set: pd.DataFrame):
    from sklearn.metrics import roc_auc_score

    rep("")
    rep("=" * 72)
    rep("Part B · recipe→P(缺陷) 分类器（先验，LOO-CV）")
    rep("=" * 72)

    y = (eval_set["label"] == "pos").to_numpy(int)
    ids = eval_set["recipe"].to_numpy()
    Xr = eval_set[RECIPE_COLS].to_numpy(np.float32)

    rep("获取 stageA phys7 描述符（CPU）...")
    phys7 = get_phys7(Xr)
    Xp = np.concatenate([Xr, phys7], axis=1)

    feat_sets = {"recipe-only": (Xr, RECIPE_SHORT),
                 "recipe+phys7": (Xp, RECIPE_SHORT + list(__import__("stageB_util").PHYS7_NAMES))}
    models = {"LogReg": make_lr, "GBM": make_gbm}

    summary = {}
    pos_rows = {}
    for fs_name, (X, fnames) in feat_sets.items():
        for m_name, mk in models.items():
            prob = loo_probs(X, y, mk)
            auc = roc_auc_score(y, prob)
            summary[(fs_name, m_name)] = auc
            pos_rows[(fs_name, m_name)] = prob
            rep(f"  LOO AUC [{fs_name:13s}] [{m_name:7s}] = {auc:.3f}  "
                f"(n={len(y)}, pos={int(y.sum())})")

    rep("")
    rep("5 条正例的 LOO 预测概率（按 GBM recipe+phys7 排序）：")
    order = np.argsort(-pos_rows[("recipe+phys7", "GBM")])
    hdr = "  " + f"{'recipe':8s}" + "".join(f"{fs+'/'+m:>22s}" for fs, m in pos_rows.keys())
    rep(hdr)
    for i in order:
        if y[i] != 1:
            continue
        row = f"  {ids[i]:8s}" + "".join(f"{pos_rows[k][i]:>22.3f}" for k in pos_rows.keys())
        rep(row)

    # 特征重要性：GBM 在全量数据上 fit
    for fs_name, (X, fnames) in feat_sets.items():
        gbm = make_gbm().fit(X, y)
        imp = pd.Series(gbm.feature_importances_, index=fnames).sort_values(ascending=False)
        rep("")
        rep(f"特征重要性 top5 [{fs_name}]（GBM, 全量 fit）：")
        for k, v in imp.head(5).items():
            rep(f"  {k:20s} {v:.3f}")

    # B116（温度异常类，未参与训练）的预测概率，作参考
    rep("")
    rep(f"参考：B116（-20°C 温度异常，未参与训练/评估）不属于本二分类口径，未单独预测。")

    # 保存 LOO 预测明细
    out = pd.DataFrame({"recipe": ids, "y": y})
    for k, prob in pos_rows.items():
        out[f"prob_{k[0]}_{k[1]}"] = prob
    out.to_csv(os.path.join(OUT_DIR, "loo_predictions.csv"), index=False)
    return summary


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rep("defect_guard — R1-C2 缺陷护栏（CPU）")
    rep(f"数据: {os.path.basename(EXCEL_PATH)} + {os.path.relpath(CSV_PATH, _HERE)}")
    rep("")

    m = assign_labels(join_recipe_params(load_measured()))
    n_lab = m["label"].value_counts().to_dict()
    rep(f"已测配方 {len(m)} 条：pos={n_lab.get('pos',0)}, neg={n_lab.get('neg',0)}, "
        f"temp_anomaly={n_lab.get('temp_anomaly',0)}（单独类，不进训练/评估）")
    rep(f"未测/不可评（-20°C，登记不进任何集合）: {UNMEASURED_IDS}")
    m = apply_rules(m)

    eval_set = part_a(m)
    summary = part_b(eval_set)

    rep("")
    rep("=" * 72)
    rep("结论与局限")
    rep("=" * 72)
    rep("- 规则护栏（后验 QC）：5/5 正例捕获，是主防线；阈值从 40 条已测数据归纳，"
        "B166/B192 依赖 R3 bowing 剖面规则，个别样本贴近阈值（B166 rebound=1.137 vs 1.12，"
        "B192 dip=4.07% vs 4%），新数据到来时应复核。")
    auc_r = summary.get(("recipe-only", "GBM"), float("nan"))
    auc_p = summary.get(("recipe+phys7", "GBM"), float("nan"))
    rep(f"- 分类器（先验）：正例仅 5 条，LOO AUC 波动大，仅作量化补充（proof-of-concept）；"
        f"GBM AUC recipe-only={auc_r:.3f} / recipe+phys7={auc_p:.3f}。")
    rep("- B143 的缺陷由 0°C 温度驱动，但 stageB 的 7 维 recipe 特征不含温度；"
        "先验分类器对温度型缺陷的识别能力有限，温度由 R1 规则兜底。")
    rep("- B117/118/119（未测、-20°C）待作者确认上机失败后再决定是否并入正例。")

    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    print(f"\n报告已写入: {os.path.join(OUT_DIR, 'report.txt')}", flush=True)


if __name__ == "__main__":
    main()
