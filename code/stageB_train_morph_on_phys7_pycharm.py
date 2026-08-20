# -*- coding: utf-8 -*-
import os, re, csv, math, json, time, argparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from dataclasses import dataclass
from stageB_util import *
from stageB_util import (
    _PHYS7_GROUPS,
    _canon,
    _default_family_sign_and_nonneg,
    _detect_recipe_cols,
    _detect_target_col,
    _ensure_dir,
    _mask_coverage,
    _norm_case_id,
    _zscore_apply,
    _zscore_fit, _torch_load_ckpt_trusted,
)

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

plt.ioff()


@dataclass
class StageAHeadInfo:
    head_index: int
    head_name: str
    ckpt_path: str


def scan_stageA_heads(heads_root: str, expect_k: int = 7) -> List[StageAHeadInfo]:
    if os.path.isfile(heads_root) and heads_root.endswith(".pth"):
        raise RuntimeError(
            f"[StageA] heads_root must be a directory with 7 heads, but got a single ckpt file:\n"
            f"  {heads_root}\n"
            f"Please set Cfg.stageA_heads_root = runs_stageA_phys7/best_by_test (or similar)."
        )

    if not os.path.isdir(heads_root):
        raise FileNotFoundError(f"StageA heads_root not found: {heads_root}")

    infos: List[StageAHeadInfo] = []
    for name in os.listdir(heads_root):
        if not name.startswith("head_"):
            continue
        m = re.match(r"head_(\d+)_", name)
        if not m:
            continue
        idx = int(m.group(1))
        head_dir = os.path.join(heads_root, name)
        ckpt_path = os.path.join(head_dir, "phys7_best.pth")
        if os.path.isfile(ckpt_path):
            head_name = name.split("_", 2)[-1] if len(name.split("_", 2)) == 3 else f"h{idx}"
            infos.append(StageAHeadInfo(idx, head_name, ckpt_path))

    infos.sort(key=lambda x: x.head_index)
    if len(infos) != expect_k:
        raise RuntimeError(
            f"[StageA] heads count mismatch: found={len(infos)} expect={expect_k}\n"
            f"root={heads_root}\n"
            f"found={[(i.head_index, os.path.basename(os.path.dirname(i.ckpt_path))) for i in infos]}"
        )
    return infos


def _resolve_recipe_cols_from_df(df_cols: List[str]) -> Dict[str, str]:
    cols_c = {c: _canon(c) for c in df_cols}
    out = {}
    for key, pats in RECIPE_KEY_ALIAS.items():
        hit = None
        for c, v in cols_c.items():
            if any(p in v for p in pats):
                hit = c
                break
        if hit is not None:
            out[key] = hit
    return out


def build_job_list_fullgrid() -> List[Dict[str, str]]:
    """完全复刻你现在的全网格 sweep（不含 family/seed）。"""
    jobs: List[Dict[str, str]] = []
    for model_type in Cfg.model_types:
        for phys_source in Cfg.phys_sources:
            for aug in Cfg.recipe_aug_modes:
                for pm in Cfg.phys7_modes:
                    jobs.append(dict(
                        model_type=model_type,
                        phys_source=phys_source,
                        recipe_aug_mode=aug,
                        phys7_mode=pm,
                    ))
    return jobs


def build_job_list_ablationA() -> List[Dict[str, str]]:
    base = dict(
        model_type=getattr(Cfg, "baseline_model_type", "transformer"),
        phys_source=getattr(Cfg, "baseline_phys_source", "stageA_pred"),
        recipe_aug_mode=getattr(Cfg, "baseline_recipe_aug_mode", "time"),
        phys7_mode=getattr(Cfg, "baseline_phys7_mode", "full"),
    )

    jobs: List[Dict[str, str]] = [base]

    # 1) phys_source ablation（baseline 已含 stageA_pred）
    for ps in getattr(Cfg, "ablate_phys_sources", ["none"]):
        jobs.append({**base, "phys_source": ps})

    # 2) model ablation
    for mt in getattr(Cfg, "ablate_model_types", ["gru", "mlp"]):
        jobs.append({**base, "model_type": mt})

    # 3) aug ablation（time 已是 baseline）
    for aug in getattr(Cfg, "ablate_recipe_aug_modes", ["base", "gas", "rf", "squares"]):
        jobs.append({**base, "recipe_aug_mode": aug})

    # 4) phys7 ablation（full 已是 baseline）
    for pm in getattr(Cfg, "ablate_phys7_modes", ["only_energy", "only_flux", "none"]):
        jobs.append({**base, "phys7_mode": pm})

    uniq = []
    seen = set()
    for j in jobs:
        key = (j["model_type"], j["phys_source"], j["recipe_aug_mode"], j["phys7_mode"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(j)
    return uniq


def build_job_list(plan: str) -> List[Dict[str, str]]:
    """根据 plan 生成要跑的配置列表（不含 family/seed）。"""
    plan = (plan or "fullgrid").lower().strip()

    if plan in ["full", "grid", "fullgrid"]:
        return build_job_list_fullgrid()
    if plan in ["ablationa", "abl_a", "ablation"]:
        return build_job_list_ablationA()

    if plan in ["phase1", "p1"]:
        return build_job_list_phase1()
    if plan in ["phase2", "p2"]:
        return build_job_list_phase2()

    # ✅ 新增：auto 只是一个调度入口，这里返回空列表即可（真正跑在 main 里分两段跑）
    if plan in ["auto", "p12", "phase12"]:
        return []

    raise ValueError(f"Unknown plan={plan}, expected: fullgrid / ablationA / phase1 / phase2 / auto")


def _run_one_plan(
        runs_root: str,
        plan: str,
        fam_list: List[str],
        df_cache: pd.DataFrame,
        recipe_cols_cache: List[str],
        recipe_raw_cache: np.ndarray,
        stageA_provider_cache: Optional["StageAEnsemblePhys7Provider"],
        targets_full: np.ndarray,
        mask_full: np.ndarray,
        phys7_seq_cache: dict,
        case_ids_cache: Optional[np.ndarray],
        do_plots: bool = False,
):
    plan_l = str(plan).lower().strip()

    # seeds 策略：Phase1 少量 seeds；Phase2 固定 Phase1 best seed
    if plan_l in ["phase1", "p1"]:
        split_seeds = list(getattr(Cfg, "phase1_split_seeds", [0, 1, 2]))
    elif plan_l in ["phase2", "p2"]:
        best_seed = _load_best_seed_from_phase1(runs_root)
        if best_seed is None:
            split_seeds = list(getattr(Cfg, "phase2_split_seeds", [0]))
            log(f"[RUN] phase2: best_seed not found, fallback split_seeds={split_seeds}")
        else:
            split_seeds = [int(best_seed)]
            log(f"[RUN] phase2: use Phase1 best_seed={best_seed}")
    else:
        split_seeds = list(Cfg.split_seeds)

    job_confs = build_job_list(plan_l)
    if plan_l in ["auto", "p12", "phase12"]:
        raise ValueError("_run_one_plan should not be called with plan=auto")

    log("=" * 60)
    log(f"[RUN] runs_root={runs_root}")
    log(f"[RUN] plan={plan_l}")
    log(f"[RUN] families_to_train={fam_list}")
    log(f"[RUN] split_seeds={split_seeds}")
    log(f"[RUN] job_confs={len(job_confs)} (model/phys/aug/phys7 combinations)")
    log("=" * 60)

    total_jobs = len(fam_list) * len(job_confs) * len(split_seeds)
    log(f"[RUN] total_jobs={total_jobs}")

    rows = []
    job_idx = 0
    summary_fieldnames = None

    for fam in fam_list:
        for conf in job_confs:
            for sd in split_seeds:
                job_idx += 1
                model_type = conf["model_type"]
                phys_source = conf["phys_source"]
                aug = conf["recipe_aug_mode"]
                pm = conf["phys7_mode"]

                log(f"\n[JOB {job_idx}/{total_jobs}] fam={fam} model={model_type} phys={phys_source} aug={aug} phys7={pm} split_seed={sd}")

                try:
                    r = run_one_experiment(
                        model_type=model_type,
                        phys_source=phys_source,
                        recipe_aug_mode=aug,
                        phys7_mode=pm,
                        root_out=runs_root,
                        split_seed=sd,
                        job_idx=job_idx,
                        job_total=total_jobs,
                        target_family=fam,
                        shared_df=df_cache,
                        shared_recipe_cols=recipe_cols_cache,
                        shared_recipe_raw=recipe_raw_cache,
                        shared_targets_full=targets_full,
                        shared_mask_full=mask_full,
                        shared_phys7_seq_cache=phys7_seq_cache,
                        shared_stageA_provider=stageA_provider_cache,
                    )
                    rows.append(r)
                    if summary_fieldnames is None:
                        summary_fieldnames = list(r.keys())
                    append_summary_row(r, runs_root, fieldnames=summary_fieldnames)
                except Exception as e:
                    log(f"      [FAIL] fam={fam} seed={sd} : {e}")

    # 关键：Phase1 结束必须先聚合，才能产出 best_seed 给 Phase2 用
    postprocess_summary_from_csv(runs_root, write_excel=True)

    if do_plots:
        render_topk_plots(runs_root, topk=10, split_name="test")
        render_compare_plots(runs_root)

    return rows


def _load_best_seed_from_phase1(runs_root: str) -> Optional[int]:
    p = os.path.join(runs_root, "best_config_common_all_families.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        sd = d.get("split_seed", None)
        return int(sd) if sd is not None else None
    except Exception:
        return None


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        opt: torch.optim.Optimizer,
        device: str,
        loss_type: str = "mse",
        huber_beta: float = 0.1,
        grad_clip: float = 0.0,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for static_x, phys7_seq, y, m, time_mat in loader:
        static_x = static_x.to(device)
        phys7_seq = phys7_seq.to(device)
        y = y.to(device)
        m = m.to(device)
        time_mat = time_mat.to(device)

        pred = model(static_x, phys7_seq, time_mat)
        loss = masked_loss(pred, y, m, loss_type=loss_type, huber_beta=huber_beta)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()

        total += float(loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_epoch(
        model: nn.Module,
        loader,
        device: str,
        return_pack: bool = False,
        max_batches: Optional[int] = None,
) -> Tuple[float, Dict]:
    model.eval()
    total = 0.0
    n = 0

    preds_cpu = []
    ys_cpu = []
    ms_cpu = []

    for bi, (static_x, phys7_seq, y, m, time_mat) in enumerate(loader):
        if (max_batches is not None) and (bi >= max_batches):
            break

        static_x = static_x.to(device, non_blocking=True)
        phys7_seq = phys7_seq.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        time_mat = time_mat.to(device, non_blocking=True)

        pred = model(static_x, phys7_seq, time_mat)
        loss = masked_mse(pred, y, m)

        total += float(loss.item())
        n += 1

        if return_pack:
            # 用 torch tensor 先在 CPU 累积，最后一次性 cat 再转 numpy（比每步 numpy 更省）
            preds_cpu.append(pred.detach().cpu())
            ys_cpu.append(y.detach().cpu())
            ms_cpu.append(m.detach().cpu())

    if return_pack and preds_cpu:
        pred_all = torch.cat(preds_cpu, dim=0).numpy()
        y_all = torch.cat(ys_cpu, dim=0).numpy()
        m_all = torch.cat(ms_cpu, dim=0).numpy().astype(bool)
    else:
        # 兜底：保持你原来推断 K/T 的逻辑（避免空loader时崩）
        K = None
        try:
            if hasattr(model, "out") and isinstance(model.out, nn.Linear):
                K = int(model.out.out_features)
        except Exception:
            pass
        if K is None:
            try:
                if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
                    for layer in reversed(list(model.net)):
                        if isinstance(layer, nn.Linear):
                            K = int(layer.out_features)
                            break
            except Exception:
                pass
        if K is None:
            K = len(FAMILIES)

        T = len(TIME_LIST)
        pred_all = np.zeros((0, K, T), np.float32)
        y_all = np.zeros((0, K, T), np.float32)
        m_all = np.zeros((0, K, T), bool)

    pack = {"pred_norm": pred_all, "y_norm": y_all, "mask": m_all}
    return total / max(1, n), pack


@torch.no_grad()
def eval_epoch_loss(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        max_batches: Optional[int] = None,
        loss_type: str = "mse",
        huber_beta: float = 0.1,
) -> float:
    loss, _ = eval_epoch_stats(
        model=model,
        loader=loader,
        device=device,
        loss_type=loss_type,
        huber_beta=huber_beta,
        max_batches=max_batches,
    )
    return float(loss)


@torch.no_grad()
def eval_epoch_stats(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        loss_type: str = "mse",
        huber_beta: float = 0.1,
        max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    计算：
      - mean loss（masked）
      - masked R2（在 norm 空间直接算，够用且稳定）
    不存全量数组，流式统计，开销小。
    """
    model.eval()

    total_loss = 0.0
    nb = 0

    # R2 streaming stats over masked points:
    # ss_res = sum((y - pred)^2)
    # ss_tot = sum(y^2) - sum(y)^2 / n
    n_pts = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    ss_res = 0.0

    for bi, (static_x, phys7_seq, y, m, time_mat) in enumerate(loader):
        if (max_batches is not None) and (bi >= max_batches):
            break

        static_x = static_x.to(device, non_blocking=True)
        phys7_seq = phys7_seq.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        time_mat = time_mat.to(device, non_blocking=True)

        pred = model(static_x, phys7_seq, time_mat)
        loss = masked_loss(pred, y, m, loss_type=loss_type, huber_beta=huber_beta)

        total_loss += float(loss.item())
        nb += 1

        yt = y[m]
        yp = pred[m]
        if yt.numel() > 0:
            dy = (yt - yp)
            ss_res += float(torch.sum(dy * dy).item())
            sum_y += float(torch.sum(yt).item())
            sum_y2 += float(torch.sum(yt * yt).item())
            n_pts += float(yt.numel())

    mean_loss = total_loss / max(1, nb)

    if n_pts <= 1:
        return mean_loss, float("nan")

    ss_tot = sum_y2 - (sum_y * sum_y) / max(1.0, n_pts)
    if ss_tot <= 1e-12:
        return mean_loss, float("nan")

    r2 = 1.0 - ss_res / ss_tot
    return mean_loss, float(r2)

def load_best_common_config_from_tune_verify(runs_root: str):
    """
    读取 runs_root/_tuneV_verify/best_config_common_all_families.json
    返回 best_conf(dict) 或 None

    ✅关键：旧 best_config 可能只有 hp_tag=..._hb0p05，但缺少 hp_huber_beta / hp_loss_type
    这里会从 hp_tag 自动解析并补齐，避免 ablation “看似用了 best”，但 loss 实际没对齐。
    """
    best_json_path = os.path.join(runs_root, "_tuneV_verify", "best_config_common_all_families.json")
    if not os.path.exists(best_json_path):
        log(f"[AUTO-LOAD] best config not found: {best_json_path}")
        return None

    def _parse_hb_from_tag(tag: str):
        # 支持 hb0p05 / hb0.05 / hb0p1 等
        if tag is None:
            return None
        tag = str(tag).strip()
        if not tag:
            return None
        for tok in tag.split("_"):
            if tok.startswith("hb"):
                s = tok[2:]
                s = s.replace("p", ".")
                try:
                    return float(s)
                except Exception:
                    return None
        return None

    try:
        with open(best_json_path, "r", encoding="utf-8") as f:
            best_conf = json.load(f)

        # ---------- backfill: huber_beta / loss_type ----------
        changed = False
        hp_tag = best_conf.get("hp_tag", "")

        # 1) huber_beta：优先用显式字段；没有就从 hp_tag 解析 hb
        if "hp_huber_beta" not in best_conf:
            hb = _parse_hb_from_tag(hp_tag)
            if hb is not None:
                best_conf["hp_huber_beta"] = float(hb)
                changed = True
                log(f"[AUTO-LOAD] Backfill hp_huber_beta={hb} from hp_tag={hp_tag}")

        # 2) loss_type：旧版 hp_tag 不含 loss 名字；只要出现 hb，就默认 huber
        if "hp_loss_type" not in best_conf:
            if "hb" in str(hp_tag):
                best_conf["hp_loss_type"] = "huber"
                changed = True
                log(f"[AUTO-LOAD] Backfill hp_loss_type='huber' (hp_tag contains hb)")

        # 可选：把补齐后的字段写回 json，避免下次仍然缺字段
        if changed:
            try:
                with open(best_json_path, "w", encoding="utf-8") as f:
                    json.dump(best_conf, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                log(f"[AUTO-LOAD] Patched and rewrote best config: {best_json_path}")
            except Exception as e:
                log(f"[AUTO-LOAD][WARN] Failed to rewrite patched best config: {e}")

        log(f"[AUTO-LOAD] Loaded best config: {best_json_path}")
        return best_conf
    except Exception as e:
        log(f"[AUTO-LOAD][WARN] Failed to load best config: {e}")
        return None

def apply_hp_from_best_conf_to_cfg(best_conf: dict) -> int:
    """
    将 best_conf 中的超参覆盖到 Cfg.<key>。

    支持两种 key 形式：
    - `hp_<key>` → Cfg.<key>（旧格式）
    - 直接 `<key>` → Cfg.<key>（新格式，例如 tf_layers/tf_d_model/lr 等）

    说明：
    - 只有当 Cfg 上存在对应属性时才会覆盖（避免拼写错误导致静默污染）
    - 会尽量按旧值类型做强制类型转换（int/float/bool/str）
    """
    if not isinstance(best_conf, dict):
        return 0

    cnt = 0
    for k, v in best_conf.items():
        if str(k).startswith("hp_"):
            cfg_key = str(k)[3:]
        else:
            cfg_key = str(k)
        if not hasattr(Cfg, cfg_key):
            continue

        old_val = getattr(Cfg, cfg_key)
        try:
            if isinstance(old_val, bool):
                if isinstance(v, str):
                    v2 = v.strip().lower() in ("1", "true", "yes", "y", "t")
                else:
                    v2 = bool(v)
            elif isinstance(old_val, int) and not isinstance(old_val, bool):
                v2 = int(v)
            elif isinstance(old_val, float):
                v2 = float(v)
            else:
                v2 = v
        except Exception:
            v2 = v

        setattr(Cfg, cfg_key, v2)
        log(f"[AUTO-LOAD] Override Cfg.{cfg_key}: {old_val} -> {v2}")
        cnt += 1

    log(f"[AUTO-LOAD] Updated {cnt} hyperparameters from best config.")
    return cnt
def get_best_split_seed_from_best_conf(best_conf: dict, default_seed: int = 0) -> int:
    """
    best_config_common_all_families.json 里本身包含 split_seed（因为聚合 key 含 split_seed）。
    """
    try:
        if isinstance(best_conf, dict) and "split_seed" in best_conf:
            return int(best_conf["split_seed"])
    except Exception:
        pass
    return int(default_seed)

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    loss, pack = eval_epoch(model, loader, device=device, return_pack=True, max_batches=None)
    # pack = {"pred_norm","y_norm","mask"}，loss 你要的话也可以塞进去
    pack["loss_norm"] = float(loss)
    return pack

def render_topk_plots(runs_root: str, topk: int = 10, split_name: str = "test"):
    # 目前 StageB 的对比图统一由 render_compare_plots 负责
    render_compare_plots(runs_root)

def run_one_experiment(
        model_type: str, phys_source: str, recipe_aug_mode: str, phys7_mode: str,
        root_out: str, split_seed: int, job_idx: int, job_total: int,
        target_family: str = None,
        shared_df=None, shared_recipe_cols=None, shared_recipe_raw=None,
        shared_targets_full=None, shared_mask_full=None, shared_phys7_seq_cache=None,
        shared_stageA_provider=None, hp_override=None, hp_tag=None
):
    hp_override = hp_override or {}
    fam_tag = target_family if target_family is not None else "multi"

    # -------------------------
    # 0) 先确定“完整训练口径超参”（用于 hp_tag / exp_name / 复现）
    # -------------------------
    lr = float(hp_override.get("lr", Cfg.lr))
    wd = float(hp_override.get("weight_decay", Cfg.weight_decay))
    tf_d_model = int(hp_override.get("tf_d_model", Cfg.tf_d_model))
    tf_layers = int(hp_override.get("tf_layers", Cfg.tf_layers))
    tf_dropout = float(hp_override.get("tf_dropout", Cfg.tf_dropout))
    loss_type = str(hp_override.get("loss_type", getattr(Cfg, "loss_type", "mse")))
    huber_beta = float(hp_override.get("huber_beta", getattr(Cfg, "huber_beta", 0.1)))

    # ✅ hp_tag：优先用外部传入（比如 best_conf["hp_tag"]），否则按真实 hp 自动生成
    if hp_tag is not None and str(hp_tag).strip() != "":
        used_hp_tag = str(hp_tag).strip()
    else:
        used_hp_tag = make_hp_tag(dict(
            lr=lr,
            weight_decay=wd,
            tf_dropout=tf_dropout,
            tf_d_model=tf_d_model,
            tf_layers=tf_layers,
            huber_beta=huber_beta,
        ))

    abbr_mt = {"transformer": "tf", "gru": "gru", "mlp": "mlp"}
    abbr_ps = {"stagea_pred": "stA", "stagea": "stA", "none": "no", "zero": "0", "only_phys": "op"}
    abbr_aug = {"base": "b", "time": "t", "gas": "g", "rf": "r", "squares": "sq", "phys": "ph"}
    abbr_p7 = {"full": "full", "only_energy": "Egy", "only_flux": "Flux", "none": "no"}

    mt = abbr_mt.get(str(model_type).lower(), str(model_type))
    ps = abbr_ps.get(str(phys_source).lower(), str(phys_source))
    au = abbr_aug.get(str(recipe_aug_mode).lower(), str(recipe_aug_mode))
    p7 = abbr_p7.get(str(phys7_mode).lower(), str(phys7_mode))

    # ✅ exp_name 用真实 used_hp_tag，避免“abl 混跑覆盖”和无法对齐 verify
    exp_name = f"{mt}_{used_hp_tag}_{ps}_{au}_{p7}_{fam_tag}_s{int(split_seed)}"

    out_dir = os.path.join(root_out, exp_name)
    _ensure_dir(out_dir)
    ckpt_path = os.path.join(out_dir, "best.pth")

    # -------------------------
    # 1) 固定训练随机种子（模型初始化 + dataloader shuffle）→ 可复现
    #    split_seed 只负责数据划分；训练随机性统一用 Cfg.seed（与 verify 口径一致，abl 可复现 verify）
    #    如需特殊控制，可在 hp_override 里传入 train_seed
    # -------------------------
    train_seed = int(hp_override.get("train_seed", getattr(Cfg, "seed", 42)))
    set_seed(train_seed)

    log(f"\n[JOB {job_idx}/{job_total}] exp={exp_name}")
    log(f"      [SEED] split_seed={int(split_seed)}  train_seed={int(train_seed)}")
    t0 = time.time()

    # --- 数据集：从缓存取 (N,7) raw(full) ---
    phys7_seq_full = None
    if shared_phys7_seq_cache is not None:
        ps_key = str(phys_source).lower().strip()
        phys7_seq_full = shared_phys7_seq_cache.get(ps_key, None)
        if phys7_seq_full is None and ps_key == "only_phys":
            # only_phys 的描述符与 stageA_pred 完全相同，直接复用其缓存
            phys7_seq_full = shared_phys7_seq_cache.get("stagea_pred", None)

        if phys7_seq_full is None:
            log(f"      [P7CACHE] MISS ps_key={ps_key}  (will infer via StageA inside util)")
        else:
            log(f"      [P7CACHE] HIT  ps_key={ps_key}  shape={np.asarray(phys7_seq_full).shape}")

    ds, meta = build_morph_dataset_phys7(
        Cfg.excel_path, Cfg.sheet_name, Cfg.case_id_col,
        target_family=target_family, phys_source=phys_source, recipe_aug_mode=recipe_aug_mode,
        phys7_mode=phys7_mode, df=shared_df, recipe_cols=shared_recipe_cols,
        recipe_raw=shared_recipe_raw, targets_full=shared_targets_full, mask_full=shared_mask_full,
        phys7_seq_full=phys7_seq_full, stageA_provider=shared_stageA_provider,
        stageA_heads_root=getattr(Cfg, "stageA_heads_root", None),
        fit_norm_idx=None,  # 默认：用全量 fit norm（与原脚本/verify 口径一致）
    )

    split = split_dataset_indices(meta["N"], seed=split_seed, train_ratio=Cfg.train_ratio, val_ratio=Cfg.val_ratio)

    device = Cfg.device
    use_vram_cache = (len(ds) < 10000) and (str(device) != "cpu")

    if use_vram_cache:
        tensors = ds.tensors
        gpu_tensors = [t.to(device) for t in tensors]
        ds_gpu = TensorDataset(*gpu_tensors)
        train_ds = Subset(ds_gpu, split["train"])
        val_ds = Subset(ds_gpu, split["val"])
        test_ds = Subset(ds_gpu, split["test"])
        kw_loader = dict(num_workers=0, pin_memory=False, persistent_workers=False)
    else:
        train_ds = Subset(ds, split["train"])
        val_ds = Subset(ds, split["val"])
        test_ds = Subset(ds, split["test"])
        if os.name == 'nt':
            safe_workers = min(Cfg.num_workers, 4) if Cfg.num_workers > 0 else 0
            kw_tr = dict(num_workers=safe_workers, pin_memory=True, persistent_workers=(safe_workers > 0))
        else:
            kw_tr = dict(num_workers=Cfg.num_workers, pin_memory=True, persistent_workers=(Cfg.num_workers > 0))
        kw_loader = dict(**kw_tr)

    train_loader = DataLoader(train_ds, batch_size=Cfg.batch_size, shuffle=True, drop_last=False, **kw_loader)
    val_loader = DataLoader(val_ds, batch_size=Cfg.batch_size_eval, shuffle=False, drop_last=False, **kw_loader)
    test_loader = DataLoader(test_ds, batch_size=Cfg.batch_size_eval, shuffle=False, drop_last=False, **kw_loader)

    Ds = meta.get("Ds", meta.get("static_dim"))
    K = meta.get("K", meta.get("out_dim"))

    if model_type == "transformer":
        model = MorphTransformer(
            static_dim=Ds,
            d_model=tf_d_model,
            nhead=Cfg.tf_nhead,
            num_layers=tf_layers,
            dropout=tf_dropout,
            out_dim=K
        )
    elif model_type == "gru":
        model = MorphGRU(static_dim=Ds, hidden=Cfg.gru_hidden, num_layers=Cfg.gru_layers, out_dim=K)
    elif model_type == "mlp":
        model = MorphMLP(static_dim=Ds, hidden=Cfg.mlp_hidden, num_layers=Cfg.mlp_layers, out_dim=K)
    elif model_type == "lstm":
        model = MorphLSTM(static_dim=Ds, hidden=Cfg.lstm_hidden, num_layers=Cfg.lstm_layers, out_dim=K)
    elif model_type == "tcn":
        model = MorphTCN(static_dim=Ds, hidden=Cfg.tcn_hidden, num_layers=Cfg.tcn_layers,
                         kernel_size=Cfg.tcn_kernel, out_dim=K)
    elif model_type == "ar_mlp":
        model = MorphARMLP(static_dim=Ds, hidden=Cfg.armlp_hidden, num_layers=Cfg.armlp_layers, out_dim=K)
    else:
        raise ValueError(f"Unknown model_type={model_type}")

    model = model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    # --- 训练控制：best 只按 val_r2，early_patience 允许 override ---
    grad_clip = float(hp_override.get("grad_clip", 0.0))
    test_eval_every = int(hp_override.get("test_eval_every", Cfg.test_eval_every))
    early_patience = int(hp_override.get("early_patience", Cfg.early_patience))
    max_epochs = int(hp_override.get("epochs", Cfg.epochs))

    best_val_r2 = -1e18
    best_val_loss_at_best = float("inf")
    best_ep = -1
    bad_epochs = 0
    train_losses, val_losses = [], []
    test_losses_by_epoch = {}

    for ep in range(1, max_epochs + 1):
        tr = train_epoch(model, train_loader, opt, device, loss_type, huber_beta, grad_clip)
        va_loss, va_r2 = eval_epoch_stats(model, val_loader, device, loss_type, huber_beta)

        train_losses.append(float(tr))
        val_losses.append(float(va_loss))

        if test_eval_every > 0 and (ep % test_eval_every == 0):
            te_loss, te_r2 = eval_epoch_stats(model, test_loader, device, loss_type, huber_beta)
            test_losses_by_epoch[int(ep)] = float(te_loss)

        # ✅ best 只按 R2
        improved = (float(va_r2) > best_val_r2 + 1e-12)

        if improved:
            best_val_r2 = float(va_r2)
            best_val_loss_at_best = float(va_loss)
            best_ep = int(ep)
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)
        else:
            bad_epochs += 1

        if (Cfg.log_every > 0) and (ep % Cfg.log_every == 0):
            log(f"      [E{ep:03d}] tr={tr:.5f} | va_loss={va_loss:.5f} | va_r2={va_r2:.4f} | best_r2={best_val_r2:.4f} @E{best_ep}")

        if (early_patience > 0) and (bad_epochs >= early_patience):
            log(f"      [STOP] EarlyStop by val_r2: no improve for {early_patience} eps")
            break

    if os.path.exists(ckpt_path):
        ck = _torch_load_ckpt_trusted(ckpt_path)
        model.load_state_dict(ck["model"])
        meta = ck.get("meta", meta)

    pack = evaluate_model(model, test_loader, device=device)
    met = export_experiment(out_dir, pack, meta, exp_name, "test", make_plots=False, clip_nonneg=True)

    try:
        export_loss_curve(out_dir, exp_name, train_losses, val_losses, test_losses_by_epoch=test_losses_by_epoch)
    except Exception:
        pass

    pf = met.get("per_family_r2", {})
    min_pf_r2 = float(np.nanmin(list(pf.values()))) if isinstance(pf, dict) and len(pf) > 0 else float("nan")
    dt = time.time() - t0

    row = {
        "exp_name": exp_name,
        "hp_tag": used_hp_tag,  # ✅ 与 exp_name 一致

        "model_type": model_type,
        "phys_source": phys_source,
        "recipe_aug_mode": recipe_aug_mode,
        "phys7_mode": phys7_mode,
        "split_seed": int(split_seed),
        "train_seed": int(train_seed),  # ✅ 可复现关键

        "best_metric_name": "val_r2",
        "best_metric_value": float(best_val_r2),
        "best_epoch": int(best_ep),
        "best_val": float(best_val_r2),
        "best_val_r2": float(best_val_r2),
        "best_val_loss_at_best": float(best_val_loss_at_best),

        "test_loss_norm": float(pack.get("loss_norm", float("nan"))),
        "overall_mae_nm": float(met.get("overall_mae_nm", float("nan"))),
        "overall_r2": float(met.get("overall_r2", float("nan"))),
        "min_pf_r2": float(min_pf_r2),

        "time_min": float(dt / 60.0),
        "out_dir": out_dir,
        "ckpt_path": ckpt_path,
        "family_mode": fam_tag,

        "hp_epochs": int(max_epochs),
        "hp_lr": float(lr),
        "hp_weight_decay": float(wd),
        "hp_early_patience": int(early_patience),
        "hp_tf_d_model": int(tf_d_model),
        "hp_tf_layers": int(tf_layers),
        "hp_tf_dropout": float(tf_dropout),
        "hp_select_best_by": "val_r2",

        # ✅ 关键：把 loss/huber 写入 hp_，让 verify 生成的 best json 能完整表达训练口径
        "hp_loss_type": str(loss_type),
        "hp_huber_beta": float(huber_beta),
    }

    # 允许 hp_override 透传额外 hp_* 字段（但不覆盖已有键）
    for k, v in hp_override.items():
        if str(k).startswith("hp_") and k not in row:
            row[k] = v

    return row


def run_job_list(jobs, runs_root, seeds=None, **kwargs):
    if seeds is None:
        seeds = list(getattr(Cfg, "split_seeds", (0, 1, 2)))

    out_root = runs_root
    _ensure_dir(out_root)

    # ✅ 避免 hp_tag / hp_override 在显式参数和 **kwargs 中重复传入（否则会 TypeError: multiple values）
    call_kwargs = dict(kwargs)
    call_kwargs.pop("hp_tag", None)
    call_kwargs.pop("hp_override", None)

    total = len(jobs) * len(seeds) * len(FAMILIES)
    cnt = 0
    summary_fieldnames = None

    for j in jobs:
        for seed in seeds:
            for fam in FAMILIES:
                cnt += 1
                row = run_one_experiment(
                    j.get("model_type", "transformer"),
                    j.get("phys_source", "stageA_pred"),
                    j.get("recipe_aug_mode", "time"),
                    j.get("phys7_mode", "full"),
                    out_root,
                    seed,
                    cnt,
                    total,
                    fam,

                    # ✅ 允许 job 自带 override；否则用 run_job_list 的全局 kwargs
                    hp_override=j.get("hp_override", kwargs.get("hp_override", None)),
                    # ✅ hp_tag 优先 job 自带；否则用 run_job_list 的全局 kwargs
                    hp_tag=j.get("hp_tag", kwargs.get("hp_tag", None)),

                    **call_kwargs
                )
                if summary_fieldnames is None:
                    summary_fieldnames = list(row.keys())

                append_summary_row(row, out_root, fieldnames=summary_fieldnames)

    # ✅ 聚合+画图
    postprocess_summary_from_csv(out_root, write_excel=True)

def append_summary_row(row: Dict, root_out: str, fieldnames: List[str]):
    """
    只负责“追加一行”到 results_summary.csv
    """
    _ensure_dir(root_out)
    csv_path = os.path.join(root_out, "results_summary.csv")
    is_new = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            w.writeheader()
        fixed = {k: row.get(k, "") for k in fieldnames}
        w.writerow(fixed)

def postprocess_summary_from_csv(root_out: str, write_excel: bool = True):
    """
    从 results_summary.csv 读回全部 rows，交给 save_summary 聚合产物，
    并强制输出 compare_figs（你的要求：comparefig 必画）。
    """
    csv_path = os.path.join(root_out, "results_summary.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")

    # 1) 聚合：best_common / agg / leaderboard 等
    save_summary(rows, root_out, write_excel=write_excel)

    # 2) 强制画 compare figs
    try:
        render_compare_plots(root_out)
    except Exception as e:
        log(f"[POST][WARN] render_compare_plots failed: {e}")

def _load_eval_pack_npz(npz_path: str) -> Tuple[Dict, Dict]:
    z = np.load(npz_path, allow_pickle=False)
    pack = {k: z[k] for k in z.files if k not in ["meta"]}
    meta = {}
    if "meta" in z.files:
        meta_str = z["meta"]
        if isinstance(meta_str, np.ndarray):
            meta_str = meta_str.item()
        if isinstance(meta_str, (bytes, bytearray)):
            meta_str = meta_str.decode("utf-8", errors="ignore")
        if isinstance(meta_str, str) and meta_str.strip():
            try:
                meta = json.loads(meta_str)
            except Exception:
                meta = {}
    return pack, meta

def save_summary(rows: List[Dict], root_out: str, write_excel: bool = False):
    """
    ✅ StageB summary 唯一入口（请保证全文件只保留这一份）

    输出：
    - results_summary.csv（去重后的全量明细）
    - leaderboard_top20.csv（按 min_pf_r2 排序的明细榜单）
    - results_grouped.csv / xlsx（legacy：按结构取 best）
    - results_config_seed_agg.csv（config+hp+seed 粒度，跨 family 聚合）
    - best_config_common_all_families.json（complete==1 的 best common）
    - leaderboard_config_top20.csv（config+hp+seed 榜单）
    """
    _ensure_dir(root_out)
    if not rows:
        return

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # -------------------------
    # 1) 合并旧 summary（防止覆盖）
    # -------------------------
    csv_path = os.path.join(root_out, "results_summary.csv")
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            df = pd.concat([df_old, df], ignore_index=True)
        except Exception:
            pass

    # -------------------------
    # 2) 基础数值列 & 去重
    # -------------------------
    df["min_pf_r2"] = pd.to_numeric(df.get("min_pf_r2"), errors="coerce")
    df["overall_r2"] = pd.to_numeric(df.get("overall_r2"), errors="coerce")
    df["split_seed"] = pd.to_numeric(df.get("split_seed"), errors="coerce")

    # exp_name + split_seed 可能重复（重复跑/append 多次），保留 min_pf_r2 最大的一条
    if "exp_name" in df.columns and "split_seed" in df.columns:
        df = df.sort_values("min_pf_r2", ascending=False)
        df = df.drop_duplicates(subset=["exp_name", "split_seed"], keep="first")

    # 写回全量 summary
    df.to_csv(csv_path, index=False)

    # -------------------------
    # 3) 明细榜单（单次 run 的 top20）
    # -------------------------
    lb_top20_path = os.path.join(root_out, "leaderboard_top20.csv")
    df.sort_values("min_pf_r2", ascending=False).head(20).to_csv(lb_top20_path, index=False)

    # best single row（便于你日志里那句 best single-row）
    best_single = None
    if "min_pf_r2" in df.columns and not df["min_pf_r2"].isna().all():
        best_single = df.sort_values("min_pf_r2", ascending=False).iloc[0].to_dict()

    # -------------------------
    # 4) legacy grouped（保留你之前的 grouped 输出）
    # -------------------------
    base_gcols = ["model_type", "phys_source", "recipe_aug_mode", "phys7_mode"]
    gcols = [c for c in base_gcols if c in df.columns]
    if gcols:
        df_legacy = df.sort_values("min_pf_r2", ascending=False).groupby(gcols, as_index=False).head(1)
        df_legacy.to_csv(os.path.join(root_out, "results_grouped.csv"), index=False)
        if write_excel:
            try:
                df_legacy.to_excel(os.path.join(root_out, "results_grouped.xlsx"), index=False)
            except Exception:
                pass

    # -------------------------
    # 5) config+hp+seed 粒度聚合（跨 family）
    #    关键：一定要把 hp_tag + hp_* 放进 key，否则会把不同超参“混”在一起
    # -------------------------
    fam_col = "family_mode"
    required_fams = list(FAMILIES)

    # 动态抓 hp_ 列，顺序要稳定（不要 set()）
    hp_cols = sorted([c for c in df.columns if c.startswith("hp_")])
    idx_cols = []
    for c in (base_gcols + ["hp_tag"] + hp_cols + ["split_seed"]):
        if c in df.columns and c not in idx_cols:
            idx_cols.append(c)

    agg_path = os.path.join(root_out, "results_config_seed_agg.csv")
    best_common_path = os.path.join(root_out, "best_config_common_all_families.json")
    lb_config_path = os.path.join(root_out, "leaderboard_config_top20.csv")

    df_single = df.copy()
    if fam_col in df_single.columns:
        df_single = df_single[df_single[fam_col].isin(required_fams)].copy()
    else:
        df_single = df_single.iloc[0:0].copy()  # 没有 family_mode 无法聚合

    if not df_single.empty and idx_cols:
        pv = df_single.pivot_table(
            index=idx_cols,
            columns=fam_col,
            values="min_pf_r2",
            aggfunc="max"  # 同 key 多条时取更好那条
        )

        # 补全列
        for f in required_fams:
            if f not in pv.columns:
                pv[f] = np.nan
        pv = pv[required_fams]

        complete = pv.notna().all(axis=1)
        min_all = pv.min(axis=1)
        mean_all = pv.mean(axis=1)

        agg_df = pv.copy()
        agg_df["complete_all_families"] = complete.astype(int)
        agg_df["min_r2_all_families"] = min_all
        agg_df["mean_r2_all_families"] = mean_all
        agg_df = agg_df.reset_index()
        agg_df.to_csv(agg_path, index=False)

        # best_common：只在 complete==1 里选 min 最大；tie-break mean
        cand = agg_df[agg_df["complete_all_families"] == 1].copy()
        best_common = None
        if not cand.empty:
            cand = cand.sort_values(["min_r2_all_families", "mean_r2_all_families"], ascending=False)
            best_common = cand.iloc[0].to_dict()
            with open(best_common_path, "w", encoding="utf-8") as f:
                json.dump(best_common, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        # config+seed 榜单 top20
        agg_df.sort_values(
            ["complete_all_families", "min_r2_all_families", "mean_r2_all_families"],
            ascending=[False, False, False]
        ).head(20).to_csv(lb_config_path, index=False)
    else:
        # 仍然输出空文件/提示（可选）
        pass

    # -------------------------
    # 6) 统一打印（对应你期待看到的那几行）
    # -------------------------
    log(f"[SUMMARY] saved: {csv_path}")
    if best_single is not None:
        tag = str(best_single.get("hp_tag", ""))
        log(f"[SUMMARY] best single-row: {tag}  min_pf_r2={best_single.get('min_pf_r2')}  overall_R2={best_single.get('overall_r2')}")
    log(f"[SUMMARY] aggregated all-families: {agg_path}")
    log(f"[SUMMARY] best common(all families): {best_common_path}")
    log(f"[SUMMARY] leaderboard configs: {lb_config_path}")

def render_compare_plots(root_out: str):
    agg_path = os.path.join(root_out, "results_config_seed_agg.csv")
    if not os.path.exists(agg_path):
        log(f"[COMPARE] skip: {agg_path} not found")
        return

    out_fig_dir = os.path.join(root_out, "compare_figs")
    _ensure_dir(out_fig_dir)

    agg = pd.read_csv(agg_path)
    if agg.empty:
        log("[COMPARE] skip: agg empty")
        return

    # fig1: top configs by min_r2_all_families
    try:
        tmp = agg.copy()
        grp = tmp.groupby(["model_type", "phys_source", "recipe_aug_mode", "phys7_mode"], as_index=False).agg(
            min_r2=("min_r2_all_families", "mean"),
            mean_r2=("mean_r2_all_families", "mean"),
            complete=("complete_all_families", "max"),
        )
        grp["config_id"] = grp.apply(
            lambda x: f"{x['model_type']}|{x['phys_source']}|{x['recipe_aug_mode']}|{x['phys7_mode']}", axis=1)
        grp = grp.sort_values(["complete", "min_r2", "mean_r2"], ascending=[False, False, False]).head(15)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.barh(grp["config_id"].tolist()[::-1], grp["min_r2"].tolist()[::-1])
        ax.set_xlabel("min R2 across families (avg over seeds)")
        ax.set_title("Top configs by min_R2_all_families")
        fig.tight_layout()
        fig.savefig(os.path.join(out_fig_dir, "compare_top_configs_minR2.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        log(f"[COMPARE] fig1 fail: {e}")

    # fig2: best_common per-family R2（注意这里用 zmin/h0/...）
    try:
        best_path = os.path.join(root_out, "best_config_common_all_families.json")
        if os.path.exists(best_path):
            best = json.load(open(best_path, "r", encoding="utf-8"))
            vals = [float(best.get(fam, np.nan)) for fam in FAMILIES]

            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.bar(FAMILIES, vals)
            ax.set_ylim(-0.2, 1.0)
            ax.set_title("Best common config: per-family R2")
            fig.tight_layout()
            fig.savefig(os.path.join(out_fig_dir, "compare_best_common_per_family_r2.png"), dpi=150)
            plt.close(fig)
    except Exception as e:
        log(f"[COMPARE] fig2 fail: {e}")

    # fig3: baseline vs stageA_pred（同样用 fam 列名）
    try:
        base = agg[agg["phys_source"].astype(str).str.lower().isin(["none", "zero"])].copy()
        stag = agg[agg["phys_source"].astype(str).str.lower().isin(["stagea_pred", "stagea", "stagea_ensemble"])].copy()

        if (not base.empty) and (not stag.empty):
            def pick_best(df):
                df = df[df["complete_all_families"] == 1].copy()
                if df.empty:
                    return None
                df = df.sort_values(["min_r2_all_families", "mean_r2_all_families"], ascending=False)
                return df.iloc[0].to_dict()

            b = pick_best(base)
            s = pick_best(stag)
            if b and s:
                bv = [float(b.get(fam, np.nan)) for fam in FAMILIES]
                sv = [float(s.get(fam, np.nan)) for fam in FAMILIES]

                x = np.arange(len(FAMILIES))
                w = 0.35
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(111)
                ax.bar(x - w / 2, bv, width=w, label="baseline (none/zero)")
                ax.bar(x + w / 2, sv, width=w, label="stageA_pred")
                ax.set_xticks(x)
                ax.set_xticklabels(FAMILIES)
                ax.set_ylim(-0.2, 1.0)
                ax.legend()
                ax.set_title("Best common: baseline vs stageA_pred (per-family R2)")
                fig.tight_layout()
                fig.savefig(os.path.join(out_fig_dir, "compare_baseline_vs_stageA_per_family_r2.png"), dpi=150)
                plt.close(fig)
    except Exception as e:
        log(f"[COMPARE] fig3 fail: {e}")


def quick_self_test(n: int = 16):
    log("\n[SELF-TEST] ===== start =====")

    # 0) 设备检查
    log(f"[SELF-TEST] torch={torch.__version__}")
    dev = get_default_device()
    log(f"[SELF-TEST] device={dev}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        log(f"[SELF-TEST] gpu={name}, capability={cap}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log("[SELF-TEST] Using Apple Silicon MPS backend")
    else:
        log("[SELF-TEST] Using CPU backend")

    # 1) 读表 + recipe 列检测
    df = pd.read_excel(Cfg.excel_path, sheet_name=Cfg.sheet_name)
    cols = df.columns.tolist()
    recipe_cols = _detect_recipe_cols(cols)
    recipe_raw = df[recipe_cols].values.astype(np.float32)[:n]

    # 2) StageA 推理 Phys7（顺带验证 recipe 对齐逻辑）
    if "stagea_pred" in [s.lower() for s in Cfg.phys_sources]:
        p7 = infer_phys7_from_stageA_ckpt(Cfg.stageA_heads_root, recipe_raw, case_recipe_cols=recipe_cols)  # (n,7)
        if not (isinstance(p7, np.ndarray) and p7.shape[0] == recipe_raw.shape[0] and p7.shape[1] >= 7):
            raise RuntimeError(f"StageA infer output bad shape: {None if p7 is None else getattr(p7, 'shape', None)}")
    else:
        p7 = np.zeros((recipe_raw.shape[0], 7), np.float32)

    phys7_seq = np.repeat(p7[:, :, None], len(TIME_LIST), axis=2)  # (n,7,T)

    # 3) recipe aug + static zscore
    aug_mode = Cfg.recipe_aug_modes[0]
    static_x = augment_recipe_features(recipe_raw, aug_mode)
    x_mean, x_std = _zscore_fit(static_x)
    static_xn = _zscore_apply(static_x, x_mean, x_std)

    # 4) phys7 mode（只测一个）
    pmode = Cfg.phys7_modes[0]
    N, Dp, T = phys7_seq.shape
    phys7_flat = phys7_seq.transpose(0, 2, 1).reshape(-1, Dp)
    phys7_flat = apply_phys7_mode(phys7_flat, pmode)
    phys7_seq = phys7_flat.reshape(N, T, Dp).transpose(0, 2, 1)

    # 5) targets：只拿一个 family 做单任务自检（避免 K=6 拉一堆不必要的检查）
    fam = (Cfg.families_to_train[0] if getattr(Cfg, "families_to_train", None) else FAMILIES[0])
    targets = np.zeros((N, 1, len(TIME_LIST)), np.float32)
    mask = np.zeros((N, 1, len(TIME_LIST)), bool)

    eps = 1e-12
    for j, t in enumerate(TIME_LIST):
        cn = _detect_target_col(cols, fam, t)
        if cn and cn in df.columns:
            vals = pd.to_numeric(df[cn], errors="coerce").to_numpy(dtype=np.float32)[:N]
            valid = np.isfinite(vals) & (np.abs(vals) > eps)
            targets[valid, 0, j] = vals[valid]
            mask[valid, 0, j] = True

    cov = _mask_coverage(mask)
    log(f"[SELF-TEST] family={fam}, mask_coverage={cov * 100:.2f}%")
    if cov < 0.01:
        raise RuntimeError(f"Mask coverage too low for family={fam}. 可能列名检测不对或数据缺失。")

    # 6) target zscore（只对 mask 部分统计）
    y_flat = targets[mask]
    y_mean_val = float(np.mean(y_flat)) if y_flat.size else 0.0
    y_std_val = float(np.std(y_flat)) + 1e-6 if y_flat.size else 1.0
    targets_n = (targets - y_mean_val) / y_std_val

    # 7) 转 torch + 跑一次 forward/backward
    device = Cfg.device
    static_x_t = torch.from_numpy(static_xn).to(device)
    phys7_seq_t = torch.from_numpy(phys7_seq).to(device)
    y_t = torch.from_numpy(targets_n).to(device)
    m_t = torch.from_numpy(mask).to(device)
    time_mat_t = torch.from_numpy(np.tile(TIME_VALUES[None, :], (N, 1)).astype(np.float32)).to(device)

    model = MorphTransformer(
        static_dim=static_xn.shape[1],
        d_model=Cfg.tf_d_model, nhead=Cfg.tf_nhead,
        num_layers=Cfg.tf_layers, dropout=Cfg.tf_dropout,
        out_dim=1
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    model.train()
    pred = model(static_x_t, phys7_seq_t, time_mat_t)
    loss = masked_mse(pred, y_t, m_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

    log(f"[SELF-TEST] one-step train ok. loss={float(loss.item()):.6f}")
    log("[SELF-TEST] ===== passed =====\n")


def prepare_shared_cache():
    df = pd.read_excel(Cfg.excel_path, sheet_name=Cfg.sheet_name)
    before_N = len(df)  # 清洗前行数

    bad_row = None  # (before_N,) bool
    kept_idx = np.arange(before_N)  # 清洗后保留的原始行索引

    # case_ids（可选）
    case_ids = None
    if getattr(Cfg, "case_id_col", None) and (Cfg.case_id_col in df.columns):
        case_ids = df[Cfg.case_id_col].astype(str).map(_norm_case_id).values

    # ------------------ 1) recipe ------------------
    recipe_cols = _detect_recipe_cols(df.columns.tolist())
    recipe_raw = df[recipe_cols].values.astype(np.float32)

    # ------------------ 2) targets & mask (N,6,T) ------------------
    N0 = len(df)
    targets_full = np.zeros((N0, len(FAMILIES), len(TIME_LIST)), dtype=np.float32)
    mask_full = np.zeros((N0, len(FAMILIES), len(TIME_LIST)), dtype=bool)

    cols_list = df.columns.tolist()
    for i, fam in enumerate(FAMILIES):
        for j, t in enumerate(TIME_LIST):
            col_name = _detect_target_col(cols_list, fam, t)
            if col_name and col_name in df.columns:
                vals = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=np.float32)
                # ✅ 只要是数就算有效（0 也有效）；缺失是 NaN
                valid = np.isfinite(vals)
                targets_full[valid, i, j] = vals[valid]
                mask_full[valid, i, j] = True

    # ------------------ 3) 【A】行级剔除：任何 family/time 的有效点超界 -> 整行 bad ------------------
    if hasattr(Cfg, "target_bounds_um") and Cfg.target_bounds_um:
        bad_row = detect_bad_rows_by_bounds(
            targets_full=targets_full,
            mask_full=mask_full,
            families=list(FAMILIES),
            bounds_um=Cfg.target_bounds_um,
            zmin_use_abs=bool(getattr(Cfg, "zmin_use_abs_for_bounds", True)),
        )
        kept_idx = np.where(~bad_row)[0].astype(np.int64)

        if getattr(Cfg, "print_clean_stats", True):
            log(f"[CLEAN] rows_total={N0}  bad_rows={int(bad_row.sum())}  kept={int(kept_idx.size)}")

        # ✅ 超限样本彻底不参与后续任何步骤（df/recipe/targets/mask/case_ids 全切片）
        df = df.iloc[kept_idx].reset_index(drop=True)
        recipe_raw = recipe_raw[kept_idx]
        targets_full = targets_full[kept_idx]
        mask_full = mask_full[kept_idx]
        if case_ids is not None:
            case_ids = case_ids[kept_idx]

    # ------------------ 4) 【B】小负值 clip 为 0（保留监督） ------------------
    clipped_stat = {}
    if getattr(Cfg, "print_clean_stats", True):
        neg_before = count_negative_points(
            targets_full=targets_full,
            mask_full=mask_full,
            families=list(FAMILIES),
            exclude_fams=["zmin"]
        )
        log(f"[CLEAN] negative points before clip (exclude zmin): {neg_before}")

    if getattr(Cfg, "clip_small_neg_to_zero", True):
        tol = float(getattr(Cfg, "small_neg_tol_um", 0.02))
        clipped_stat = clip_small_negative_to_zero(
            targets_full=targets_full,
            mask_full=mask_full,
            families=list(FAMILIES),
            neg_tol_um=tol,
            exclude_fams=["zmin"],
        )
        if getattr(Cfg, "print_clean_stats", True):
            log(f"[CLEAN] clip_small_negative_to_zero tol={tol}um  clipped_points={clipped_stat}")

    if getattr(Cfg, "print_clean_stats", True):
        neg_after = count_negative_points(
            targets_full=targets_full,
            mask_full=mask_full,
            families=list(FAMILIES),
            exclude_fams=["zmin"]
        )
        log(f"[CLEAN] negative points after clip (exclude zmin): {neg_after}")

    # ------------------ 5) ✅ 缺失/空列报告：一定要放在 targets_full/mask_full 构造 + 清洗之后 ------------------
    if getattr(Cfg, "print_clean_stats", True):
        print_missingness_report(
            df=df,  # ✅ 这里是清洗后的 df（用于训练）
            targets_full=targets_full,
            mask_full=mask_full,
            families=list(FAMILIES),
            time_list=list(TIME_LIST),
            title="[MISSINGNESS REPORT] after cleaning (used for training)",
            unit="um",
            recipe_cols=recipe_cols,
            show_examples=3,
        )

    # ------------------ 6) StageA provider（init once） ------------------
    provider = None
    need_stageA = any(
        str(ps).lower().strip() in ["stagea_pred", "stagea", "stagea_ensemble", "only_phys"] for ps in Cfg.phys_sources)
    if need_stageA:
        provider = StageAEnsemblePhys7Provider(
            heads_root=Cfg.stageA_heads_root,
            device="cpu",  # ✅ StageA 推理在 MPS 上数值不稳定，固定用 CPU
            recipe_cols_in=recipe_cols,
            expect_k=7,
        )

    # ------------------ 7) phys7_raw_full_cache：缓存 (N,7) 的 raw(full) ------------------
    phys7_raw_full_cache = {}
    phys_sources_unique = sorted(set([str(ps).lower().strip() for ps in Cfg.phys_sources]))

    N = recipe_raw.shape[0]
    for ps in phys_sources_unique:
        if ps in ["none", "zero"]:
            phys7_raw_full_cache[ps] = np.zeros((N, 7), dtype=np.float32)
        elif ps in ["stagea_pred", "stagea", "stagea_ensemble", "only_phys"]:
            # only_phys 与 stageA_pred 共用同一份描述符（provider 内部有 infer 缓存）
            if provider is None:
                raise RuntimeError("need stageA provider but provider is None")
            p7_full = provider.infer(recipe_raw, phys7_mode="full", use_cache=True).astype(np.float32)  # (N,7)
            phys7_raw_full_cache[ps] = p7_full
        else:
            raise ValueError(f"Unknown phys_source in Cfg.phys_sources: {ps}")

    return (df, recipe_cols, recipe_raw, provider,
            targets_full, mask_full, phys7_raw_full_cache, case_ids,
            bad_row, before_N, kept_idx, clipped_stat)


def build_job_list_phase1() -> List[Dict[str, str]]:
    """
    Phase1：只跑主配置（你指定的 baseline）
    目的：找到 best split_seed（用 save_summary 里 best_config_common_all_families.json）
    """
    base = dict(
        model_type=getattr(Cfg, "baseline_model_type", "transformer"),
        phys_source=getattr(Cfg, "baseline_phys_source", "stageA_pred"),
        recipe_aug_mode=getattr(Cfg, "baseline_recipe_aug_mode", "time"),
        phys7_mode=getattr(Cfg, "baseline_phys7_mode", "full"),
    )
    return [base]


def build_job_list_phase2() -> List[Dict[str, str]]:
    return build_job_list_ablationA()

def autoload_best_hp_from_tune_verify(runs_root: str) -> bool:
    """
    从 runs_root/_tuneV_verify/best_config_common_all_families.json 读取 hp_*，
    覆盖到 Cfg 中对应字段（hp_lr -> Cfg.lr 等）。

    返回：是否成功加载并覆盖至少 1 个超参
    """
    best_json_path = os.path.join(runs_root, "_tuneV_verify", "best_config_common_all_families.json")
    if not os.path.exists(best_json_path):
        log(f"[AUTO-LOAD] No best config found: {best_json_path}")
        return False

    try:
        with open(best_json_path, "r", encoding="utf-8") as f:
            best_conf = json.load(f)

        log(f"\n[AUTO-LOAD] Found best config from: {best_json_path}")
        cnt = 0
        for k, v in best_conf.items():
            if not str(k).startswith("hp_"):
                continue
            cfg_key = str(k)[3:]
            if not hasattr(Cfg, cfg_key):
                continue

            old_val = getattr(Cfg, cfg_key)
            # 类型转换对齐
            try:
                if isinstance(old_val, int):
                    v2 = int(v)
                elif isinstance(old_val, float):
                    v2 = float(v)
                else:
                    v2 = v
            except Exception:
                v2 = v

            setattr(Cfg, cfg_key, v2)
            log(f"  > Override Cfg.{cfg_key}: {old_val} -> {v2}")
            cnt += 1

        log(f"[AUTO-LOAD] Successfully updated {cnt} hyperparameters!\n")
        return cnt > 0
    except Exception as e:
        log(f"[AUTO-LOAD][WARN] Failed to load best config: {e}")
        return False

def build_tune_candidates(C_screen: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    cand: List[Dict[str, Any]] = []

    # baseline（以当前Cfg为中心）
    base = dict(
        lr=float(Cfg.lr),
        weight_decay=float(Cfg.weight_decay),
        tf_dropout=float(Cfg.tf_dropout),
        tf_d_model=int(Cfg.tf_d_model),
        tf_layers=int(Cfg.tf_layers),
        loss_type="huber",
        huber_beta=0.08,
        select_best_by="val_r2",
        grad_clip=1.0,
    )
    cand.append(base)

    # 一些结构化点（少量，提升稳定性）
    structured = []
    for lr in [1e-4, 2e-4, 3e-4, 4e-4]:
        for do in [0.05, 0.10, 0.15]:
            structured.append(dict(base, lr=float(lr), tf_dropout=float(do)))
    for dm in [256, 384, 512]:
        structured.append(dict(base, tf_d_model=int(dm)))
    for L in [4, 6]:
        structured.append(dict(base, tf_layers=int(L), tf_dropout=0.15))
    for hb in [0.05, 0.08, 0.12]:
        structured.append(dict(base, huber_beta=float(hb)))

    # 去重加入
    seen = set()

    def key(h):
        return (round(h["lr"], 12), round(h["weight_decay"], 12), round(h["tf_dropout"], 6),
                int(h["tf_d_model"]), int(h["tf_layers"]), round(h["huber_beta"], 6))

    for h in structured:
        if key(h) not in seen:
            cand.append(h)
            seen.add(key(h))
        if len(cand) >= C_screen:
            break

    # 随机补齐
    while len(cand) < C_screen:
        lr = float(10 ** rng.uniform(np.log10(8e-5), np.log10(6e-4)))
        wd = float(10 ** rng.uniform(np.log10(5e-6), np.log10(3e-4)))
        do = float(rng.uniform(0.03, 0.22))
        dm = int(rng.choice([256, 384, 512]))
        L = int(rng.choice([4, 6]))
        hb = float(rng.choice([0.05, 0.08, 0.12]))
        h = dict(base, lr=lr, weight_decay=wd, tf_dropout=do, tf_d_model=dm, tf_layers=L, huber_beta=hb)

        if key(h) in seen:
            continue
        seen.add(key(h))
        cand.append(h)

    return cand[:C_screen]


def tune_minper_auto(
        runs_root: str,
        df_cache, recipe_cols_cache, recipe_raw_cache,
        stageA_provider_cache, targets_full, mask_full,
        phys7_seq_cache, case_ids_cache
):
    import shutil

    # 你要的 baseline 固定配置（不变）
    model_type = "transformer"
    phys_source = "stageA_pred"
    recipe_aug_mode = "time"
    phys7_mode = "full"

    # ---------- 可调参数（硬编码，满足 runs<1000） ----------
    C_screen = 60
    topK = 10

    screen_fams = ["d1", "h1"]
    verify_fams = list(FAMILIES)

    screen_seeds = [0]
    verify_seeds = [0, 1, 2]  # 如果你更稳：改成 [0,1,2,3,4]，仍<1000

    # screen/verify 训练日程（screen更快）
    hp_screen_extra = dict(epochs=300, early_patience=30, test_eval_every=5)
    hp_verify_extra = dict(epochs=300, early_patience=30, test_eval_every=5)

    screen_root = os.path.join(runs_root, "_tuneS_screen")
    verify_root = os.path.join(runs_root, "_tuneV_verify")
    _ensure_dir(screen_root)
    _ensure_dir(verify_root)

    # 清理旧的summary（避免混）
    for r in [screen_root, verify_root]:
        p = os.path.join(r, "results_summary.csv")
        if os.path.exists(p):
            os.remove(p)

    cand = build_tune_candidates(C_screen=C_screen, seed=Cfg.seed + 123)
    cand_json = os.path.join(screen_root, "candidates.json")
    with open(cand_json, "w", encoding="utf-8") as f:
        json.dump([dict(h, hp_tag=make_hp_tag(h)) for h in cand], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # ---------------------------
    # Stage S: screen (d1/h1)
    # ---------------------------
    log("=" * 60)
    log(f"[TUNE] Stage-S screen: C={C_screen}, fams={screen_fams}, seeds={screen_seeds}")
    log("=" * 60)

    rows = []
    job_total = len(cand) * len(screen_fams) * len(screen_seeds)
    job_idx = 0
    summary_fieldnames = None

    for h in cand:
        tag = make_hp_tag(h)
        hp = dict(h)
        hp.update(hp_screen_extra)

        for fam in screen_fams:
            for sd in screen_seeds:
                job_idx += 1
                log(f"\n[SCREEN {job_idx}/{job_total}] hp={tag} fam={fam} seed={sd}")
                r = run_one_experiment(
                    model_type=model_type,
                    phys_source=phys_source,
                    recipe_aug_mode=recipe_aug_mode,
                    phys7_mode=phys7_mode,
                    root_out=screen_root,
                    split_seed=sd,
                    job_idx=job_idx,
                    job_total=job_total,
                    target_family=fam,
                    shared_df=df_cache,
                    shared_recipe_cols=recipe_cols_cache,
                    shared_recipe_raw=recipe_raw_cache,
                    shared_targets_full=targets_full,
                    shared_mask_full=mask_full,
                    shared_phys7_seq_cache=phys7_seq_cache,
                    shared_stageA_provider=stageA_provider_cache,
                    hp_override=hp,
                    hp_tag=tag,
                )

                rows.append(r)
                if summary_fieldnames is None:
                    summary_fieldnames = list(r.keys())
                append_summary_row(r, screen_root, fieldnames=summary_fieldnames)

    postprocess_summary_from_csv(screen_root, write_excel=True)

    # 从 screen 的 results_summary.csv 里选 topK（按 min(d1,h1)）
    dfS = pd.read_csv(os.path.join(screen_root, "results_summary.csv"))
    dfS = dfS[dfS["family_mode"].isin(screen_fams)]
    score = dfS.groupby(["hp_tag", "split_seed"])["min_pf_r2"].min().reset_index()
    score = score.sort_values(by="min_pf_r2", ascending=False)

    top = score.head(topK)
    top_tags = list(top["hp_tag"].values)

    top_json = os.path.join(screen_root, "topK.json")
    with open(top_json, "w", encoding="utf-8") as f:
        json.dump({"topK": top_tags, "table": top.to_dict(orient="records")}, f, indent=2, ensure_ascii=False,
                  cls=NumpyEncoder)

    log("=" * 60)
    log(f"[TUNE] Stage-S done. Selected topK={len(top_tags)}")
    log(top.to_string(index=False))
    log("=" * 60)

    # ---------------------------
    # Stage V: verify (6 families + multi seeds)
    # ---------------------------
    # 找回 topK 对应超参 dict
    tag2hp = {make_hp_tag(h): h for h in cand}
    verify_hp_list = []
    for tag in top_tags:
        if tag in tag2hp:
            verify_hp_list.append((tag, dict(tag2hp[tag])))
        else:
            log(f"[WARN] tag not found in candidates: {tag}")

    log("=" * 60)
    log(f"[TUNE] Stage-V verify: K={len(verify_hp_list)}, fams={verify_fams}, seeds={verify_seeds}")
    log("=" * 60)

    rows = []
    job_total = len(verify_hp_list) * len(verify_fams) * len(verify_seeds)
    job_idx = 0
    summary_fieldnames = None

    for tag, h in verify_hp_list:
        hp = dict(h)
        hp.update(hp_verify_extra)

        for fam in verify_fams:
            for sd in verify_seeds:
                job_idx += 1
                log(f"\n[VERIFY {job_idx}/{job_total}] hp={tag} fam={fam} seed={sd}")
                r = run_one_experiment(
                    model_type=model_type,
                    phys_source=phys_source,
                    recipe_aug_mode=recipe_aug_mode,
                    phys7_mode=phys7_mode,
                    root_out=verify_root,
                    split_seed=sd,
                    job_idx=job_idx,
                    job_total=job_total,
                    target_family=fam,
                    shared_df=df_cache,
                    shared_recipe_cols=recipe_cols_cache,
                    shared_recipe_raw=recipe_raw_cache,
                    shared_targets_full=targets_full,
                    shared_mask_full=mask_full,
                    shared_phys7_seq_cache=phys7_seq_cache,
                    shared_stageA_provider=stageA_provider_cache,
                    hp_override=hp,
                    hp_tag=tag,
                )

                rows.append(r)
                if summary_fieldnames is None:
                    summary_fieldnames = list(r.keys())
                append_summary_row(r, verify_root, fieldnames=summary_fieldnames)

    postprocess_summary_from_csv(verify_root, write_excel=True)

    # 最终输出：verify_root 下会有 best_config_common_all_families.json / results_config_seed_agg.csv 等
    log("[OK] tune_minper_auto all done.")
    log(f"  screen_root={screen_root}")
    log(f"  verify_root={verify_root}")


def main(runs_root: str = None, plan: str = "auto"):
    if runs_root is None:
        runs_root = Cfg.save_root
    _ensure_dir(runs_root)

    set_seed(Cfg.seed)

    (df_cache, recipe_cols_cache, recipe_raw_cache,
     stageA_provider_cache, targets_full, mask_full,
     phys7_seq_cache, case_ids_cache,
     bad_row, before_N, kept_idx, clipped_stats) = prepare_shared_cache()

    kwargs = dict(
        df_cache=df_cache,
        recipe_cols_cache=recipe_cols_cache,
        recipe_raw_cache=recipe_raw_cache,
        stageA_provider_cache=stageA_provider_cache,
        targets_full=targets_full,
        mask_full=mask_full,
        phys7_seq_cache=phys7_seq_cache,
        case_ids_cache=case_ids_cache,
    )

    plan_l = str(plan).lower().strip()

    # -------------------------
    # auto：tune -> best(hp+split) -> ablationA(only best split)
    # -------------------------
    if plan_l in ["auto", "tune_then_ablation", "tune+ablation"]:
        # 1) tune
        tune_minper_auto(runs_root, **kwargs)

        # 2) load best config (hp + best split_seed)
        best_conf = load_best_common_config_from_tune_verify(runs_root) or {}
        apply_hp_from_best_conf_to_cfg(best_conf)
        best_split_seed = get_best_split_seed_from_best_conf(best_conf, default_seed=0)

        best_hp_tag = None
        if isinstance(best_conf, dict):
            best_hp_tag = best_conf.get("hp_tag", None)

        log(f"[AUTO] best_split_seed={best_split_seed}")
        log("=" * 40)
        log("[ABLATION START] Base Config Loaded.")
        log(f"  > Best Seed: {best_split_seed}")
        log(f"  > Current Model Architecture in Code: {getattr(Cfg, 'model_archs', ['transformer','gru','mlp'])}")
        log("  > (Please ensure this architecture matches the one you tuned!)")
        log("=" * 40)
        log(f"[ABL] use best_split_seed={best_split_seed}")

        out_root = os.path.join(runs_root, "ablationA")
        _ensure_dir(out_root)

        jobs = build_job_list_ablationA()
        run_job_list(
            jobs, out_root, seeds=[best_split_seed],
            hp_tag=best_hp_tag,  # ✅ 关键：使 ablation 与 verify 同 tag / 同 exp_name
            **dict(
                shared_df=df_cache,
                shared_recipe_cols=recipe_cols_cache,
                shared_recipe_raw=recipe_raw_cache,
                shared_targets_full=targets_full,
                shared_mask_full=mask_full,
                shared_phys7_seq_cache=phys7_seq_cache,
                shared_stageA_provider=stageA_provider_cache,
            )
        )
        return

    # -------------------------
    # 单独跑 tune
    # -------------------------
    if plan_l == "tune_minper_auto":
        tune_minper_auto(runs_root, **kwargs)
        return

    # -------------------------
    # 单独跑 ablationA（要求先有 _tuneV_verify/best_config...）
    # -------------------------
    if plan_l == "ablationa":
        best_conf = load_best_common_config_from_tune_verify(runs_root)
        if not isinstance(best_conf, dict):
            raise RuntimeError(
                f"CRITICAL: Cannot run ablation! 'best_config...missing in {runs_root}/_tuneV_verify. Please run 'tune' first."
            )

        apply_hp_from_best_conf_to_cfg(best_conf)
        best_split_seed = get_best_split_seed_from_best_conf(best_conf, default_seed=0)
        best_hp_tag = best_conf.get("hp_tag", None)

        log("=" * 40)
        log("[ABLATION START] Base Config Loaded.")
        log(f"  > Best Seed: {best_split_seed}")
        log(f"  > Current Model Architecture in Code: {getattr(Cfg, 'model_archs', ['transformer','gru','mlp'])}")
        log("  > (Please ensure this architecture matches the one you tuned!)")
        log("=" * 40)
        log(f"[ABL] use best_split_seed={best_split_seed}")

        out_root = os.path.join(runs_root, "ablationA")
        _ensure_dir(out_root)

        jobs = build_job_list_ablationA()
        run_job_list(
            jobs, out_root, seeds=[best_split_seed],
            hp_tag=best_hp_tag,  # ✅ 关键：对齐 verify
            **dict(
                shared_df=df_cache,
                shared_recipe_cols=recipe_cols_cache,
                shared_recipe_raw=recipe_raw_cache,
                shared_targets_full=targets_full,
                shared_mask_full=mask_full,
                shared_phys7_seq_cache=phys7_seq_cache,
                shared_stageA_provider=stageA_provider_cache,
            )
        )
        return

    # -------------------------
    # 其它 plan：保持你原逻辑不动（fullgrid/phase1/phase2...）
    # -------------------------
    if plan_l == "fullgrid":
        out_root = os.path.join(runs_root, "fullgrid")
        _ensure_dir(out_root)
        jobs = build_job_list_fullgrid()
        run_job_list(jobs, out_root, seeds=list(getattr(Cfg, "split_seeds", (0, 1, 2))), **dict(
            shared_df=df_cache,
            shared_recipe_cols=recipe_cols_cache,
            shared_recipe_raw=recipe_raw_cache,
            shared_targets_full=targets_full,
            shared_mask_full=mask_full,
            shared_phys7_seq_cache=phys7_seq_cache,
            shared_stageA_provider=stageA_provider_cache,
        ))
        return

    if plan_l == "phase1":
        out_root = os.path.join(runs_root, "phase1")
        _ensure_dir(out_root)
        jobs = build_job_list_phase1()
        run_job_list(jobs, out_root, seeds=list(getattr(Cfg, "split_seeds", (0, 1, 2))), **dict(
            shared_df=df_cache,
            shared_recipe_cols=recipe_cols_cache,
            shared_recipe_raw=recipe_raw_cache,
            shared_targets_full=targets_full,
            shared_mask_full=mask_full,
            shared_phys7_seq_cache=phys7_seq_cache,
            shared_stageA_provider=stageA_provider_cache,
        ))
        return

    if plan_l == "phase2":
        out_root = os.path.join(runs_root, "phase2")
        _ensure_dir(out_root)
        jobs = build_job_list_phase2()
        run_job_list(jobs, out_root, seeds=list(getattr(Cfg, "split_seeds", (0, 1, 2))), **dict(
            shared_df=df_cache,
            shared_recipe_cols=recipe_cols_cache,
            shared_recipe_raw=recipe_raw_cache,
            shared_targets_full=targets_full,
            shared_mask_full=mask_full,
            shared_phys7_seq_cache=phys7_seq_cache,
            shared_stageA_provider=stageA_provider_cache,
        ))
        return

    raise ValueError(f"Unknown plan={plan}. Supported: auto | tune_minper_auto | ablationA | fullgrid | phase1 | phase2")

if __name__ == "__main__":
    if os.name == 'nt':
        import multiprocessing
        multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default=None, help="输出根目录（默认用 Cfg.save_root）")
    parser.add_argument(
        "--plan", type=str, default="ablationA",
        help="auto(tune->ablationA) | tune_minper_auto | ablationA | fullgrid | phase1 | phase2"
    )
    parser.add_argument("--self_test", action="store_true", help="只跑 quick_self_test，不训练")
    args = parser.parse_args()

    if args.self_test:
        quick_self_test()
    else:
        main(runs_root=args.runs_root, plan=args.plan)
