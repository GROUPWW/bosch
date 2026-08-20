# -*- coding: utf-8 -*-
"""
冒烟测试（CPU，不训练）：MorphLSTM / MorphTCN / MorphARMLP

- 用与 trainer 相同的维度构造每个模型（static_dim=Ds, phys_dim=7, T=len(TIME_LIST)）
- 喂 dummy 张量（shape 参照 MorphGRU 的输入：x (B,Ds), p7 (B,7,T), t (B,T)）
- 验证 forward 输出 shape (B,K,T)、无 NaN、各自能 backward 一步
- 同时回归验证 MorphGRU 输出 shape 未受影响
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import stageB_util as su


def check_one(name, model, x, p7, t, out_dim):
    model.train()
    y = model(x, p7, t)
    B, T = x.shape[0], p7.shape[2]
    assert y.shape == (B, out_dim, T), f"{name}: bad output shape {tuple(y.shape)}, expected {(B, out_dim, T)}"
    assert torch.isfinite(y).all(), f"{name}: NaN/Inf in output"

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = y.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_param = sum(1 for p in model.parameters())
    print(f"[OK] {name:10s} out={tuple(y.shape)} loss={loss.item():.4f} "
          f"params_with_grad={n_grad}/{n_param}")


def main():
    torch.manual_seed(0)
    B, Ds, T = 4, 37, len(su.TIME_LIST)  # Ds 取一个典型静态特征维度；T=9

    for out_dim in (6, 1):  # 联合训练 K=6 与 v9 单 family K=1 两种情况
        x = torch.randn(B, Ds)
        p7 = torch.randn(B, 7, T)
        t = torch.from_numpy(su.TIME_VALUES[None, :].repeat(B, 0) if False else
                             su.TIME_VALUES[None, :]).repeat(B, 1).float()  # (B,T)
        assert t.shape == (B, T)

        print(f"\n--- out_dim={out_dim} ---")
        check_one("MorphGRU", su.MorphGRU(static_dim=Ds, hidden=su.Cfg.gru_hidden,
                                          num_layers=su.Cfg.gru_layers, out_dim=out_dim), x, p7, t, out_dim)
        check_one("MorphLSTM", su.MorphLSTM(static_dim=Ds, hidden=su.Cfg.lstm_hidden,
                                            num_layers=su.Cfg.lstm_layers, out_dim=out_dim), x, p7, t, out_dim)
        check_one("MorphTCN", su.MorphTCN(static_dim=Ds, hidden=su.Cfg.tcn_hidden,
                                          num_layers=su.Cfg.tcn_layers, kernel_size=su.Cfg.tcn_kernel,
                                          out_dim=out_dim), x, p7, t, out_dim)
        check_one("MorphARMLP", su.MorphARMLP(static_dim=Ds, hidden=su.Cfg.armlp_hidden,
                                              num_layers=su.Cfg.armlp_layers, out_dim=out_dim), x, p7, t, out_dim)

    # TCN 因果性检查：扰动最后一个输入步，前面步的输出不应改变
    model = su.MorphTCN(static_dim=Ds, out_dim=6).eval()
    x = torch.randn(1, Ds)
    p7 = torch.randn(1, 7, T)
    t = su.TIME_VALUES[None, :].copy()
    t = torch.from_numpy(t).float()
    with torch.no_grad():
        y1 = model(x, p7, t)
        p7b = p7.clone()
        p7b[:, :, -1] += 10.0
        y2 = model(x, p7b, t)
    diff = (y1[:, :, :-1] - y2[:, :, :-1]).abs().max().item()
    assert diff == 0.0, f"MorphTCN: causality violated, diff={diff}"
    print(f"\n[OK] MorphTCN causality check passed (max diff on earlier steps = {diff})")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
