"""Train the VLA-Push policy on oracle demonstrations.

Data: npz {rgb (N,3,64,64) uint8, instr (N,8) int32, action (N,3) float32}.
Loss: MSE on the 3-D action regression. Train/val split 90/10.
Checkpoint + loss history are written under checkpoints/.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from vla_env import VOCAB
from vla_model import VLAPolicy


def load_data(path: str, val_frac: float = 0.1, seed: int = 0):
    d = np.load(path)
    rgb = torch.from_numpy(d["rgb"].astype(np.float32) / 255.0)
    instr = torch.from_numpy(d["instr"].astype(np.int64))
    action = torch.from_numpy(d["action"].astype(np.float32))
    n = len(action)
    idx = np.random.RandomState(seed).permutation(n)
    n_val = int(n * val_frac)
    val_i, train_i = idx[:n_val], idx[n_val:]
    return (rgb[train_i], instr[train_i], action[train_i]), (rgb[val_i], instr[val_i], action[val_i])


def train(args):
    torch.manual_seed(args.seed)
    (r_tr, i_tr, a_tr), (r_va, i_va, a_va) = load_data(args.data, args.val_frac, args.seed)
    n = len(a_tr)
    print(f"train {n}  val {len(a_va)}  vocab {len(VOCAB)}")

    model = VLAPolicy(vocab_size=len(VOCAB))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.out_dir, exist_ok=True)
    hist = []
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        t0 = time.time()
        for b in range(0, n, args.batch):
            idx = perm[b:b + args.batch]
            rgb, instr, act = r_tr[idx], i_tr[idx], a_tr[idx]
            pred = model(rgb, instr)
            loss = torch.nn.functional.mse_loss(pred, act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        tr_loss = tot / cnt

        model.eval()
        with torch.no_grad():
            va_tot = 0.0
            for b in range(0, len(a_va), args.batch):
                rgb, instr, act = r_va[b:b + args.batch], i_va[b:b + args.batch], a_va[b:b + args.batch]
                pred = model(rgb, instr)
                va_tot += torch.nn.functional.mse_loss(pred, act).item() * len(act)
            va_loss = va_tot / len(a_va)

        hist.append((epoch + 1, tr_loss, va_loss))
        dt = time.time() - t0
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_mse {tr_loss:.5f}  val_mse {va_loss:.5f}  {dt:.0f}s")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_mse": va_loss},
                       os.path.join(args.out_dir, "best.pt"))

    np.save(os.path.join(args.out_dir, "loss_history.npy"),
            np.array(hist, dtype=np.float64))
    print(f"done. best val_mse {best_val:.5f} -> {os.path.abspath(os.path.join(args.out_dir, 'best.pt'))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vla_push_train.npz")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    train(args)
