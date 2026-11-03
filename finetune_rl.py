"""Closed-loop fine-tuning of the behavior-cloned VLA policy with REINFORCE.

Behavior cloning alone saturates around ~15% closed-loop success because of
distribution shift (the simulator state drifts off the demonstration
distribution and errors compound). Standard VLA pipelines fine-tune the
policy in the loop; here we do the same with a tiny REINFORCE trainer:

    reward per step   = 8 * (distance_to_goal_{t-1} - distance_to_goal_t)
    + success bonus   = +10 on episode success

The policy head is the behavior-cloned VLA (CNN+GRU+MLP); the action is
sampled from a diagonal Gaussian around the network mean with fixed std=0.2,
so the policy gradient is computable from the observations already available
(rgb, instruction). Runs on CPU.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from vla_env import (REGIONS, REGION_NAMES, TEMPLATE_POOL, VOCAB,
                     LanguagePushEnv, make_oracle, tokenize_instr)
from vla_model import VLAPolicy


def make_prep():
    def prep(rgb, instr_text):
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
        it = torch.from_numpy(tokenize_instr(instr_text).astype(np.int64)).unsqueeze(0)
        return rgb_t, it
    return prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--ep-per-it", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--std", type=float, default=0.12)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=60)
    args = ap.parse_args()

    env = LanguagePushEnv(img=64, seed=7)
    prep = make_prep()

    model = VLAPolicy(vocab_size=len(VOCAB))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    std = args.std
    rng = np.random.RandomState(11)

    def rollout(region):
        rgb, _, obs, _ = env.reset(region)
        tpls = [t for r, t in TEMPLATE_POOL if r == region]
        text = tpls[rng.randint(len(tpls))]
        log_probs, rewards = [], []
        d_prev = float(np.linalg.norm(obs["achieved_goal"][:2] - REGIONS[region][:2]))
        done = False
        for _ in range(args.max_steps):
            rgb_t, itok = prep(rgb, text)
            mean = model(rgb_t, itok).squeeze(0)
            # sample action; detach it so d log_p / d mean = (a - mean)/std^2
            # (keeping the sampled action attached would cancel the gradient)
            act = (mean + torch.randn(3) * std).detach()
            diff = act - mean
            log_p = -0.5 * ((diff / std) ** 2).sum() - 1.5 * math.log(std) - 1.5 * math.log(2 * math.pi)
            rgb, _, done, info = env.step(act.numpy())
            d = float(np.linalg.norm(info["obs"]["achieved_goal"][:2] - REGIONS[region][:2]))
            r = 8.0 * (d_prev - d)
            rewards.append(r); log_probs.append(log_p)
            d_prev = d
            if done:
                break
        if info["success"]:
            rewards[-1] += 10.0
        return log_probs, rewards, bool(info["success"]), d_prev

    best_succ = -1.0
    best_state = None
    for it in range(1, args.iters + 1):
        log_probs_all, rewards_all = [], []
        succ_this = 0
        for _ in range(args.ep_per_it):
            region = REGION_NAMES[rng.randint(4)]
            lps, rs, succ, _ = rollout(region)
            succ_this += int(succ)
            # discount reward-to-go
            G = 0.0
            gs = []
            for r in reversed(rs):
                G = r + args.gamma * G
                gs.append(G)
            gs.reverse()
            log_probs_all += lps
            rewards_all += gs
        # REINFORCE with mean baseline
        base = float(np.mean(rewards_all))
        loss = 0.0
        for lp, g in zip(log_probs_all, rewards_all):
            loss = loss - lp * (g - base)
        loss = loss / len(log_probs_all)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                ok = tot = 0
                for region in REGION_NAMES:
                    for _ in range(5):
                        rgb, _, obs, _ = env.reset(region)
                        tpls = [t for r, t in TEMPLATE_POOL if r == region]
                        text = tpls[rng.randint(len(tpls))]
                        for _ in range(args.max_steps):
                            rgb_t, itok = prep(rgb, text)
                            act = model(rgb_t, itok).squeeze(0).numpy()
                            rgb, _, done, info = env.step(act)
                            if done:
                                break
                        ok += int(info["success"]); tot += 1
            succ = np.float64(ok) / np.float64(tot)
            print(f"iter {it}  train_succ {succ_this}/{args.ep_per_it}  eval_succ {ok}/{tot} = {100.0 * np.float64(ok) / np.float64(tot):.1f}%")
            if succ > best_succ:
                best_succ, best_state = succ, {k: v.clone() for k, v in model.state_dict().items()}
            model.train()

    env.close()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": best_state, "eval_succ": best_succ}, "checkpoints/finetuned.pt")
    print(f"best eval_succ {best_succ:.0%} -> checkpoints/finetuned.pt")


if __name__ == "__main__":
    main()
