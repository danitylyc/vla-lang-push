"""Closed-loop evaluation of the trained VLA policy in the simulator.

Three conditions, each measured as success rate over episodes:
  * VLA (seen phrasings)      - instruction wording seen during training
  * VLA (unseen phrasings)    - hold-out wording (generalization)
  * random baseline           - uniform random actions

Every rollout runs the real physics sim (PandaPush-v3); success = cube within
0.05 m of the region center within max_steps.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from vla_env import (EVAL_ONLY_TEMPLATES, REGIONS, REGION_NAMES, TEMPLATE_POOL,
                     VOCAB, LanguagePushEnv, make_oracle, tokenize_instr)
from vla_model import VLAPolicy


def rollout(env, region, instr_text, policy, max_steps=60):
    rgb, _, obs, _ = env.reset(region)
    instr = tokenize_instr(instr_text)
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
    instr_t = torch.from_numpy(instr.astype(np.int64)).unsqueeze(0)
    final_cube = None
    for _ in range(max_steps):
        with torch.no_grad():
            act = policy(rgb_t, instr_t).squeeze(0).numpy()
        rgb, _, done, info = env.step(act)
        final_cube = info["obs"]["achieved_goal"][:2].copy()
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
        if done:
            break
    return bool(info["success"]), final_cube


def region_of(pos2):
    """index of the region whose center is nearest to pos2"""
    best, best_d = -1, 1e9
    for i, r in enumerate(REGION_NAMES):
        d = np.linalg.norm(pos2 - REGIONS[r][:2])
        if d < best_d:
            best, best_d = i, d
    return best


def eval_condition(env, region, texts, policy, n_ep, seed):
    rng = np.random.RandomState(seed)
    ok = hit = 0
    total = 0
    dists = []
    for t in texts:
        for _ in range(n_ep):
            succ, final_cube = rollout(env, region, t, policy)
            ok += int(succ)
            if final_cube is not None:
                hit += int(region_of(final_cube) == REGION_NAMES.index(region))
                dists.append(float(np.linalg.norm(final_cube - REGIONS[region][:2])))
            total += 1
    return ok, hit, total, float(np.mean(dists)) if dists else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--n-per-cond", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", action="store_true", help="also run the analytic oracle")
    args = ap.parse_args()

    env = LanguagePushEnv(img=64, seed=1)
    rng = np.random.RandomState(args.seed)

    model = VLAPolicy(vocab_size=len(VOCAB))
    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # trusted local file
    model.load_state_dict(ck["model"])
    model.eval()
    ep = ck.get("epoch", ck.get("eval_succ", "?"))
    print(f"loaded {args.ckpt} (epoch/eval_succ {ep})")

    results = {}
    # --- VLA seen phrasings (training wording, all regions) ---
    seen_ok = seen_hit = seen_tot = 0
    seen_d = []
    for region in REGION_NAMES:
        texts = [t for r, t in TEMPLATE_POOL if r == region]
        ok, hit, tot, md = eval_condition(env, region, texts, model, args.n_per_cond, args.seed + 1)
        seen_ok += ok; seen_hit += hit; seen_tot += tot; seen_d.append(md)
        print(f"  seen {region:13s} succ {ok}/{tot} ({ok / tot:.0%})  region-hit {hit}/{tot} ({hit / tot:.0%})  final-dist {md:.3f}")
    results["vla_seen"] = (seen_ok, seen_hit, seen_tot, float(np.mean(seen_d)))

    # --- VLA unseen phrasings (hold-out wording) ---
    un_ok = un_hit = un_tot = 0
    un_d = []
    for region in REGION_NAMES:
        ok, hit, tot, md = eval_condition(env, region, EVAL_ONLY_TEMPLATES[region], model,
                                         args.n_per_cond, args.seed + 2)
        un_ok += ok; un_hit += hit; un_tot += tot; un_d.append(md)
        print(f"  unseen {region:10s} succ {ok}/{tot} ({ok / tot:.0%})  region-hit {hit}/{tot} ({hit / tot:.0%})  final-dist {md:.3f}")
    results["vla_unseen"] = (un_ok, un_hit, un_tot, float(np.mean(un_d)))

    # --- random baseline ---
    def random_policy(rgb_t, instr_t):
        return torch.from_numpy(np.random.uniform(-1, 1, 3).astype(np.float32)).unsqueeze(0)
    r_ok = r_hit = r_tot = 0
    r_d = []
    for region in REGION_NAMES:
        texts = [t for r, t in TEMPLATE_POOL if r == region]
        ok, hit, tot, md = eval_condition(env, region, texts, random_policy, args.n_per_cond, args.seed + 3)
        r_ok += ok; r_hit += hit; r_tot += tot; r_d.append(md)
    results["random"] = (r_ok, r_hit, r_tot, float(np.mean(r_d)))
    print(f"  random         succ {r_ok}/{r_tot} ({r_ok / r_tot:.0%})  region-hit {r_hit}/{r_tot} ({r_hit / r_tot:.0%})  final-dist {np.mean(r_d):.3f}")

    # --- oracle upper bound (optional) ---
    if args.oracle:
        o_ok = o_hit = o_tot = 0
        o_d = []
        for region in REGION_NAMES:
            texts = [t for r, t in TEMPLATE_POOL if r == region]
            for t in texts:
                for _ in range(args.n_per_cond):
                    rgb, _, obs, _ = env.reset(region)
                    oracle = make_oracle()
                    for _ in range(60):
                        act = oracle(obs)
                        rgb, _, done, info = env.step(act)
                        obs = info["obs"]
                        if done:
                            break
                    final_cube = info["obs"]["achieved_goal"][:2]
                    o_ok += int(info["success"]); o_tot += 1
                    o_hit += int(region_of(final_cube) == REGION_NAMES.index(region))
                    o_d.append(float(np.linalg.norm(final_cube - REGIONS[region][:2])))
        results["oracle"] = (o_ok, o_hit, o_tot, float(np.mean(o_d)))
        print(f"  oracle         succ {o_ok}/{o_tot} ({o_ok / o_tot:.0%})  region-hit {o_hit}/{o_tot} ({o_hit / o_tot:.0%})  final-dist {np.mean(o_d):.3f}")

    env.close()
    os.makedirs("results", exist_ok=True)
    np.save(os.path.join("results", "eval_summary.npy"),
            np.array([(k,) + tuple(v) for k, v in results.items()], dtype=object))
    print("saved -> results/eval_summary.npy")


if __name__ == "__main__":
    main()
