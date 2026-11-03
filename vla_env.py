"""VLA-Push: language-conditioned pushing with the Panda arm (panda-gym / PyBullet).

A from-scratch vision-language-action task:
  image (64x64 RGB) + natural-language instruction -> 3D end-effector delta action.

This module provides:
  * discrete goal regions with natural-language instructions (vocabulary + tokenizer)
  * a LanguagePushEnv wrapper around gymnasium PandaPush-v3
  * an oracle (analytic pushing) policy used to generate demonstrations
  * dataset collection into a single .npz (train split)

Usage:
  python vla_env.py probe          # smoke-test the env + render timing
  python vla_env.py verify-oracle  # measure oracle success rate
  python vla_env.py collect --n 1200 --out data/vla_push_train.npz
"""
from __future__ import annotations

import argparse
import os
import time

import gymnasium
import numpy as np
import panda_gym  # noqa: F401  (registers PandaPush-v3 etc. in gymnasium)

# ---------------------------------------------------------------------------
# language-conditioned goal regions
# ---------------------------------------------------------------------------
# table-top plane: object/goal sampled in x,y in [-0.15, 0.15]; z is cube-half-height
_CUBE_HALF = 0.04  # Push.object_size / 2
_REGION_R = 0.075

# region -> (name, center_xyz)
REGIONS = {
    "top-left":      np.array([-_REGION_R, +_REGION_R, _CUBE_HALF]),
    "top-right":     np.array([+_REGION_R, +_REGION_R, _CUBE_HALF]),
    "bottom-left":   np.array([-_REGION_R, -_REGION_R, _CUBE_HALF]),
    "bottom-right":  np.array([+_REGION_R, -_REGION_R, _CUBE_HALF]),
}
REGION_NAMES = list(REGIONS.keys())

# instruction templates -> (region key). Multi-word forms per region.
_TEMPLATES = {
    "top-left": [
        "push the green cube to the top left target",
        "move the cube to the upper left target",
        "push the block towards the top left",
    ],
    "top-right": [
        "push the green cube to the top right target",
        "move the cube to the upper right target",
        "push the block towards the top right",
    ],
    "bottom-left": [
        "push the green cube to the bottom left target",
        "move the cube to the lower left target",
        "push the block towards the bottom left",
    ],
    "bottom-right": [
        "push the green cube to the bottom right target",
        "move the cube to the lower right target",
        "push the block towards the bottom right",
    ],
}
TEMPLATE_POOL = [(r, t) for r, ts in _TEMPLATES.items() for t in ts]

# fixed small vocabulary (template-generated instructions -> closed set)
VOCAB = ["<pad>", "<cls>",
         "push", "move", "put", "the", "green", "cube", "block", "to", "target",
         "in", "at", "top", "bottom", "left", "right", "upper", "lower",
         "towards", "please", "region", "corner"]
VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}
MAX_INSTR_LEN = 8

# hold-out instruction phrasings, unseen at training time (for generalization eval)
EVAL_ONLY_TEMPLATES = {
    "top-left":      ["put the cube in the top left region", "at the upper left corner please"],
    "top-right":     ["put the cube in the top right region", "at the upper right corner please"],
    "bottom-left":   ["put the cube in the bottom left region", "at the lower left corner please"],
    "bottom-right":  ["put the cube in the bottom right region", "at the lower right corner please"],
}


def tokenize_instr(text: str) -> np.ndarray:
    """Tokenize an instruction into padded token ids (int32, MAX_INSTR_LEN)."""
    ids = [VOCAB_INDEX["<cls>"]] + [VOCAB_INDEX[w] for w in text.split() if w in VOCAB_INDEX]
    ids = ids[:MAX_INSTR_LEN]
    ids += [VOCAB_INDEX["<pad>"]] * (MAX_INSTR_LEN - len(ids))
    return np.array(ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# environment wrapper
# ---------------------------------------------------------------------------
class LanguagePushEnv:
    """PandaPush-v3 wrapped for language-conditioned discrete goal regions.

    reset(region) fixes the target marker to the region center while the cube
    starts at a random pose (panda-gym default). Image is rendered at IMG size.
    """

    IMG = 64

    def __init__(self, img: int = 64, max_episode_steps: int = 60, seed: int = 0):
        self.env = gymnasium.make(
            "PandaPush-v3",
            render_mode="rgb_array",
            renderer="OpenGL",
            render_width=img,
            render_height=img,
            render_yaw=45.0,
            render_pitch=-30.0,
            render_distance=1.4,
        )
        self._max_steps = max_episode_steps
        self.img = img
        self._rng = np.random.RandomState(seed)

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, region: str | int, start_min: float = 0.060, start_max: float = 0.085) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Reset for a given region.

        The cube starts at a random pose in the annulus
        [start_min, start_max] around the region center (outside the success
        disk of radius 0.05), sampled uniformly with rejection. The task is:
        "follow the language instruction, push the cube onto the region center".
        Returns (rgb, instr_tokens, obs, info).
        """
        if isinstance(region, int):
            region = REGION_NAMES[region]
        obs, _ = self.env.reset()
        goal = REGIONS[region].copy()
        task = self.env.unwrapped.task
        task.goal = goal.copy()
        task.sim.set_base_pose("target", goal.copy(), np.array([0.0, 0.0, 0.0, 1.0]))
        # cube start: uniform in the annulus around the region center
        for _ in range(200):
            r = start_min + self._rng.uniform() * (start_max - start_min)
            a = self._rng.uniform(0.0, 2 * np.pi)
            xy = goal[:2] + np.array([r * np.cos(a), r * np.sin(a)])
            if np.max(np.abs(xy)) <= 0.20:  # keep well inside the table
                break
        obj_pos = np.array([xy[0], xy[1], _CUBE_HALF])
        task.sim.set_base_pose("object", obj_pos, np.array([0.0, 0.0, 0.0, 1.0]))
        obs = self.env.unwrapped._get_obs()
        rgb = np.asarray(self.env.render(), dtype=np.uint8)
        instr = tokenize_instr(_TEMPLATES[region][0])
        return rgb, instr, obs, {"region": region}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = np.asarray(self.env.render(), dtype=np.uint8)
        success = bool(info.get("is_success", False))
        return rgb, float(reward), bool(terminated or truncated), {"success": success, "obs": obs}

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# oracle: analytic pushing policy (uses proprioceptive obs, generates demos)
# ---------------------------------------------------------------------------
def make_oracle(grasp_gap: float = 0.06, ee_h: float = 0.030,
                k_approach: float = 8.0, push_speed: float = 0.85,
                z_gain: float = 4.0, fly_h: float = 0.10):
    """Return a stateful analytic push policy: (obs) -> action.

    Phase A (approach): the end-effector *flies over the cube* (z = fly_h, above
    the cube top) to a point `grasp_gap` behind the cube, so it never touches the
    cube while approaching; Phase B (lower): straight down onto the push height
    ee_h (below the cube centre, so the fingertip sphere overlaps the cube side);
    Phase C (push): open-loop thrust along the *recomputed* cube->goal direction
    with z held, giving long pushes a stable contact. If the cube drifts away
    (end-effector farther than 0.14 m behind it), re-approach once.
    """
    state = {"phase": "approach"}

    def policy(obs: dict) -> np.ndarray:
        cube = np.asarray(obs["achieved_goal"], dtype=np.float64).copy()
        goal = np.asarray(obs["desired_goal"], dtype=np.float64).copy()
        ee = np.asarray(obs["observation"][:3], dtype=np.float64).copy()

        to_goal = goal - cube
        to_goal[2] = 0.0
        dist_xy = np.linalg.norm(to_goal)
        if dist_xy > 1e-6:
            dirn = to_goal / dist_xy
        else:
            dirn = np.zeros(3)
            state["phase"] = "done"
        behind = cube - grasp_gap * dirn

        if state["phase"] == "approach":
            target = np.array([behind[0], behind[1], fly_h])
            to = target - ee
            if np.linalg.norm(to[:2]) < 0.02 and abs(to[2]) < 0.03:
                state["phase"] = "lower"
            else:
                return np.clip(k_approach * to, -1.0, 1.0).astype(np.float32)

        if state["phase"] == "lower":
            target = np.array([behind[0], behind[1], ee_h])
            to = target - ee
            if np.linalg.norm(to[:2]) < 0.025 and abs(to[2]) < 0.02:
                state["phase"] = "push"
            else:
                return np.clip(k_approach * to, -1.0, 1.0).astype(np.float32)

        # push phase: open-loop thrust along the *current* cube->goal direction
        # (direction is recomputed every step -> closed-loop correction), z held.
        # When the cube gets close to the goal (< 0.08 m), slow down (0.35) so
        # the demos contain a "finishing" behaviour the clone can imitate
        # instead of overshooting through the goal region.
        if np.linalg.norm(ee[:2] - cube[:2]) > 0.14 and not state.get("reapproached"):
            state["reapproached"] = True
            state["phase"] = "approach"
            return policy(obs)
        speed = push_speed if dist_xy > 0.08 else 0.35
        act = np.array([speed * dirn[0], speed * dirn[1],
                        z_gain * (ee_h - ee[2])])
        return np.clip(act, -1.0, 1.0).astype(np.float32)

    return policy


# ---------------------------------------------------------------------------
# dataset collection
# ---------------------------------------------------------------------------
def collect(env: LanguagePushEnv, n_episodes: int, seed: int = 0,
            max_steps: int = 60, keep_only_success: bool = True) -> dict:
    rng = np.random.RandomState(seed)
    rgb_all, instr_all, action_all, region_all = [], [], [], []
    n_success = 0

    for ep in range(n_episodes):
        region = REGION_NAMES[rng.randint(len(REGION_NAMES))]
        rgb, instr, obs, _ = env.reset(region)
        # vary the instruction wording per episode (same region semantics)
        tpls = [t for r, t in TEMPLATE_POOL if r == region]
        instr = tokenize_instr(tpls[rng.randint(len(tpls))])
        oracle = make_oracle()
        done = False
        steps = 0
        ep_rgb, ep_instr, ep_act = [], [], []
        while not done and steps < max_steps:
            act = oracle(obs)
            rgb, _, done, info = env.step(act)
            ep_rgb.append(rgb.transpose(2, 0, 1))  # (3,H,W) for torch
            ep_instr.append(instr)
            ep_act.append(act)
            obs = info["obs"]
            steps += 1
        if info["success"]:
            n_success += 1
            rgb_all += ep_rgb
            instr_all += ep_instr
            action_all += ep_act
            region_all += [REGION_NAMES.index(region)] * len(ep_act)
        if (ep + 1) % 100 == 0:
            print(f"  ep {ep + 1}/{n_episodes}  success_so_far={n_success}")

    n = len(action_all)
    print(f"oracle success: {n_success}/{n_episodes} ({n_success / n_episodes:.1%}); "
          f"kept {n} transition steps (success trajectories only)")
    return {
        "rgb": np.stack(rgb_all).astype(np.uint8),
        "instr": np.stack(instr_all).astype(np.int32),
        "action": np.stack(action_all).astype(np.float32),
        "region": np.array(region_all, dtype=np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "verify-oracle", "collect"])
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="data/vla_push_train.npz")
    ap.add_argument("--img", type=int, default=64)
    args = ap.parse_args()

    if args.cmd == "probe":
        env = LanguagePushEnv(img=args.img)
        rgb, instr, obs, info = env.reset("top-left")
        print("rgb", rgb.shape, rgb.dtype, "instr", instr, "region", info["region"])
        t0 = time.time()
        for _ in range(10):
            rgb2, _, done, info = env.step(np.zeros(3, dtype=np.float32))
        dt = (time.time() - t0) / 10
        print(f"step+render {dt * 1000:.1f} ms/step -> ~{1 / dt:.0f} steps/s")
        env.close()

    elif args.cmd == "verify-oracle":
        env = LanguagePushEnv(img=args.img)
        n, ok, ep = 100, 0, 0
        rng = np.random.RandomState(0)
        while ep < n:
            region = REGION_NAMES[rng.randint(4)]
            rgb, instr, obs, _ = env.reset(region)
            oracle = make_oracle()
            for _ in range(60):
                act = oracle(obs)
                rgb, _, done, info = env.step(act)
                obs = info["obs"]
                if done:
                    break
            ok += int(info["success"])
            ep += 1
        print(f"oracle success {ok}/{n} ({ok / n:.1%})")
        env.close()

    elif args.cmd == "collect":
        out = args.out
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        env = LanguagePushEnv(img=args.img)
        data = collect(env, n_episodes=args.n)
        env.close()
        np.savez_compressed(out, **data)
        for k, v in data.items():
            print(f"  {k}: {v.shape} {v.dtype}")
        print("saved ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
