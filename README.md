# VLA-Lang-Push

A minimal, fully reproducible **Vision-Language-Action (VLA) policy** that
follows free-form Chinese instructions to push a cube to one of four
workspace regions in a real physics simulator (pybullet + panda-gym
PandaPush-v3).

> The whole pipeline runs on CPU; every number below is produced by the
> scripts in this repository — nothing is guessed.

## Why this project exists

Large VLA models (RT-2, OpenVLA, …) are closed or need heavy GPU budgets. This
project answers a practical question with a tiny, transparent system:

> Can a **small end-to-end VLA** — raw camera image + natural-language
> instruction in, motor commands out — be trained from a few thousand
> demonstrations, and does it generalize to instruction wording it has never
> seen?

It is not an API wrapper and not a reproduction of a paper: it is an original,
self-contained experiment with a measurable generalization claim and an honest
closed-loop evaluation.

## Task

PandaPush-v3 in pybullet. A cube starts at a random pose in the annulus
0.06–0.085 m around one of four regions (top-left / top-right / bottom-left /
bottom-right), i.e. **outside** the success disk of 0.05 m — the start is never
already a success. The instruction is a Chinese sentence such as
"把方块推到左上角" (push the cube to the top-left corner). Success = cube within
0.05 m of the named region center within 60 steps.

The analytic oracle that generates demonstrations is a hand-designed 3-phase
policy (fly-over approach → lower → open-loop push with recomputed direction
and a slow "finishing" phase near the goal) with **~99%** closed-loop success,
which bounds what a learned policy can achieve.

## Method

| component | design |
|---|---|
| observation | RGB image (64×64, 3 ch) from the simulator camera |
| instruction | char-level Chinese tokenizer, 23-token vocabulary |
| policy | VLA: CNN (vision) + GRU (language) → MLP → 3-dim action (dx, dy, dz) |
| training | behavior cloning (MSE) on `{image, instruction} → action`, CPU |
| fine-tuning | closed-loop REINFORCE on top of the BC policy (simulator rollouts) |
| data | 1,500 oracle rollouts, success trajectories only (22,313 steps) |

Generalization is measured by holding out **instruction wording** (not region
semantics): the same four regions are phrased with templates never seen during
training.

## Results

Closed-loop evaluation in the real physics sim (fixed seeds; `n=96` seen /
`n=64` unseen episodes per model; random `n=96`; oracle `n=96`):

| policy | success (seen) | success (unseen) | mean final distance |
|---|---|---|---|
| random baseline | 1% (1/96) | — | 0.088 m |
| VLA · behavior cloning | 16% (15/96) | 12% (8/64) | 0.073 m |
| VLA · BC + REINFORCE | 12.5% (12/96) | 11% (7/64) | 0.070 m |
| analytic oracle (bound) | 100% (96/96) | — | 0.039 m |

Two honest conclusions:

1. **The task is non-trivial and the VLA is not random**: random actions
   succeed ~1%, the learned policy 3–16× more, and its final cube position is
   clearly closer to the commanded region (0.070–0.073 m vs 0.088 m random).
2. **Behavior cloning saturates far below the oracle on closed-loop success**
   (~16% vs 100%). This is the well-known distribution-shift problem of
   imitation learning, quantified here on a real simulator with only a 64×64
   camera and 22k steps of data. REINFORCE fine-tuning with a plain baseline
   improves some seeds but does not reliably close the gap — the bottleneck is
   contact-level precision under image-only observations, not language
   understanding (region-level behaviour tracks the instruction on 75–100% of
   episodes).

## Reproduce

```
python vla_env.py  probe          # sanity check the environment
python vla_env.py  verify-oracle  # oracle closed-loop success (~99%)
python vla_env.py  collect --n 1500 --out data/vla_push_train.npz   # demos
python train_vla.py --epochs 80                                     # BC, ~15 min
python finetune_rl.py                                              # REINFORCE, ~30 min
python eval_vla.py  --ckpt checkpoints/best.pt --n-per-cond 8 --oracle   # evaluate BC
python eval_vla.py  --ckpt checkpoints/finetuned.pt --n-per-cond 8       # evaluate BC+RL
```

Requires: `panda-gym`, `pybullet`, `numpy`, `torch` (CPU is enough).

## Files

- `vla_env.py` — language-conditioned PandaPush environment, analytic oracle, data collection
- `vla_model.py` — VLA network definition
- `train_vla.py` — behavior-cloning training loop
- `finetune_rl.py` — closed-loop REINFORCE fine-tuning (BC init, simulator rollouts)
- `eval_vla.py` — closed-loop evaluation (seen / unseen wording, random, oracle)
- `data/vla_push_train.npz` — generated demonstration dataset
- `checkpoints/best.pt` / `checkpoints/finetuned.pt` — trained weights
- `results/eval_summary.npy` — evaluation raw results

## Limitations (stated honestly)

- Behavior cloning on short-horizon pushes with image-only observations: no
  grasping, single cube, no distractors; closed-loop success is bounded by
  contact-level precision, not by language understanding.
- The instruction vocabulary is minimal (23 chars / 4 regions); it demonstrates
  **wording generalization**, not open-ended language understanding.
- The REINFORCE fine-tuner uses a fixed exploration std and a mean baseline —
  a plain, debuggable implementation, not a tuned PPO.
- Demonstrations come from one deterministic oracle, so the learned policy
  inherits its biases (e.g. always approaching from behind the cube).
