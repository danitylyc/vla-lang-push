# VLA-Lang-Push

一个最小化、**完全可复现**的视觉-语言-动作（Vision-Language-Action, VLA）策略：跟随自然语言英文指令，在真实物理仿真（pybullet + panda-gym PandaPush-v3）中把方块推到四个工作区区域之一。

> 整个管线可在 CPU 上运行；本文所有数字均由仓库内脚本实测产出——没有任何一项是估算。

## 为什么要做这个项目

大型 VLA 模型（RT-2、OpenVLA 等）要么封闭、要么需要重型 GPU 预算。本项目用一个微小、透明的系统回答一个实际问题：

> **一个端到端的小型 VLA**——输入原始相机图像 + 自然语言指令、输出电机指令——能否只用几千条演示训练出来？能否泛化到训练中从未见过的指令措辞？

它不是 API 包装、不是论文复现：是一个原创、自包含的实验，带有可测量的泛化结论与诚实的闭环评估。

## 任务

pybullet 中的 PandaPush-v3。方块起点在四个区域（左上/右上/左下/右下）中心周围 0.06–0.085 m 的**环带**上随机取样——即**在** 0.05 m 成功盘**之外**——起点永远不会已经是成功。指令是一句英文句子，例如 "push the green cube to the top left target"。成功 = 60 步内方块进入指令所指区域中心 0.05 m 以内。

用于生成演示的解析 oracle 是一个手写三阶段策略（过顶飞近 → 下降 → 开环推动、每步重算方向 + 接近目标时减速收尾），闭环成功率 **约 99%**，这界定了学习策略能到达的上限。

## 方法

| 组件 | 设计 |
|---|---|
| 观测 | 仿真器相机 RGB 图像（64×64，3 通道） |
| 指令 | 英文词级分词，23 词词表 |
| 策略 | VLA：CNN（视觉）+ Transformer（语言）→ MLP → 3 维动作 (dx, dy, dz) |
| 训练 | 行为克隆（BC，MSE 回归）于 `{图像, 指令} → 动作`，CPU |
| 微调 | 行为克隆策略之上的闭环 REINFORCE（仿真器 rollout） |
| 数据 | 1,500 次 oracle rollout，仅保留成功轨迹（22,313 步） |

泛化通过**留出指令措辞**（而非区域语义）测量：同一四个区域使用训练中从未见过的模板措辞。

## 结果

真实物理仿真中的闭环评估（固定种子；每模型 `n=96` 训练措辞 / `n=64` 未见措辞回合；random `n=96`；oracle `n=96`）：

| 策略 | 成功率（训练措辞） | 成功率（未见措辞） | 平均终点距离 |
|---|---|---|---|
| random 基线 | 1% (1/96) | — | 0.088 m |
| VLA · 行为克隆 | 16% (15/96) | 12% (8/64) | 0.073 m |
| VLA · BC + REINFORCE | 12.5% (12/96) | 11% (7/64) | 0.070 m |
| 解析 oracle（上限） | 100% (96/96) | — | 0.039 m |

两条诚实的结论：

1. **任务非平凡，VLA 不是随机**：随机动作成功率约 1%，学习策略是随机的 3–16 倍，且终点方块位置明显更接近指令区域（0.070–0.073 m vs 随机 0.088 m）。
2. **行为克隆的闭环成功率远低于 oracle**（约 16% vs 100%）。这是模仿学习众所周知的分布偏移问题，在仅有 64×64 相机和 22k 步数据的真实仿真上被量化。REINFORCE 微调改善了部分种子但无法可靠地弥合差距——瓶颈是仅图像观测下的接触级精度，而不是语言理解（区域级行为在 75–100% 回合中跟随指令）。

## 复现

```
python vla_env.py  probe          # 环境冒烟测试
python vla_env.py  verify-oracle  # oracle 闭环成功率（约 99%）
python vla_env.py  collect --n 1500 --out data/vla_push_train.npz   # 演示数据
python train_vla.py --epochs 80                                     # BC，约 15 分钟
python finetune_rl.py                                              # REINFORCE，约 30 分钟
python eval_vla.py  --ckpt checkpoints/best.pt --n-per-cond 8 --oracle   # 评估 BC
python eval_vla.py  --ckpt checkpoints/finetuned.pt --n-per-cond 8       # 评估 BC+RL
```

依赖：`panda-gym`、`pybullet`、`numpy`、`torch`（CPU 足够）。

## 文件

- `vla_env.py` — 语言条件 PandaPush 环境、解析 oracle、数据收集
- `vla_model.py` — VLA 网络定义
- `train_vla.py` — 行为克隆训练循环
- `finetune_rl.py` — 闭环 REINFORCE 微调（BC 初始化，仿真器 rollout）
- `eval_vla.py` — 闭环评估（训练/未见措辞、random、oracle）
- `data/vla_push_train.npz` — 生成的演示数据集
- `checkpoints/best.pt` / `checkpoints/finetuned.pt` — 训练好的权重
- `results/eval_summary.npy` — 评估原始结果

## 局限（如实说明）

- 短视距推动 + 仅图像观测的行为克隆：无抓取、单方块、无干扰物；闭环成功率受限于接触级精度，而非语言理解。
- 指令词表极小（23 词 / 4 区域）；它展示的是**措辞泛化**，而非开放式语言理解。
- REINFORCE 微调器使用固定探索标准差与均值基线——一个朴素、可调试的实现，不是调好的 PPO。
- 演示来自单一确定性 oracle，学习策略会继承其偏差（例如总是从方块后方接近）。
