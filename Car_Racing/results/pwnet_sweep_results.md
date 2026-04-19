# PW-Net K-Sweep Results

**Pipeline:** Grid-balanced action clustering → visual latent centroid prototype selection → PW-Net wrapper training  
**Date:** 2026-04-17  
**Iterations per k:** 5  
**Simulation episodes per iteration:** 30  
**Training epochs:** 10  
**Dataset:** 43,134 timesteps from pre-trained PPO agent (`prototype_data.npz`)

---

## Summary Table

| k | Reward Mean | Reward SE | Sim MSE Mean | Sim MSE SE |
|:-:|------------:|----------:|-------------:|-----------:|
| 2 | 192.26 | 0.91 | 8.3359 | 0.2317 |
| 4 | **223.22** | **0.96** | 3.6431 | 0.0868 |
| 6 | 219.64 | 1.44 | 1.8201 | 0.0782 |
| 8 | 219.34 | 0.68 | 1.2431 | 0.0531 |

> **Reward** = shaped cumulative reward per episode (per-step reward clipped to [-1, 1]), averaged over 30 simulation episodes and 5 random seeds.  
> **Sim MSE** = sum of per-step squared error between PW-Net action and black-box PPO action, averaged over 30 episodes. Scales with episode length (~500 steps), so per-step MSE ≈ Sim MSE / 500.

---

## Per-Iteration Results

### k = 2

| Iteration | Reward | Sim MSE |
|:---------:|-------:|--------:|
| 1 | 192.76 | 9.1868 |
| 2 | 190.10 | 7.8836 |
| 3 | 194.35 | 8.4005 |
| 4 | 189.68 | 7.7155 |
| 5 | 194.41 | 8.4932 |
| **Mean ± SE** | **192.26 ± 0.91** | **8.3359 ± 0.2317** |

Training converged to eval MSE ≈ 0.0003 by epoch 1 and held steady.

### k = 4

| Iteration | Reward | Sim MSE |
|:---------:|-------:|--------:|
| 1 | 223.57 | 3.4825 |
| 2 | 224.12 | 3.9667 |
| 3 | 220.68 | 3.5507 |
| 4 | 226.61 | 3.4550 |
| 5 | 221.16 | 3.7606 |
| **Mean ± SE** | **223.22 ± 0.96** | **3.6431 ± 0.0868** |

Training converged to eval MSE ≈ 0.0002. **Best reward of any k value.**

### k = 6

| Iteration | Reward | Sim MSE |
|:---------:|-------:|--------:|
| 1 | 222.43 | 2.0018 |
| 2 | 222.46 | 1.5528 |
| 3 | 214.14 | 2.0130 |
| 4 | 221.36 | 1.8186 |
| 5 | 217.83 | 1.7145 |
| **Mean ± SE** | **219.64 ± 1.44** | **1.8201 ± 0.0782** |

Training converged to eval MSE ≈ 0.0001.

### k = 8

| Iteration | Reward | Sim MSE |
|:---------:|-------:|--------:|
| 1 | 220.52 | 1.3465 |
| 2 | 219.69 | 1.0301 |
| 3 | 216.34 | 1.1977 |
| 4 | 219.93 | 1.3072 |
| 5 | 220.23 | 1.3337 |
| **Mean ± SE** | **219.34 ± 0.68** | **1.2431 ± 0.0531** |

Training converged to eval MSE ≈ 0.0001. Most consistent reward across iterations (lowest SE).

---

## Key Observations

**Reward peaks at k=4, not k=8.** Despite sim MSE decreasing monotonically with k (better action-level fidelity), driving reward peaks at k=4 (223.22) and plateaus for k=6 and k=8 (~219-220). The differences between k=4, 6, and 8 are small but consistent across all 5 iterations.

**k=2 is clearly insufficient.** A ~15% reward drop relative to k=4 and substantially higher action error suggest two prototypes cannot capture the necessary maneuver diversity.

**Sim MSE and reward decouple at higher k.** Sim MSE continues to improve from k=4→8 (3.64→1.24), but reward does not follow. The extra prototypes at k=6 and k=8 likely represent finer subdivisions of existing maneuvers rather than genuinely new ones, creating redundancy in the frozen W matrix without improving driving decisions.

**k=4 is the interpretability sweet spot** — best reward with the fewest prototypes.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Latent size | 256 |
| Prototype size | 50 |
| Batch size | 32 |
| Learning rate | 0.01 |
| LR scheduler | ExponentialLR (γ=0.97) |
| W matrix | Frozen, initialized from prototype actions |
| Clustering | 3D grid-balanced (20 bins/axis, max 200/bin) → action k-means → latent visual centroid |
| Optimizer | Adam |

---

## Missing Baseline

Black-box PPO reward on this environment (with shaped rewards) has not yet been evaluated. This number is needed to compute the "% of oracle performance" metric from the proposal. A standalone evaluation script should be run to establish this ceiling.
