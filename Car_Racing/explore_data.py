"""
explore_data.py
===============
Quick inspection of the prototype_data.npz produced by collect_prototype_data.py.

Run:
    python explore_data.py

Outputs:
    - summary statistics printed to console
    - sample_frames.png  – a grid of 25 random frames
    - action_distributions.png  – histograms of steer / accel / brake
    - latent_pca.png  – PCA scatter of the 256-d latent space (coloured by action)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

DATA_FILE = "prototype_data.npz"
OUT_DIR   = Path("explore_output")
OUT_DIR.mkdir(exist_ok=True)


def load():
    d = np.load(DATA_FILE)
    return d["frames"], d["latents"], d["actions"], d["rewards"]


def print_summary(frames, latents, actions, rewards):
    print("=" * 55)
    print("DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total timesteps : {len(frames):,}")
    print(f"  frames          : {frames.shape}  dtype={frames.dtype}")
    print(f"  latents         : {latents.shape}  dtype={latents.dtype}")
    print(f"  actions         : {actions.shape}  dtype={actions.dtype}")
    print(f"  rewards         : {rewards.shape}  dtype={rewards.dtype}")
    print()
    print("  Action statistics (steer, accel, brake):")
    labels = ["steer", "accel", "brake"]
    for i, lbl in enumerate(labels):
        a = actions[:, i]
        print(f"    {lbl:6s}  min={a.min():.3f}  max={a.max():.3f}  "
              f"mean={a.mean():.3f}  std={a.std():.3f}")
    print()
    print(f"  Reward  min={rewards.min():.2f}  max={rewards.max():.2f}  "
          f"mean={rewards.mean():.2f}")
    print("=" * 55)


def plot_sample_frames(frames, n=25):
    idx  = np.random.choice(len(frames), n, replace=False)
    cols = 5
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(frames[i])
        ax.axis("off")
    fig.suptitle("25 Random Observations", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "sample_frames.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_action_distributions(actions):
    labels = ["Steering (−1 to 1)", "Acceleration (0 to 1)", "Brake (0 to 1)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, col, lbl in zip(axes, range(3), labels):
        ax.hist(actions[:, col], bins=60, color="steelblue", edgecolor="white")
        ax.set_title(lbl)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.suptitle("Action Distributions Across All Collected Timesteps", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "action_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_latent_pca(latents, actions, max_samples=5000):
    """
    2-D PCA of the encoder latent space.
    Points are coloured by steering value (red=left, blue=right).
    """
    idx = np.random.choice(len(latents), min(max_samples, len(latents)), replace=False)
    z   = latents[idx]
    a   = actions[idx]

    pca    = PCA(n_components=2)
    z_2d   = pca.fit_transform(z)
    steer  = a[:, 0]   # −1 … 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Colour by steering ──────────────────────────────────────────────────
    sc = axes[0].scatter(z_2d[:, 0], z_2d[:, 1],
                         c=steer, cmap="RdBu_r", s=4, alpha=0.5,
                         vmin=-1, vmax=1)
    plt.colorbar(sc, ax=axes[0])
    axes[0].set_title("PCA  |  colour = steering")

    # ── Colour by acceleration ───────────────────────────────────────────────
    sc2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1],
                          c=a[:, 1], cmap="Greens", s=4, alpha=0.5,
                          vmin=0, vmax=1)
    plt.colorbar(sc2, ax=axes[1])
    axes[1].set_title("PCA  |  colour = acceleration")

    # ── Colour by brake ──────────────────────────────────────────────────────
    sc3 = axes[2].scatter(z_2d[:, 0], z_2d[:, 1],
                          c=a[:, 2], cmap="Oranges", s=4, alpha=0.5,
                          vmin=0, vmax=1)
    plt.colorbar(sc3, ax=axes[2])
    axes[2].set_title("PCA  |  colour = brake")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    fig.suptitle("Encoder Latent Space (PCA 2-D projection)", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "latent_pca.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    frames, latents, actions, rewards = load()
    print_summary(frames, latents, actions, rewards)
    plot_sample_frames(frames)
    plot_action_distributions(actions)
    plot_latent_pca(latents, actions)
    print("\nDone – check the explore_output/ folder.")
