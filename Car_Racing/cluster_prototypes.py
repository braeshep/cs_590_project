"""
cluster_prototypes.py
=====================
Runs k-means clustering in 3-D action space (steer, accel, brake) for
k = 2, 4, 6, 8.  For each k we save:

  1. CENTROID PROTOTYPE  – the single frame whose action vector is closest
                           to the cluster centroid.  Good for a fixed,
                           deterministic prototype set.

  2. SAMPLE POOL         – N_SAMPLES frames drawn from each cluster with
                           probability proportional to 1/distance_to_centroid
                           (closer = more likely).  Use these to randomly
                           sample prototypes during training of your simpler
                           model, giving it some variability.

Outputs (one folder per k):
    prototypes/
      k2/
        cluster_0_centroid.png
        cluster_0_centroid_action.npy   # [steer, accel, brake]
        cluster_0_samples/
          sample_00.png
          sample_00_action.npy
          ...
        cluster_1_centroid.png
        ...
      k4/  ...
      k6/  ...
      k8/  ...
    prototypes/
      kmeans_results.npz   -- labels, centroids, distances for all k values
      cluster_summary.txt  -- human-readable summary

USAGE
-----
    python cluster_prototypes.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless – no display needed
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE   = "prototype_data.npz"
OUT_ROOT    = Path("prototypes")
K_VALUES    = [2, 4, 6, 8]
N_SAMPLES   = 10          # random samples to draw per cluster
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
d       = np.load(DATA_FILE)
frames  = d["frames"]    # (N, 96, 96, 3) uint8
latents = d["latents"]   # (N, 256)       float32
actions = d["actions"]   # (N, 3)         float32  [steer, accel, brake]
rewards = d["rewards"]   # (N,)           float32
N       = len(frames)
print(f"  {N:,} timesteps loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Normalise action space for clustering
# (steer is [-1,1], accel/brake are [0,1] — standardise so no axis dominates)
# ─────────────────────────────────────────────────────────────────────────────
scaler         = StandardScaler()
actions_scaled = scaler.fit_transform(actions)   # (N, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_frame(frame_rgb: np.ndarray, path: Path):
    """Save a single 96x96 RGB frame as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def distance_weighted_sample(indices, distances, n, rng):
    """
    Sample n indices from `indices` with probability proportional to
    1 / (distance + eps).  Closer points are more likely.
    Returns an array of sampled indices (with replacement if needed).
    """
    eps    = 1e-6
    weights = 1.0 / (distances[indices] + eps)
    weights = weights / weights.sum()
    replace = len(indices) < n
    chosen  = rng.choice(indices, size=n, replace=replace, p=weights)
    return chosen


def plot_3d_clusters(actions_raw, labels, centroids_raw, k, save_path):
    """
    3-D scatter of action space coloured by cluster label.
    centroids_raw: centroid positions in original (un-scaled) action space.
    """
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i / max(k - 1, 1)) for i in range(k)]

    # Subsample for speed if huge
    idx = np.random.choice(len(actions_raw), min(5000, len(actions_raw)), replace=False)

    for ki in range(k):
        mask = labels[idx] == ki
        pts  = actions_raw[idx][mask]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[colors[ki]], s=6, alpha=0.4, label=f"Cluster {ki}")

    # Plot centroids as large stars
    for ki in range(k):
        c = centroids_raw[ki]
        ax.scatter(*c, c=[colors[ki]], s=200, marker="*",
                   edgecolors="black", linewidths=0.5, zorder=10)

    ax.set_xlabel("Steering")
    ax.set_ylabel("Acceleration")
    ax.set_zlabel("Brake")
    ax.set_title(f"K-Means k={k}  |  3-D Action Space")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"    Saved 3-D plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main clustering loop
# ─────────────────────────────────────────────────────────────────────────────
rng          = np.random.default_rng(RANDOM_SEED)
OUT_ROOT.mkdir(exist_ok=True)
summary_lines = []
all_results   = {}   # for npz save

for k in K_VALUES:
    print(f"\n{'='*55}")
    print(f"  K-Means  k = {k}")
    print(f"{'='*55}")

    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(actions_scaled)          # (N,)

    # Centroid positions back in original action space
    centroids_scaled = km.cluster_centers_           # (k, 3)
    centroids_raw    = scaler.inverse_transform(centroids_scaled)

    # Distance of every point to ITS cluster centroid (in scaled space)
    distances = np.linalg.norm(
        actions_scaled - centroids_scaled[labels], axis=1
    )                                                # (N,)

    k_dir = OUT_ROOT / f"k{k}"
    k_dir.mkdir(exist_ok=True)

    summary_lines.append(f"\n{'='*55}")
    summary_lines.append(f"k = {k}")
    summary_lines.append(f"{'='*55}")

    centroid_frame_indices = []

    for ki in range(k):
        cluster_mask    = labels == ki
        cluster_indices = np.where(cluster_mask)[0]
        cluster_dists   = distances[cluster_indices]
        centroid_raw    = centroids_raw[ki]

        n_pts = len(cluster_indices)
        print(f"  Cluster {ki}: {n_pts:,} points  |  "
              f"centroid steer={centroid_raw[0]:.3f}  "
              f"accel={centroid_raw[1]:.3f}  "
              f"brake={centroid_raw[2]:.3f}")

        summary_lines.append(
            f"  Cluster {ki:2d}: n={n_pts:5d}  "
            f"steer={centroid_raw[0]:+.3f}  "
            f"accel={centroid_raw[1]:.3f}  "
            f"brake={centroid_raw[2]:.3f}"
        )

        # ── 1. CENTROID PROTOTYPE ────────────────────────────────────────────
        # Index within cluster of the point closest to centroid
        closest_in_cluster = np.argmin(cluster_dists)
        closest_global_idx = cluster_indices[closest_in_cluster]
        centroid_frame_indices.append(closest_global_idx)

        centroid_frame  = frames[closest_global_idx]
        centroid_action = actions[closest_global_idx]

        save_frame(centroid_frame,
                   k_dir / f"cluster_{ki}_centroid.png")
        np.save(str(k_dir / f"cluster_{ki}_centroid_action.npy"),
                centroid_action)

        summary_lines.append(
            f"           centroid prototype idx={closest_global_idx}  "
            f"action=[{centroid_action[0]:+.3f}, "
            f"{centroid_action[1]:.3f}, "
            f"{centroid_action[2]:.3f}]"
        )

        # ── 2. SAMPLE POOL ───────────────────────────────────────────────────
        sample_dir = k_dir / f"cluster_{ki}_samples"
        sample_dir.mkdir(exist_ok=True)

        sampled_indices = distance_weighted_sample(
            cluster_indices, distances, N_SAMPLES, rng
        )

        for s_num, s_idx in enumerate(sampled_indices):
            save_frame(frames[s_idx],
                       sample_dir / f"sample_{s_num:02d}.png")
            np.save(str(sample_dir / f"sample_{s_num:02d}_action.npy"),
                    actions[s_idx])

        summary_lines.append(
            f"           {N_SAMPLES} distance-weighted samples saved"
        )

    # ── 3-D cluster plot ─────────────────────────────────────────────────────
    plot_3d_clusters(
        actions, labels, centroids_raw, k,
        OUT_ROOT / f"k{k}" / f"cluster_plot_k{k}.png"
    )

    # Store for npz
    all_results[f"k{k}_labels"]    = labels
    all_results[f"k{k}_centroids"] = centroids_raw
    all_results[f"k{k}_distances"] = distances
    all_results[f"k{k}_centroid_frame_indices"] = np.array(centroid_frame_indices)

# ─────────────────────────────────────────────────────────────────────────────
# Save combined results
# ─────────────────────────────────────────────────────────────────────────────
np.savez_compressed(str(OUT_ROOT / "kmeans_results.npz"), **all_results)
print(f"\nSaved combined results → {OUT_ROOT / 'kmeans_results.npz'}")

summary_text = "\n".join(summary_lines)
summary_path = OUT_ROOT / "cluster_summary.txt"
summary_path.write_text(summary_text)
print(f"Saved summary          → {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Overview plot: centroid prototypes for all k values side by side
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating centroid overview plot...")
fig, axes = plt.subplots(
    len(K_VALUES), max(K_VALUES),
    figsize=(max(K_VALUES) * 2.2, len(K_VALUES) * 2.5)
)

for row, k in enumerate(K_VALUES):
    labels_k    = all_results[f"k{k}_labels"]
    centroids_k = all_results[f"k{k}_centroids"]
    cf_indices  = all_results[f"k{k}_centroid_frame_indices"]

    for col in range(max(K_VALUES)):
        ax = axes[row, col]
        if col < k:
            frame  = frames[cf_indices[col]]
            action = centroids_k[col]
            ax.imshow(frame)
            ax.set_title(
                f"k={k} C{col}\n"
                f"s={action[0]:+.2f}\n"
                f"a={action[1]:.2f} b={action[2]:.2f}",
                fontsize=7
            )
        ax.axis("off")

fig.suptitle("Centroid Prototypes for Each k and Cluster", fontsize=13)
plt.tight_layout()
overview_path = OUT_ROOT / "centroid_overview.png"
plt.savefig(str(overview_path), dpi=150)
plt.close()
print(f"Saved overview plot    → {overview_path}")

print("\nDone! Prototype folder structure:")
print(f"  prototypes/")
for k in K_VALUES:
    print(f"    k{k}/")
    print(f"      cluster_i_centroid.png  +  _action.npy  (x{k})")
    print(f"      cluster_i_samples/      sample_jj.png + _action.npy  (x{k} clusters, {N_SAMPLES} each)")
    print(f"      cluster_plot_k{k}.png")
print(f"    kmeans_results.npz")
print(f"    cluster_summary.txt")
print(f"    centroid_overview.png")
