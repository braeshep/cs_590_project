"""
cluster_prototypes_latent.py
============================
Runs k-means clustering in the 256-d ENCODER LATENT SPACE for
k = 2, 4, 6, 8.

Why latent space instead of action space?
  - The encoder learned a rich visual representation of the scene.
  - Left-turn and right-turn frames look visually very different even
    if their action magnitudes happen to be similar.
  - Clustering in latent space finds prototypes that differ in what the
    agent *sees*, not just what it *does* -- giving more visually
    distinct and balanced prototypes.

For each cluster we save:
  1. CENTROID PROTOTYPE  – frame whose latent vector is closest to the
                           cluster centroid.
  2. SAMPLE POOL         – N_SAMPLES frames drawn with probability
                           proportional to 1/distance_to_centroid.

We also print a per-cluster action breakdown so you can verify that
left/right turns are being separated correctly.

Outputs:
    prototypes_latent/
      k{k}/
        cluster_{i}_centroid.png
        cluster_{i}_centroid_action.npy
        cluster_{i}_samples/
          sample_{j:02d}.png
          sample_{j:02d}_action.npy
      kmeans_latent_results.npz
      cluster_summary.txt
      centroid_overview.png
      k{k}/cluster_plot_k{k}_pca.png   (2-D PCA coloured by cluster)
      k{k}/cluster_plot_k{k}_actions.png  (action breakdown per cluster)

USAGE
-----
    python cluster_prototypes_latent.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE   = "prototype_data.npz"
OUT_ROOT    = Path("prototypes_latent")
K_VALUES    = [2, 4, 6, 8]
N_SAMPLES   = 10
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
d       = np.load(DATA_FILE)
frames  = d["frames"]    # (N, 96, 96, 3) uint8
latents = d["latents"]   # (N, 256)
actions = d["actions"]   # (N, 3)  [steer, accel, brake]
rewards = d["rewards"]
N       = len(frames)
print(f"  {N:,} timesteps loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Normalise latent space before clustering
# ─────────────────────────────────────────────────────────────────────────────
print("Scaling latent vectors...")
scaler         = StandardScaler()
latents_scaled = scaler.fit_transform(latents)   # (N, 256)

# Pre-compute PCA for visualisation (done once, reused for all k)
print("Computing PCA (2-D) for visualisation...")
pca    = PCA(n_components=2, random_state=RANDOM_SEED)
lat_2d = pca.fit_transform(latents_scaled)       # (N, 2)
print(f"  PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_frame(frame_rgb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def distance_weighted_sample(cluster_indices, distances, n, rng):
    eps     = 1e-6
    weights = 1.0 / (distances[cluster_indices] + eps)
    weights = weights / weights.sum()
    replace = len(cluster_indices) < n
    return rng.choice(cluster_indices, size=n, replace=replace, p=weights)


def plot_pca_clusters(lat_2d, labels, k, centroid_2d, save_path):
    """2-D PCA scatter coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap    = plt.get_cmap("tab10")

    idx = np.random.choice(len(lat_2d), min(6000, len(lat_2d)), replace=False)
    for ki in range(k):
        mask = labels[idx] == ki
        pts  = lat_2d[idx][mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[cmap(ki / max(k-1, 1))],
                   s=6, alpha=0.4, label=f"Cluster {ki}")

    for ki in range(k):
        ax.scatter(*centroid_2d[ki], c=[cmap(ki / max(k-1, 1))],
                   s=250, marker="*", edgecolors="black",
                   linewidths=0.6, zorder=10)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Latent Space Clusters  |  k={k}")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"    Saved PCA plot       → {save_path}")


def plot_action_breakdown(actions, labels, k, centroids_raw, save_path):
    """
    For each cluster: box plots of steer / accel / brake.
    Helps verify left/right turns are being separated.
    """
    action_names = ["Steering", "Acceleration", "Brake"]
    fig, axes    = plt.subplots(1, 3, figsize=(14, 4))
    cmap         = plt.get_cmap("tab10")
    colors       = [cmap(ki / max(k-1, 1)) for ki in range(k)]

    for col, name in enumerate(action_names):
        ax   = axes[col]
        data = [actions[labels == ki, col] for ki in range(k)]
        bp   = ax.boxplot(data, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, k + 1))
        ax.set_xticklabels([f"C{ki}" for ki in range(k)])
        ax.set_title(name)
        ax.set_xlabel("Cluster")

        # Annotate with centroid value
        for ki in range(k):
            ax.text(ki + 1, ax.get_ylim()[1] * 0.97,
                    f"{centroids_raw[ki, col]:+.2f}",
                    ha="center", va="top", fontsize=7, color="black")

    fig.suptitle(f"Action Distribution per Cluster  |  k={k}", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"    Saved action plot    → {save_path}")



def plot_3d_action_clusters(actions, labels, k, centroids_action, save_path):
    """
    3-D scatter of action space coloured by latent cluster label.
    elev=25, azim=-60 makes the acceleration axis tall and readable.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    idx = np.random.choice(len(actions), min(5000, len(actions)), replace=False)
    for ki in range(k):
        mask = labels[idx] == ki
        pts  = actions[idx][mask]
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
                   c=[cmap(ki / max(k-1, 1))], s=8, alpha=0.4, label=f"Cluster {ki}")
    for ki in range(k):
        c = centroids_action[ki]
        ax.scatter(c[0], c[2], c[1], c=[cmap(ki / max(k-1, 1))],
                   s=250, marker="*", edgecolors="black", linewidths=0.6, zorder=10)
    ax.view_init(elev=25, azim=-60)
    ax.set_xlabel("Steering", labelpad=8)
    ax.set_ylabel("Brake", labelpad=8)
    ax.set_zlabel("Acceleration", labelpad=8)
    ax.set_title(f"Action Space  |  Latent Clusters  |  k={k}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"    Saved 3-D action plot -> {save_path}")

def print_cluster_stats(ki, cluster_indices, actions, centroid_raw):
    """Print left/right/straight breakdown for a cluster."""
    a      = actions[cluster_indices]
    steer  = a[:, 0]
    n      = len(cluster_indices)
    n_left  = (steer < -0.05).sum()
    n_right = (steer >  0.05).sum()
    n_str   = n - n_left - n_right
    print(f"    Cluster {ki:2d}: n={n:5d}  "
          f"steer={centroid_raw[0]:+.3f}  "
          f"accel={centroid_raw[1]:.3f}  "
          f"brake={centroid_raw[2]:.3f}  |  "
          f"left={n_left:4d} ({100*n_left/n:.0f}%)  "
          f"right={n_right:4d} ({100*n_right/n:.0f}%)  "
          f"straight={n_str:4d} ({100*n_str/n:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Main clustering loop
# ─────────────────────────────────────────────────────────────────────────────
OUT_ROOT.mkdir(exist_ok=True)
rng           = np.random.default_rng(RANDOM_SEED)
summary_lines = []
all_results   = {}

for k in K_VALUES:
    print(f"\n{'='*65}")
    print(f"  K-Means in latent space  |  k = {k}")
    print(f"{'='*65}")

    km     = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(latents_scaled)          # (N,)

    # Cluster centroids in scaled latent space
    centroids_scaled = km.cluster_centers_           # (k, 256)

    # Map centroids back through PCA for plotting
    centroid_2d = pca.transform(centroids_scaled)    # (k, 2)

    # Compute centroids in action space (mean action per cluster)
    centroids_action = np.array(
        [actions[labels == ki].mean(axis=0) for ki in range(k)]
    )                                                # (k, 3)

    # Distance of every point to its cluster centroid (scaled latent space)
    distances = np.linalg.norm(
        latents_scaled - centroids_scaled[labels], axis=1
    )                                                # (N,)

    k_dir = OUT_ROOT / f"k{k}"
    k_dir.mkdir(exist_ok=True)

    summary_lines.append(f"\n{'='*65}")
    summary_lines.append(f"k = {k}  (latent-space clustering)")
    summary_lines.append(f"{'='*65}")

    centroid_frame_indices = []

    for ki in range(k):
        cluster_indices = np.where(labels == ki)[0]
        cluster_dists   = distances[cluster_indices]
        centroid_action = centroids_action[ki]

        print_cluster_stats(ki, cluster_indices, actions, centroid_action)

        summary_lines.append(
            f"  Cluster {ki:2d}: n={len(cluster_indices):5d}  "
            f"steer={centroid_action[0]:+.3f}  "
            f"accel={centroid_action[1]:.3f}  "
            f"brake={centroid_action[2]:.3f}"
        )

        # ── 1. Centroid prototype ────────────────────────────────────────────
        closest_in_cluster = np.argmin(cluster_dists)
        closest_global_idx = cluster_indices[closest_in_cluster]
        centroid_frame_indices.append(closest_global_idx)

        save_frame(frames[closest_global_idx],
                   k_dir / f"cluster_{ki}_centroid.png")
        np.save(str(k_dir / f"cluster_{ki}_centroid_action.npy"),
                actions[closest_global_idx])

        act = actions[closest_global_idx]
        summary_lines.append(
            f"           centroid idx={closest_global_idx}  "
            f"action=[{act[0]:+.3f}, {act[1]:.3f}, {act[2]:.3f}]"
        )

        # ── 2. Sample pool ───────────────────────────────────────────────────
        sample_dir = k_dir / f"cluster_{ki}_samples"
        sample_dir.mkdir(exist_ok=True)
        sampled    = distance_weighted_sample(
            cluster_indices, distances, N_SAMPLES, rng
        )
        for s_num, s_idx in enumerate(sampled):
            save_frame(frames[s_idx],
                       sample_dir / f"sample_{s_num:02d}.png")
            np.save(str(sample_dir / f"sample_{s_num:02d}_action.npy"),
                    actions[s_idx])

        summary_lines.append(
            f"           {N_SAMPLES} distance-weighted samples saved"
        )

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_pca_clusters(
        lat_2d, labels, k, centroid_2d,
        k_dir / f"cluster_plot_k{k}_pca.png"
    )
    plot_action_breakdown(
        actions, labels, k, centroids_action,
        k_dir / f"cluster_plot_k{k}_actions.png"
    )
    plot_3d_action_clusters(
        actions, labels, k, centroids_action,
        k_dir / f"cluster_plot_k{k}_3d_actions.png"
    )

    # Store
    all_results[f"k{k}_labels"]               = labels
    all_results[f"k{k}_centroids_action"]      = centroids_action
    all_results[f"k{k}_distances"]             = distances
    all_results[f"k{k}_centroid_frame_indices"]= np.array(centroid_frame_indices)

# ─────────────────────────────────────────────────────────────────────────────
# Save combined results
# ─────────────────────────────────────────────────────────────────────────────
np.savez_compressed(str(OUT_ROOT / "kmeans_latent_results.npz"), **all_results)
print(f"\nSaved combined results → {OUT_ROOT / 'kmeans_latent_results.npz'}")

summary_path = OUT_ROOT / "cluster_summary.txt"
summary_path.write_text("\n".join(summary_lines))
print(f"Saved summary          → {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Centroid overview grid
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating centroid overview grid...")
fig, axes = plt.subplots(
    len(K_VALUES), max(K_VALUES),
    figsize=(max(K_VALUES) * 2.4, len(K_VALUES) * 2.8)
)

for row, k in enumerate(K_VALUES):
    cf_indices  = all_results[f"k{k}_centroid_frame_indices"]
    centroids_a = all_results[f"k{k}_centroids_action"]
    for col in range(max(K_VALUES)):
        ax = axes[row, col]
        if col < k:
            ax.imshow(frames[cf_indices[col]])
            a = centroids_a[col]
            ax.set_title(
                f"k={k} C{col}\n"
                f"s={a[0]:+.2f}\n"
                f"a={a[1]:.2f} b={a[2]:.2f}",
                fontsize=7
            )
        ax.axis("off")

fig.suptitle("Centroid Prototypes — Latent Space Clustering", fontsize=13)
plt.tight_layout()
overview_path = OUT_ROOT / "centroid_overview.png"
plt.savefig(str(overview_path), dpi=150)
plt.close()
print(f"Saved overview         → {overview_path}")

print("\nDone!")
print(f"\nFolder structure:")
print(f"  prototypes_latent/")
for k in K_VALUES:
    print(f"    k{k}/")
    print(f"      cluster_i_centroid.png + _action.npy   (x{k})")
    print(f"      cluster_i_samples/  sample_jj.png + _action.npy  ({N_SAMPLES} each)")
    print(f"      cluster_plot_k{k}_pca.png")
    print(f"      cluster_plot_k{k}_actions.png")
print(f"    kmeans_latent_results.npz")
print(f"    cluster_summary.txt")
print(f"    centroid_overview.png")
