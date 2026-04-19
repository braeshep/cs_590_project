"""
cluster_prototypes_grid.py
==========================
Two-stage hierarchical prototype selection with 3D Grid-Based Stratified
Sampling for action balancing:

  Stage 0 — Grid-Based Stratified Sampling: partition the 3-D action space
             (steer, accel, brake) into a NUM_BINS x NUM_BINS x NUM_BINS
             histogram. For each occupied bin, keep at most MAX_SAMPLES_PER_BIN
             frames (random subsample if the bin overflows). This prevents
             K-means from being dominated by the dense "coast straight" region
             without requiring manual threshold tuning.

  Stage 1 — K-means on the 3-D action space (steer, accel, brake),
             using the balanced dataset.

  Stage 2 — For each action cluster, find the VISUAL center by
             computing the mean of the 256-d latent vectors belonging
             to that cluster, then selecting the frame whose latent is
             closest to that mean.

This bridges action-space interpretability with visual representativeness:
prototypes are chosen because they look like the most "normal" version of
a particular maneuver, not just because their action vector happens to be
nearest the centroid.

Outputs (mirrors cluster_prototypes_hybrid.py layout):
    prototypes_grid/
      k{k}/
        cluster_{i}_centroid.png
        cluster_{i}_centroid_action.npy
        cluster_{i}_samples/
          sample_{j:02d}.png
          sample_{j:02d}_action.npy
        cluster_plot_k{k}.png         (3-D action scatter)
        cluster_plot_k{k}_pca.png     (2-D latent PCA, coloured by cluster)
        cluster_plot_k{k}_actions.png (box-plots of action distributions)
      kmeans_grid_results.npz
      cluster_summary.txt
      centroid_overview.png

USAGE
-----
    python cluster_prototypes_grid.py
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
# SETTINGS  (all tunable)
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE = "prototype_data.npz"
OUT_ROOT = Path("prototypes_grid")
K_VALUES = [2, 4, 6, 8]
N_SAMPLES = 10  # distance-weighted samples saved per cluster
RANDOM_SEED = 42

# --- Downsampling parameters -------------------------------------------------
NUM_BINS = 20             # bins per axis in the 3-D action histogram
MAX_SAMPLES_PER_BIN = 200 # max frames kept from any single bin

# --- Action normalisation before clustering ----------------------------------
# Set True to standardise actions so no axis dominates k-means.
# Currently left False to match the existing scripts; flip here to experiment.
NORMALISE_ACTIONS = True

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
d = np.load(DATA_FILE)
frames = d["frames"]  # (N, 96, 96, 3) uint8
latents = d["latents"]  # (N, 256)       float32
actions = d["actions"]  # (N, 3)         float32  [steer, accel, brake]
rewards = d["rewards"]  # (N,)           float32
N = len(frames)
print(f"  {N:,} timesteps loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — 3D Grid-Based Stratified Sampling
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

# Compute bin edges from the actual data range on each axis
_, bin_edges = np.histogramdd(actions, bins=NUM_BINS)

# Assign each frame to a bin index (0-based) on each axis.
# np.digitize returns 1-based indices; clip to [0, NUM_BINS-1].
bin_indices = np.stack([
    np.clip(np.digitize(actions[:, ax], bin_edges[ax]) - 1, 0, NUM_BINS - 1)
    for ax in range(3)
], axis=1)  # (N, 3)

# Group frame indices by their 3-D bin address
from collections import defaultdict
bin_map = defaultdict(list)
for i, (bx, by, bz) in enumerate(bin_indices):
    bin_map[(bx, by, bz)].append(i)

total_bins     = NUM_BINS ** 3
occupied_bins  = len(bin_map)
keep_indices   = []

for bin_key, idx_list in bin_map.items():
    if len(idx_list) <= MAX_SAMPLES_PER_BIN:
        keep_indices.extend(idx_list)
    else:
        sampled = rng.choice(idx_list, size=MAX_SAMPLES_PER_BIN, replace=False)
        keep_indices.extend(sampled.tolist())

keep_idx = np.sort(np.array(keep_indices, dtype=np.int64))

print(f"\nGrid-Based Stratified Sampling ({NUM_BINS} bins/axis, max {MAX_SAMPLES_PER_BIN}/bin):")
print(f"  Occupied bins : {occupied_bins:,} / {total_bins:,} total "
      f"({100 * occupied_bins / total_bins:.1f}%)")
print(f"  Dataset       : {N:,}  →  {len(keep_idx):,} after balancing "
      f"(kept {100 * len(keep_idx) / N:.1f}%)")

frames_ds  = frames[keep_idx]
latents_ds = latents[keep_idx]
actions_ds = actions[keep_idx]
rewards_ds = rewards[keep_idx]
N_ds       = len(keep_idx)

# ─────────────────────────────────────────────────────────────────────────────
# Prepare latent space for Stage 2 (visual centroid search)
# We scale latents once here; Stage 1 runs on (optionally scaled) actions.
# ─────────────────────────────────────────────────────────────────────────────
print("\nScaling latent vectors for visual-centroid search...")
lat_scaler = StandardScaler()
latents_scaled = lat_scaler.fit_transform(latents_ds)  # (N_ds, 256)

print("Computing PCA (2-D) for visualisation...")
pca = PCA(n_components=2, random_state=RANDOM_SEED)
lat_2d = pca.fit_transform(latents_scaled)  # (N_ds, 2)
print(
    f"  PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
    f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
)

# ─────────────────────────────────────────────────────────────────────────────
# Optionally scale actions for Stage 1 clustering
# ─────────────────────────────────────────────────────────────────────────────
if NORMALISE_ACTIONS:
    act_scaler = StandardScaler()
    actions_fit = act_scaler.fit_transform(actions_ds)
    print("Action normalisation: ON")
else:
    actions_fit = actions_ds
    act_scaler = None
    print("Action normalisation: OFF (raw action values used for clustering)")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_frame(frame_rgb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def distance_weighted_sample(cluster_indices, distances, n, rng):
    eps = 1e-6
    weights = 1.0 / (distances[cluster_indices] + eps)
    weights = weights / weights.sum()
    replace = len(cluster_indices) < n
    return rng.choice(cluster_indices, size=n, replace=replace, p=weights)


def plot_3d_clusters(actions_raw, labels, centroids_raw, k, save_path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / max(k - 1, 1)) for i in range(k)]
    idx = rng.choice(len(actions_raw), min(5000, len(actions_raw)), replace=False)
    for ki in range(k):
        mask = labels[idx] == ki
        pts = actions_raw[idx][mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=[colors[ki]],
            s=6,
            alpha=0.4,
            label=f"Cluster {ki}",
        )
    for ki in range(k):
        c = centroids_raw[ki]
        ax.scatter(
            *c,
            c=[colors[ki]],
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )
    ax.set_xlabel("Steering")
    ax.set_ylabel("Acceleration")
    ax.set_zlabel("Brake")
    ax.set_title(f"K-Means k={k}  |  Action Space (grid-balanced)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"    Saved 3-D action plot  → {save_path}")


def plot_pca_clusters(lat_2d, labels, k, centroid_2d, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    idx = rng.choice(len(lat_2d), min(6000, len(lat_2d)), replace=False)
    for ki in range(k):
        mask = labels[idx] == ki
        pts = lat_2d[idx][mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=[cmap(ki / max(k - 1, 1))],
            s=6,
            alpha=0.4,
            label=f"Cluster {ki}",
        )
    for ki in range(k):
        ax.scatter(
            *centroid_2d[ki],
            c=[cmap(ki / max(k - 1, 1))],
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            zorder=10,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Latent Space (visual centroid)  |  k={k}")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"    Saved PCA plot         → {save_path}")


def plot_action_breakdown(actions_raw, labels, k, centroids_action, save_path):
    action_names = ["Steering", "Acceleration", "Brake"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(ki / max(k - 1, 1)) for ki in range(k)]
    for col, name in enumerate(action_names):
        ax = axes[col]
        data = [actions_raw[labels == ki, col] for ki in range(k)]
        bp = ax.boxplot(data, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, k + 1))
        ax.set_xticklabels([f"C{ki}" for ki in range(k)])
        ax.set_title(name)
        ax.set_xlabel("Cluster")
        for ki in range(k):
            ax.text(
                ki + 1,
                ax.get_ylim()[1] * 0.97,
                f"{centroids_action[ki, col]:+.2f}",
                ha="center",
                va="top",
                fontsize=7,
            )
    fig.suptitle(f"Action Distribution per Cluster  |  k={k}", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"    Saved action breakdown → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main clustering loop
# ─────────────────────────────────────────────────────────────────────────────
OUT_ROOT.mkdir(exist_ok=True)
summary_lines = []
all_results = {}

for k in K_VALUES:
    print(f"\n{'='*65}")
    print(f"  Grid  k = {k}")
    print(f"{'='*65}")

    # ── Stage 1: K-means on action space ────────────────────────────────────
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(actions_fit)  # (N_ds,)

    centroids_fit = km.cluster_centers_  # (k, 3) in fitting space

    # Recover centroid positions in original action space for display
    if NORMALISE_ACTIONS:
        centroids_action = act_scaler.inverse_transform(centroids_fit)
    else:
        centroids_action = centroids_fit.copy()  # already raw

    # Action-space distance to centroid (for sample weighting)
    act_dists = np.linalg.norm(actions_fit - centroids_fit[labels], axis=1)  # (N_ds,)

    k_dir = OUT_ROOT / f"k{k}"
    k_dir.mkdir(exist_ok=True)

    summary_lines.append(f"\n{'='*65}")
    summary_lines.append(f"k = {k}  (grid-balanced: action cluster → visual centroid)")
    summary_lines.append(f"{'='*65}")

    centroid_frame_indices = []
    visual_centroid_2d = []  # for PCA plot

    for ki in range(k):
        cluster_mask = labels == ki
        cluster_indices = np.where(cluster_mask)[0]
        centroid_raw = centroids_action[ki]

        n_pts = len(cluster_indices)

        # ── Stage 2: Visual centroid in latent space ─────────────────────────
        cluster_latents = latents_scaled[cluster_indices]  # (n_pts, 256)
        mean_latent = cluster_latents.mean(axis=0)  # (256,)

        latent_dists_in_cluster = np.linalg.norm(
            cluster_latents - mean_latent, axis=1
        )  # (n_pts,)
        closest_in_cluster = np.argmin(latent_dists_in_cluster)
        closest_global_idx = cluster_indices[closest_in_cluster]
        centroid_frame_indices.append(closest_global_idx)

        # PCA projection of the mean latent (for the PCA plot centroid star)
        visual_centroid_2d.append(pca.transform(mean_latent[None])[0])

        proto_action = actions_ds[closest_global_idx]

        print(
            f"  Cluster {ki:2d}: n={n_pts:5d}  "
            f"action centroid steer={centroid_raw[0]:+.3f}  "
            f"accel={centroid_raw[1]:.3f}  "
            f"brake={centroid_raw[2]:.3f}  |  "
            f"visual prototype idx={closest_global_idx}  "
            f"action=[{proto_action[0]:+.3f}, {proto_action[1]:.3f}, {proto_action[2]:.3f}]"
        )

        summary_lines.append(
            f"  Cluster {ki:2d}: n={n_pts:5d}  "
            f"act_centroid=[{centroid_raw[0]:+.3f}, {centroid_raw[1]:.3f}, {centroid_raw[2]:.3f}]  "
            f"visual_proto_idx={closest_global_idx}  "
            f"proto_action=[{proto_action[0]:+.3f}, {proto_action[1]:.3f}, {proto_action[2]:.3f}]"
        )

        save_frame(frames_ds[closest_global_idx], k_dir / f"cluster_{ki}_centroid.png")
        np.save(str(k_dir / f"cluster_{ki}_centroid_action.npy"), proto_action)

        # ── Sample pool (distance-weighted in action space) ──────────────────
        sample_dir = k_dir / f"cluster_{ki}_samples"
        sample_dir.mkdir(exist_ok=True)
        sampled = distance_weighted_sample(cluster_indices, act_dists, N_SAMPLES, rng)
        for s_num, s_idx in enumerate(sampled):
            save_frame(frames_ds[s_idx], sample_dir / f"sample_{s_num:02d}.png")
            np.save(
                str(sample_dir / f"sample_{s_num:02d}_action.npy"), actions_ds[s_idx]
            )

        summary_lines.append(f"           {N_SAMPLES} distance-weighted samples saved")

    visual_centroid_2d = np.array(visual_centroid_2d)  # (k, 2)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_3d_clusters(
        actions_ds, labels, centroids_action, k, k_dir / f"cluster_plot_k{k}.png"
    )
    plot_pca_clusters(
        lat_2d, labels, k, visual_centroid_2d, k_dir / f"cluster_plot_k{k}_pca.png"
    )
    plot_action_breakdown(
        actions_ds,
        labels,
        k,
        centroids_action,
        k_dir / f"cluster_plot_k{k}_actions.png",
    )

    all_results[f"k{k}_labels"] = labels
    all_results[f"k{k}_centroids_action"] = centroids_action
    all_results[f"k{k}_act_distances"] = act_dists
    all_results[f"k{k}_centroid_frame_indices"] = np.array(centroid_frame_indices)
    all_results[f"k{k}_keep_idx"] = keep_idx

# ─────────────────────────────────────────────────────────────────────────────
# Save combined results
# ─────────────────────────────────────────────────────────────────────────────
np.savez_compressed(str(OUT_ROOT / "kmeans_grid_results.npz"), **all_results)
print(f"\nSaved combined results → {OUT_ROOT / 'kmeans_grid_results.npz'}")

summary_path = OUT_ROOT / "cluster_summary.txt"
summary_path.write_text("\n".join(summary_lines))
print(f"Saved summary          → {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Centroid overview grid
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating centroid overview grid...")
fig, axes = plt.subplots(
    len(K_VALUES), max(K_VALUES), figsize=(max(K_VALUES) * 2.4, len(K_VALUES) * 2.8)
)

for row, k in enumerate(K_VALUES):
    cf_indices = all_results[f"k{k}_centroid_frame_indices"]
    centroids_a = all_results[f"k{k}_centroids_action"]
    for col in range(max(K_VALUES)):
        ax = axes[row, col]
        if col < k:
            ax.imshow(frames_ds[cf_indices[col]])
            a = centroids_a[col]
            ax.set_title(
                f"k={k} C{col}\n" f"s={a[0]:+.2f}\n" f"a={a[1]:.2f} b={a[2]:.2f}",
                fontsize=7,
            )
        ax.axis("off")

fig.suptitle(
    "Grid-Balanced Prototypes  (action cluster → visual center)", fontsize=12
)
plt.tight_layout()
overview_path = OUT_ROOT / "centroid_overview.png"
plt.savefig(str(overview_path), dpi=150)
plt.close()
print(f"Saved overview         → {overview_path}")

print("\nDone!")
print(f"\nFolder structure:")
print(f"  prototypes_grid/")
for k in K_VALUES:
    print(f"    k{k}/")
    print(f"      cluster_i_centroid.png + _action.npy   (x{k})")
    print(f"      cluster_i_samples/  (x{N_SAMPLES} per cluster)")
    print(f"      cluster_plot_k{k}.png / _pca.png / _actions.png")
print(f"    kmeans_grid_results.npz")
print(f"    cluster_summary.txt")
print(f"    centroid_overview.png")
