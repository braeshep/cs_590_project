"""
collect_prototype_data.py
=========================
Runs the pre-trained PPO car-racing agent from the PWNet / deep-racing repo
and collects a dataset of:
    - raw observation frames  (96x96 RGB)
    - encoder latent vectors  (z  = f_enc(frame))
    - agent actions           (steering, acceleration, brake)

The resulting .npz file is ready for downstream clustering / prototype
discovery.

USAGE
-----
1.  Clone both repos side-by-side (or adjust DEEP_RACING_ROOT below):
        git clone https://github.com/EoinKenny/Prototype-Wrapper-Network-ICLR23
        git clone https://github.com/JinayJain/deep-racing

2.  Activate the pwnet virtualenv described in the PWNet README, then run:
        python collect_prototype_data.py

3.  Output: prototype_data.npz  (same directory as this script)
        frames    – (N, 96, 96, 3)   uint8
        latents   – (N, latent_dim)  float32
        actions   – (N, 3)           float32  [steer, accel, brake]
        rewards   – (N,)             float32

NOTES
-----
* gym==0.21.0  (CarRacing-v0, continuous action space)
* The agent uses a frame-stack of 4 greyscale frames internally, but we
  save the raw RGB observation at each step for human readability.
* Set NUM_EPISODES / MAX_STEPS to taste. 50 episodes × 1000 steps ≈ 50 k
  frames, which is usually plenty for clustering.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import gym
import cv2
from pathlib import Path
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# PATHS – adjust if your checkout layout differs
# ─────────────────────────────────────────────────────────────────────────────

# Root of the JinayJain/deep-racing clone (contains model.py + saved weights)
DEEP_RACING_ROOT = Path("../deep-racing")

# Path to the pre-trained actor/critic checkpoint saved by deep-racing
# The file is typically called "actor.pth" or "best_model.pth".
# Check deep-racing/checkpoints/ or wherever you saved it.
CHECKPOINT_PATH  = DEEP_RACING_ROOT / "checkpoints" / "best_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODES   = 50      # number of full episodes to collect
MAX_STEPS      = 1000    # max timesteps per episode (env resets at ~1000 anyway)
FRAME_STACK    = 4       # must match what the agent was trained with
FRAME_SIZE     = 84      # greyscale frame size used internally by the agent
OUTPUT_FILE    = "prototype_data.npz"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Agent / Encoder definition
# Mirrors the architecture used in JinayJain/deep-racing (PPO + CNN encoder)
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Convolutional encoder from JinayJain/deep-racing.
    Input:  (B, FRAME_STACK, FRAME_SIZE, FRAME_SIZE)  float32 in [0,1]
    Output: (B, 256)  latent vector z
    """
    def __init__(self, in_channels: int = FRAME_STACK):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute flat size after conv layers
        dummy = torch.zeros(1, in_channels, FRAME_SIZE, FRAME_SIZE)
        flat  = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ActorHead(nn.Module):
    """
    Gaussian actor head for continuous CarRacing actions.
    Outputs mean of [steer, accel, brake] (tanh-squashed where appropriate).
    """
    def __init__(self, latent_dim: int = 256, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )

    def forward(self, z):
        return self.net(z)


class PWNetBaseAgent(nn.Module):
    """
    Thin wrapper that combines encoder + actor so we can load the checkpoint
    and use the encoder independently.
    """
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.actor   = ActorHead(latent_dim=256, action_dim=3)

    def forward(self, obs_stack):
        z   = self.encoder(obs_stack)
        act = self.actor(z)
        return act, z

    @torch.no_grad()
    def get_action(self, obs_stack_np: np.ndarray):
        """obs_stack_np: (FRAME_STACK, H, W) float32 [0,1]"""
        t   = torch.FloatTensor(obs_stack_np).unsqueeze(0).to(DEVICE)
        act, z = self.forward(t)
        action  = act.squeeze(0).cpu().numpy()
        latent  = z.squeeze(0).cpu().numpy()
        # Clip to valid CarRacing ranges
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1],  0.0, 1.0))
        brake = float(np.clip(action[2],  0.0, 1.0))
        return np.array([steer, accel, brake], dtype=np.float32), latent


# ─────────────────────────────────────────────────────────────────────────────
# Frame pre-processing (matches deep-racing training pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_frame(rgb_frame: np.ndarray) -> np.ndarray:
    """
    96x96 RGB uint8  →  84x84 greyscale float32 [0,1]
    Crops the bottom info panel that CarRacing-v0 adds.
    """
    # Crop bottom dashboard (last 12 rows of the 96x96 frame)
    frame = rgb_frame[:84, :, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)          # → (84, 84)
    frame = frame.astype(np.float32) / 255.0
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main data collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_data():
    print(f"Device: {DEVICE}")

    # ── Load model ──────────────────────────────────────────────────────────
    model = PWNetBaseAgent().to(DEVICE)

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
        # The checkpoint might be a state_dict or a dict with nested keys.
        # Try common formats:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "actor" in ckpt:
            model.actor.load_state_dict(ckpt["actor"])
            model.encoder.load_state_dict(ckpt["encoder"])
        else:
            try:
                model.load_state_dict(ckpt)
            except RuntimeError as e:
                print(f"[WARNING] Could not load checkpoint directly: {e}")
                print("          Running with random weights – useful for testing the pipeline,")
                print("          but you MUST load a proper checkpoint for real data collection.")
    else:
        print(f"[WARNING] Checkpoint not found at {CHECKPOINT_PATH}.")
        print("          Continuing with random weights (for pipeline testing only).")

    model.eval()

    # ── Environment ─────────────────────────────────────────────────────────
    # Use render_mode=None for headless collection (faster)
    env = gym.make("CarRacing-v0")

    all_frames  = []   # raw 96×96 RGB, uint8
    all_latents = []   # 256-d float32
    all_actions = []   # [steer, accel, brake] float32
    all_rewards = []   # scalar float32

    for ep in range(NUM_EPISODES):
        obs       = env.reset()       # (96, 96, 3) uint8
        frame_buf = deque(maxlen=FRAME_STACK)

        # Fill frame buffer with the first frame
        first_frame = preprocess_frame(obs)
        for _ in range(FRAME_STACK):
            frame_buf.append(first_frame)

        ep_frames  = []
        ep_latents = []
        ep_actions = []
        ep_rewards = []

        for step in range(MAX_STEPS):
            obs_stack = np.stack(frame_buf, axis=0)          # (4, 84, 84)
            action, latent = model.get_action(obs_stack)

            next_obs, reward, done, info = env.step(action)

            # Store raw RGB frame (before preprocessing)
            ep_frames.append(obs.astype(np.uint8))
            ep_latents.append(latent)
            ep_actions.append(action)
            ep_rewards.append(float(reward))

            # Advance
            obs = next_obs
            frame_buf.append(preprocess_frame(obs))

            if done:
                break

        ep_reward = sum(ep_rewards)
        print(f"  Episode {ep+1:3d}/{NUM_EPISODES}  |  steps={len(ep_frames):4d}  |  reward={ep_reward:7.1f}")

        all_frames.append(np.array(ep_frames))
        all_latents.append(np.array(ep_latents))
        all_actions.append(np.array(ep_actions))
        all_rewards.append(np.array(ep_rewards))

    env.close()

    # ── Concatenate and save ─────────────────────────────────────────────────
    frames  = np.concatenate(all_frames,  axis=0)   # (N, 96, 96, 3)
    latents = np.concatenate(all_latents, axis=0)   # (N, 256)
    actions = np.concatenate(all_actions, axis=0)   # (N, 3)
    rewards = np.concatenate(all_rewards, axis=0)   # (N,)

    print(f"\nTotal samples collected: {len(frames):,}")
    print(f"  frames  shape : {frames.shape}   dtype={frames.dtype}")
    print(f"  latents shape : {latents.shape}  dtype={latents.dtype}")
    print(f"  actions shape : {actions.shape}  dtype={actions.dtype}")
    print(f"  rewards shape : {rewards.shape}  dtype={rewards.dtype}")

    np.savez_compressed(
        OUTPUT_FILE,
        frames=frames,
        latents=latents,
        actions=actions,
        rewards=rewards,
    )
    print(f"\nSaved → {OUTPUT_FILE}")

    # Quick sanity check
    loaded = np.load(OUTPUT_FILE)
    assert loaded["frames"].shape  == frames.shape
    assert loaded["latents"].shape == latents.shape
    print("Sanity check passed ✓")


if __name__ == "__main__":
    collect_data()
