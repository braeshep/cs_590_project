"""
collect_prototype_data.py
=========================
Runs the pre-trained PPO car-racing agent and collects:
    - raw observation frames  (96x96 RGB)
    - encoder latent vectors  (z = f_enc(frame), 256-d)
    - agent actions           (steering, acceleration, brake)
    - rewards

Architecture recovered from agent_weights.pt:
    conv.0  : Conv2d(4, 32, 8, stride=4)
    conv.2  : Conv2d(32, 64, 4, stride=2)
    conv.4  : Conv2d(64, 64, 3, stride=1)
    actor_fc: Linear(4096, 256)   <-- latent z lives here
    alpha_head: Linear(256, 2)    \\
    beta_head : Linear(256, 2)    /  Beta distribution over [steer, gas/brake]

Output: prototype_data.npz
    frames    (N, 96, 96, 3)  uint8
    latents   (N, 256)        float32
    actions   (N, 3)          float32  [steer, accel, brake]
    rewards   (N,)            float32

USAGE
-----
    python collect_prototype_data.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import cv2
from collections import deque
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path(r"weights\agent_weights.pt")
OUTPUT_FILE     = "prototype_data.npz"
NUM_EPISODES    = 50
MAX_STEPS       = 1000
FRAME_STACK     = 4
FRAME_SIZE      = 96      # greyscale size used during training
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Model — exactly matching the checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class AgentNet(nn.Module):
    """
    Matches the architecture saved in agent_weights.pt exactly.
    conv.0/2/4  →  actor_fc  →  latent z (256-d)
                             →  alpha_head (2)
                             →  beta_head  (2)
    critic is also in the checkpoint but we don't need it for inference.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # conv.0
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # conv.2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv.4
            nn.ReLU(),
        )
        self.actor_fc   = nn.Sequential(nn.Linear(4096, 256))  # actor_fc.0
        self.alpha_head = nn.Sequential(nn.Linear(256, 2))    # alpha_head.0
        self.beta_head  = nn.Sequential(nn.Linear(256, 2))    # beta_head.0

        # critic weights are in the checkpoint; include them so load_state_dict
        # doesn't complain about unexpected keys.
        self.critic = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """x: (B, 4, 84, 84) float32 in [0,1]"""
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)          # (B, 4096)
        z = F.relu(self.actor_fc(h))          # (B, 256)  <- latent
        alpha = F.softplus(self.alpha_head(z)) + 1   # Beta param alpha > 1
        beta  = F.softplus(self.beta_head(z))  + 1   # Beta param beta > 1
        return z, alpha, beta

    @torch.no_grad()
    def get_action_and_latent(self, frame_stack_np):
        """
        frame_stack_np: (4, 84, 84) float32 [0,1]
        Returns:
            action : (3,) float32  [steer, accel, brake]
            latent : (256,) float32
        """
        t = torch.FloatTensor(frame_stack_np).unsqueeze(0).to(DEVICE)
        z, alpha, beta = self.forward(t)

        # Use the mean of the Beta distribution for deterministic action
        # Beta mean = alpha / (alpha + beta),  range [0,1]
        mean = (alpha / (alpha + beta)).squeeze(0).cpu().numpy()  # (2,)

        # mean[0] -> steering:  rescale [0,1] -> [-1,1]
        # mean[1] -> combined gas/brake token; split at 0.5
        steer = float(mean[0]) * 2.0 - 1.0
        steer = float(np.clip(steer, -1.0, 1.0))

        if mean[1] >= 0.5:
            accel = float((mean[1] - 0.5) * 2.0)
            brake = 0.0
        else:
            accel = 0.0
            brake = float((0.5 - mean[1]) * 2.0)

        action = np.array([steer, accel, brake], dtype=np.float32)
        latent = z.squeeze(0).cpu().numpy()
        return action, latent


# ─────────────────────────────────────────────────────────────────────────────
# Frame preprocessing  (matches training pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(rgb_frame: np.ndarray) -> np.ndarray:
    """96x96 RGB uint8  ->  96x96 greyscale float32 [0,1]"""
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)  # (96, 96)
    return frame.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect():
    print(f"Device : {DEVICE}")

    # Load model
    model = AgentNet().to(DEVICE)
    ckpt  = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    env = gym.make("CarRacing-v0")

    all_frames, all_latents, all_actions, all_rewards = [], [], [], []

    for ep in range(NUM_EPISODES):
        obs       = env.reset()
        frame_buf = deque(maxlen=FRAME_STACK)
        first     = preprocess(obs)
        for _ in range(FRAME_STACK):
            frame_buf.append(first)

        ep_frames, ep_latents, ep_actions, ep_rewards = [], [], [], []

        for step in range(MAX_STEPS):
            stack          = np.stack(frame_buf, axis=0)          # (4,84,84)
            action, latent = model.get_action_and_latent(stack)

            next_obs, reward, done, _ = env.step([float(action[0]), float(action[1]), float(action[2])])

            ep_frames.append(obs.astype(np.uint8))
            ep_latents.append(latent)
            ep_actions.append(action)
            ep_rewards.append(float(reward))

            obs = next_obs
            frame_buf.append(preprocess(obs))

            if done:
                break

        total_r = sum(ep_rewards)
        print(f"  Episode {ep+1:3d}/{NUM_EPISODES}  |  "
              f"steps={len(ep_frames):4d}  |  reward={total_r:7.1f}")

        all_frames.append(np.array(ep_frames))
        all_latents.append(np.array(ep_latents))
        all_actions.append(np.array(ep_actions))
        all_rewards.append(np.array(ep_rewards))

    env.close()

    frames  = np.concatenate(all_frames,  axis=0)
    latents = np.concatenate(all_latents, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    print(f"\nTotal samples : {len(frames):,}")
    print(f"  frames      : {frames.shape}  {frames.dtype}")
    print(f"  latents     : {latents.shape}  {latents.dtype}")
    print(f"  actions     : {actions.shape}  {actions.dtype}")
    print(f"  rewards     : {rewards.shape}  {rewards.dtype}")

    np.savez_compressed(OUTPUT_FILE,
                        frames=frames,
                        latents=latents,
                        actions=actions,
                        rewards=rewards)
    print(f"\nSaved -> {OUTPUT_FILE}")

    # Sanity check
    d = np.load(OUTPUT_FILE)
    assert d["frames"].shape == frames.shape
    print("Sanity check passed!")


if __name__ == "__main__":
    collect()
