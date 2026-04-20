"""
visualize_pwnet.py
==================
Generates a video of the k=4 PW-Net driving in CarRacing-v2 with a
real-time prototype activation visualization.

Left panel : 2×2 grid of prototype images, each showing:
             - The prototype frame (what the agent saw at that "case")
             - A similarity bar (how much the current state resembles this prototype)
             - The prototype's action vector (steer / accel / brake)
             The most-active prototype is highlighted in gold.

Right panel: Live game frame from the environment renderer.

Output: results/pwnet_visualization_k4.mp4

USAGE
-----
    python visualize_pwnet.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import gym
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from pathlib import Path

from games.carracing import RacingNet

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
K               = 4
CHECKPOINT      = f"weights/pw_net_k{K}.pth"
PROTO_DIR       = Path(f"prototypes_grid/k{K}")
AGENT_WEIGHTS   = "weights/agent_weights.pt"
OUT_FILE        = Path("results") / f"pwnet_visualization_k{K}.mp4"

FRAME_STACK         = 4
MAX_STEPS           = 1000
FPS                 = 30
NO_REWARD_PATIENCE  = 50   # end episode if no positive reward for this many steps

LATENT_SIZE     = 256
PROTOTYPE_SIZE  = 50
NUM_CLASSES     = 3
DEVICE          = "cpu"

# ── Video layout (pixels) ─────────────────────────────────────────────────────
PANEL_W  = 430    # left prototype panel
GAME_W   = 600    # right game panel (CarRacing renders at 600px wide)
VIDEO_H  = 700
VIDEO_W  = PANEL_W + GAME_W

HEADER_H = 46     # top banner height on each panel
FOOTER_H = 70     # bottom output readout on left panel
GRID_PAD = 10     # gap between cells and panel edges

CELL_W = (PANEL_W - 3 * GRID_PAD) // 2
CELL_H = (VIDEO_H - HEADER_H - FOOTER_H - 3 * GRID_PAD) // 2
PROTO_IMG_SIZE = min(CELL_W - 16, CELL_H - 76)   # image fits inside cell

# ── Colour palette (RGB) ──────────────────────────────────────────────────────
C_BG        = (22, 22, 32)
C_PANEL_DIV = (55, 55, 75)
C_CELL      = (42, 42, 58)
C_BORDER    = (70, 70, 95)
C_ACTIVE    = (240, 200, 50)    # gold highlight for most-active prototype
C_BAR_BG    = (60, 60, 80)
C_BAR_FG    = (75, 195, 115)
C_TEXT      = (205, 205, 218)
C_HEADER    = (155, 180, 255)
C_LABEL     = (140, 140, 160)


# ─────────────────────────────────────────────────────────────────────────────
# PWNet  (must match run_pwnet_sweep.py architecture exactly for checkpoint load)
# ─────────────────────────────────────────────────────────────────────────────

class ListModule(object):
    def __init__(self, module, prefix):
        self.module     = module
        self.prefix     = prefix
        self.num_module = 0

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError("Not a Module")
        self.module.add_module(self.prefix + str(self.num_module), new_module)
        self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError("Out of bound")
        return getattr(self.module, self.prefix + str(i))


class PWNet(nn.Module):

    def __init__(self, k, proto_actions):
        super().__init__()
        self.k       = k
        self.epsilon = 1e-5
        self.ts      = ListModule(self, "ts_")
        for _ in range(k):
            self.ts.append(nn.Sequential(
                nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
                nn.InstanceNorm1d(PROTOTYPE_SIZE),
                nn.ReLU(),
                nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
            ))
        self.linear = nn.Linear(k, NUM_CLASSES, bias=False)
        self.linear.weight.data.copy_(
            torch.tensor(proto_actions, dtype=torch.float32).T
        )
        self.tanh       = nn.Tanh()
        self.relu       = nn.ReLU()
        self.nn_human_x = nn.Parameter(
            torch.randn(k, LATENT_SIZE), requires_grad=False
        )

    def _proto_l2(self, x, p):
        """Log-based similarity: higher = more similar."""
        b   = x.shape[0]
        p   = p.view(1, PROTOTYPE_SIZE).tile(b, 1)
        c   = x.view(b, PROTOTYPE_SIZE)
        l2s = ((c - p) ** 2).sum(axis=1)
        return torch.log((l2s + 1.0) / (l2s + self.epsilon))

    def forward_with_similarities(self, x):
        """
        Returns (action tensor (1, 3), similarities np.ndarray (k,)).
        similarities are softmax-normalised so they sum to 1 and can be
        read as "fraction of the decision driven by prototype i".
        """
        # Transform stored prototype latents
        latent_protos = torch.cat([
            t(self.nn_human_x[i].detach().clone().view(1, -1))
            for i, t in enumerate(self.ts)
        ], dim=0)                                        # (k, PROTOTYPE_SIZE)

        # Similarity of input to each prototype
        p_acts = torch.cat([
            self._proto_l2(t(x), latent_protos[i]).view(-1, 1)
            for i, t in enumerate(self.ts)
        ], dim=1)                                        # (1, k)

        similarities = torch.softmax(p_acts, dim=1).squeeze(0)   # (k,)

        # Weighted action output
        logits = self.linear(p_acts)                     # (1, 3)
        logits.T[0] = self.tanh(logits.T[0])            # steering  ∈ [-1, 1]
        logits.T[1] = self.relu(logits.T[1])            # accel     ≥ 0
        logits.T[2] = self.relu(logits.T[2])            # brake     ≥ 0

        return logits, similarities.detach().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_font(size):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def preprocess(obs_rgb):
    """(96, 96, 3) uint8  →  (96, 96) float32 grayscale [0, 1]"""
    gray = np.dot(obs_rgb.astype(np.float32), [0.299, 0.587, 0.114]) / 255.0
    return gray


def compose_frame(game_rgb, proto_imgs, proto_actions, similarities, action_out,
                  font_lg, font_sm, font_xs):
    """
    Compose one video frame.

    game_rgb      (H, W, 3) numpy RGB array from env.render()
    proto_imgs    list of k PIL Images
    proto_actions list of k np.ndarray [steer, accel, brake]
    similarities  (k,) float array, softmax-normalised
    action_out    (3,) float array [steer, accel, brake]
    """
    canvas = Image.new("RGB", (VIDEO_W, VIDEO_H), C_BG)
    draw   = ImageDraw.Draw(canvas)

    most_active = int(np.argmax(similarities))

    # ── Left panel: header ───────────────────────────────────────────────────
    draw.rectangle([0, 0, PANEL_W, HEADER_H], fill=(30, 30, 45))
    draw.text((PANEL_W // 2, HEADER_H // 2), "PROTOTYPES",
              fill=C_HEADER, anchor="mm", font=font_lg)

    # ── Prototype cells ──────────────────────────────────────────────────────
    for i in range(K):
        row = i // 2
        col = i % 2
        cx  = GRID_PAD + col * (CELL_W + GRID_PAD)
        cy  = HEADER_H + GRID_PAD + row * (CELL_H + GRID_PAD)

        border = C_ACTIVE if i == most_active else C_BORDER
        draw.rectangle([cx - 2, cy - 2, cx + CELL_W + 2, cy + CELL_H + 2],
                       fill=border)
        draw.rectangle([cx, cy, cx + CELL_W, cy + CELL_H], fill=C_CELL)

        # Prototype image
        img_x = cx + (CELL_W - PROTO_IMG_SIZE) // 2
        img_y = cy + 6
        canvas.paste(proto_imgs[i], (img_x, img_y))

        # Similarity bar
        bar_x = cx + 8
        bar_y = img_y + PROTO_IMG_SIZE + 5
        bar_w = CELL_W - 16
        bar_h = 16
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                       fill=C_BAR_BG)
        fill_px = max(1, int(bar_w * float(similarities[i])))
        draw.rectangle([bar_x, bar_y, bar_x + fill_px, bar_y + bar_h],
                       fill=C_BAR_FG)
        draw.text((bar_x + bar_w // 2, bar_y + bar_h // 2),
                  f"{similarities[i]:.2f}",
                  fill=C_BG, anchor="mm", font=font_xs)

        # Action text
        act   = proto_actions[i]
        txt_y = bar_y + bar_h + 5
        line_h = 14
        draw.text((bar_x, txt_y),
                  f"steer  {act[0]:+.3f}", fill=C_TEXT, font=font_xs)
        draw.text((bar_x, txt_y + line_h),
                  f"accel   {act[1]:.3f}", fill=C_TEXT, font=font_xs)
        draw.text((bar_x, txt_y + 2 * line_h),
                  f"brake   {act[2]:.3f}", fill=C_TEXT, font=font_xs)

    # ── Footer: current output readout ───────────────────────────────────────
    fy = VIDEO_H - FOOTER_H
    draw.rectangle([0, fy, PANEL_W, VIDEO_H], fill=(30, 30, 45))
    draw.line([(4, fy), (PANEL_W - 4, fy)], fill=C_ACTIVE, width=1)
    draw.text((8, fy + 6), "OUTPUT ACTION", fill=C_HEADER, font=font_sm)
    draw.text((8, fy + 26),
              f"steer  {action_out[0]:+.3f}", fill=C_TEXT, font=font_xs)
    draw.text((8 + CELL_W, fy + 26),
              f"accel   {action_out[1]:.3f}", fill=C_TEXT, font=font_xs)
    draw.text((8, fy + 42),
              f"brake   {action_out[2]:.3f}", fill=C_TEXT, font=font_xs)

    # ── Divider line ─────────────────────────────────────────────────────────
    draw.line([(PANEL_W, 0), (PANEL_W, VIDEO_H)], fill=C_PANEL_DIV, width=2)

    # ── Right panel: game frame ───────────────────────────────────────────────
    game_img = Image.fromarray(game_rgb)
    gw, gh   = game_img.size
    scale    = GAME_W / gw
    new_gh   = int(gh * scale)
    game_scaled = game_img.resize((GAME_W, new_gh), Image.LANCZOS)

    # Fill right panel background, then paste game frame
    game_top = max(HEADER_H, (VIDEO_H - new_gh) // 2)
    canvas.paste(game_scaled, (PANEL_W, game_top))

    # Right panel header overlay
    draw.rectangle([PANEL_W, 0, VIDEO_W, HEADER_H], fill=(30, 30, 45))
    draw.text((PANEL_W + GAME_W // 2, HEADER_H // 2), "CURRENT STATE",
              fill=C_HEADER, anchor="mm", font=font_lg)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────────────────────────────────────
Path("results").mkdir(exist_ok=True)

print("Loading fonts...")
font_lg = load_font(17)
font_sm = load_font(13)
font_xs = load_font(12)

print("Loading prototype images and actions...")
proto_imgs    = []
proto_actions = []
for i in range(K):
    img = Image.open(PROTO_DIR / f"cluster_{i}_centroid.png").convert("RGB")
    img = img.resize((PROTO_IMG_SIZE, PROTO_IMG_SIZE), Image.LANCZOS)
    proto_imgs.append(img)
    proto_actions.append(np.load(str(PROTO_DIR / f"cluster_{i}_centroid_action.npy")))
print(f"  Loaded {K} prototypes from {PROTO_DIR}")

print("Loading RacingNet (black-box encoder)...")
racing_net = RacingNet(state_dim=(FRAME_STACK, 96, 96), action_dim=(2,))
racing_net.load_state_dict(torch.load(AGENT_WEIGHTS, map_location=DEVICE))
racing_net.eval()

print(f"Loading PW-Net checkpoint (k={K})...")
proto_actions_np = np.stack(proto_actions)               # (k, 3)
pwnet = PWNet(K, proto_actions_np)
pwnet.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
pwnet.eval()

# ─────────────────────────────────────────────────────────────────────────────
# Environment + video writer
# ─────────────────────────────────────────────────────────────────────────────
print("Creating environment...")
env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_FILE), fourcc, FPS, (VIDEO_W, VIDEO_H))

# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────────────
print("Running simulation...")
obs, _ = env.reset()
frame_buf = deque([preprocess(obs)] * FRAME_STACK, maxlen=FRAME_STACK)

no_reward_steps = 0
total_reward    = 0

for step in range(MAX_STEPS):
    stacked = np.stack(frame_buf, axis=0)                        # (4, 96, 96)
    state_t = torch.FloatTensor(stacked).unsqueeze(0)            # (1, 4, 96, 96)

    with torch.no_grad():
        _, _, _, latent = racing_net(state_t)                    # (1, 256)
        action_t, sims  = pwnet.forward_with_similarities(latent)

    action_np = action_t[0].detach().numpy()                     # [steer, accel, brake]

    obs, reward, terminated, truncated, _ = env.step(action_np)
    done = terminated or truncated

    game_frame = env.render()                                    # (H, W, 3) RGB

    vis = compose_frame(game_frame, proto_imgs, proto_actions, sims, action_np,
                        font_lg, font_sm, font_xs)
    writer.write(cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR))

    frame_buf.append(preprocess(obs))
    total_reward += reward

    if reward > 0:
        no_reward_steps = 0
    else:
        no_reward_steps += 1

    if step % 100 == 0:
        active_str = " | ".join(f"P{i}={sims[i]:.2f}" for i in range(K))
        print(f"  step {step:4d}  [{active_str}]  rew={total_reward:.1f}")

    if done or no_reward_steps >= NO_REWARD_PATIENCE:
        reason = "done" if done else "no reward"
        print(f"  Episode ended at step {step} ({reason})")
        break

writer.release()
env.close()
print(f"\nTotal reward : {total_reward:.1f}")
print(f"Frames written: {step + 1}")
print(f"Video saved  → {OUT_FILE}")
