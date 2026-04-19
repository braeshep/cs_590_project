import gym
import torch
import torch.nn as nn
import numpy as np
import toml

from torch.utils.data import TensorDataset, DataLoader
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm


NUM_ITERATIONS     = 5
CONFIG_FILE        = "config.toml"
NUM_CLASSES        = 3
LATENT_SIZE        = 256
PROTOTYPE_SIZE     = 50
BATCH_SIZE         = 32
NUM_EPOCHS         = 10
DEVICE             = 'cpu'
SIMULATION_EPOCHS  = 30
K_VALUES           = [2, 4, 6, 8]


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        self.module.add_module(self.prefix + str(self.num_module), new_module)
        self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class PWNet(nn.Module):

    def __init__(self, k, proto_actions):
        """
        k            : number of prototypes (int)
        proto_actions: (k, 3) array of the canonical actions for each prototype;
                       used to initialise the frozen linear layer.
        """
        super(PWNet, self).__init__()
        self.k = k
        self.ts = ListModule(self, 'ts_')
        for _ in range(self.k):
            transformation = nn.Sequential(
                nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
                nn.InstanceNorm1d(PROTOTYPE_SIZE),
                nn.ReLU(),
                nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
            )
            self.ts.append(transformation)
        self.epsilon = 1e-5
        self.linear = nn.Linear(self.k, NUM_CLASSES, bias=False)
        self.__make_linear_weights(proto_actions)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.nn_human_x = nn.Parameter(torch.randn(self.k, LATENT_SIZE), requires_grad=False)

    def __make_linear_weights(self, proto_actions):
        """
        Initialise the frozen linear layer from the prototype action vectors.
        proto_actions shape: (k, 3)  →  transposed to (3, k) to fill weight (out, in).
        Each column of the weight matrix becomes the action vector for that prototype,
        so the final output is a similarity-weighted sum of prototype actions.
        """
        w = torch.tensor(proto_actions, dtype=torch.float32).T  # (3, k)
        self.linear.weight.data.copy_(w)

    def __proto_layer_l2(self, x, p):
        b_size = x.shape[0]
        p = p.view(1, PROTOTYPE_SIZE).tile(b_size, 1).to(DEVICE)
        c = x.view(b_size, PROTOTYPE_SIZE).to(DEVICE)
        l2s = ((c - p) ** 2).sum(axis=1).to(DEVICE)
        act = torch.log((l2s + 1.) / (l2s + self.epsilon)).to(DEVICE)
        return act

    def __output_act_func(self, p_acts):
        p_acts.T[0] = self.tanh(p_acts.T[0])  # steering ∈ [-1, 1]
        p_acts.T[1] = self.relu(p_acts.T[1])  # accel > 0
        p_acts.T[2] = self.relu(p_acts.T[2])  # brake > 0
        return p_acts

    def forward(self, x):
        trans_nn_human_x = [
            t(torch.tensor(self.nn_human_x[i], dtype=torch.float32).view(1, -1))
            for i, t in enumerate(self.ts)
        ]
        latent_protos = torch.cat(trans_nn_human_x, dim=0)

        p_acts = [
            self.__proto_layer_l2(t(x), latent_protos[i]).view(-1, 1)
            for i, t in enumerate(self.ts)
        ]
        p_acts = torch.cat(p_acts, axis=1)

        logits = self.linear(p_acts)
        return self.__output_act_func(logits)


def evaluate_loader(model, loader, loss_fn):
    model.eval()
    total_error, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            total_error += loss_fn(logits, labels).item()
            total += len(imgs)
    model.train()
    return total_error / total


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return toml.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# One-time setup: environment, PPO agent, training data, grid results
# ─────────────────────────────────────────────────────────────────────────────
cfg = load_config()
env = CarRacing(frame_skip=0, frame_stack=4)
net = RacingNet(env.observation_space.shape, env.action_space.shape)
ppo = PPO(
    env, net,
    lr=cfg["lr"], gamma=cfg["gamma"], batch_size=cfg["batch_size"],
    gae_lambda=cfg["gae_lambda"], clip=cfg["clip"],
    value_coef=cfg["value_coef"], entropy_coef=cfg["entropy_coef"],
    epochs_per_step=cfg["epochs_per_step"], num_steps=cfg["num_steps"],
    horizon=cfg["horizon"], save_dir=cfg["save_dir"],
    save_interval=cfg["save_interval"],
)
ppo.load("weights/agent_weights.pt")

print("Loading training data from prototype_data.npz...")
raw = np.load("prototype_data.npz")
latents      = raw["latents"]   # (N, 256)
actions_data = raw["actions"]   # (N, 3)
tensor_x     = torch.tensor(latents, dtype=torch.float32)
tensor_y     = torch.tensor(actions_data, dtype=torch.float32)
train_dataset = TensorDataset(tensor_x, tensor_y)
train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
print(f"  {len(latents):,} timesteps loaded")

print("Loading grid clustering results...")
grid_results = np.load("prototypes_grid/kmeans_grid_results.npz")

# ─────────────────────────────────────────────────────────────────────────────
# K-sweep
# ─────────────────────────────────────────────────────────────────────────────
all_k_results = {}

for k in K_VALUES:
    print(f"\n{'='*55}")
    print(f"=== Running PW-Net for k={k} ===")
    print(f"{'='*55}")

    # Map downsampled prototype indices back to original dataset indices
    keep_idx   = grid_results[f"k{k}_keep_idx"]
    p_idxs_ds  = grid_results[f"k{k}_centroid_frame_indices"]
    p_idxs     = keep_idx[p_idxs_ds]

    nn_human_x       = latents[p_idxs]       # (k, 256)
    nn_human_actions = actions_data[p_idxs]  # (k, 3)

    model_path = f"weights/pw_net_k{k}.pth"
    mse_loss   = nn.MSELoss()

    k_rewards = []
    k_errors  = []

    for iteration in range(NUM_ITERATIONS):
        print(f"\n  -- Iteration {iteration + 1}/{NUM_ITERATIONS} --")

        # ── Training ────────────────────────────────────────────────────────
        model = PWNet(k, nn_human_actions).eval()
        model.nn_human_x.data.copy_(torch.tensor(nn_human_x))
        model.linear.weight.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        best_error = float('inf')
        model.train()

        for epoch in range(NUM_EPOCHS):
            model.eval()
            train_error = evaluate_loader(model, train_loader, mse_loss)
            model.train()

            if train_error < best_error:
                torch.save(model.state_dict(), model_path)
                best_error = train_error

            running_loss = 0
            for instances, labels in train_loader:
                optimizer.zero_grad()
                instances, labels = instances.to(DEVICE), labels.to(DEVICE)
                loss = mse_loss(model(instances), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"    Epoch {epoch:2d}  loss={running_loss/len(train_loader):.4f}"
                  f"  MAE={train_error:.4f}")
            scheduler.step()

        # ── Evaluation ──────────────────────────────────────────────────────
        model = PWNet(k, nn_human_actions).eval()
        model.load_state_dict(torch.load(model_path))
        print(f"  Sanity-check MSE: {evaluate_loader(model, train_loader, mse_loss):.4f}")

        reward_arr = []
        all_errors = []
        ppo._to_tensor(env.reset())

        for _ in tqdm(range(SIMULATION_EPOCHS), desc=f"  k={k} iter={iteration+1} eval"):
            state = ppo._to_tensor(env.reset())
            rew = 0

            for _ in range(10000):
                value, alpha, beta, latent_x = ppo.net(state)
                alpha, beta = alpha.squeeze(0), beta.squeeze(0)
                bb_action = ppo.env.preprocess(Beta(alpha, beta).mean.detach())

                action = model(latent_x)
                all_errors.append(
                    mse_loss(torch.tensor(bb_action), action[0]).detach().item()
                )

                state, reward, done, _, _ = ppo.env.step(
                    action[0].detach().numpy(), real_action=True
                )
                state = ppo._to_tensor(state)
                rew += reward
                if done:
                    break

            reward_arr.append(rew)

        iter_reward = sum(reward_arr) / SIMULATION_EPOCHS
        iter_error  = sum(all_errors) / SIMULATION_EPOCHS
        k_rewards.append(iter_reward)
        k_errors.append(iter_error)
        print(f"  Iteration {iteration+1} — Reward: {iter_reward:.2f}  MAE: {iter_error:.4f}")

    k_rewards = np.array(k_rewards)
    k_errors  = np.array(k_errors)
    all_k_results[k] = {"rewards": k_rewards, "errors": k_errors}

    print(f"\n--- k={k} Summary ---")
    print(f"  MAE   : mean={k_errors.mean():.4f}  "
          f"se={k_errors.std() / np.sqrt(NUM_ITERATIONS):.4f}")
    print(f"  Reward: mean={k_rewards.mean():.2f}  "
          f"se={k_rewards.std() / np.sqrt(NUM_ITERATIONS):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# Full sweep summary table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("FULL K-SWEEP SUMMARY")
print(f"{'='*55}")
print(f"{'k':>4}  {'MAE Mean':>10}  {'MAE SE':>8}  {'Rew Mean':>10}  {'Rew SE':>8}")
for k in K_VALUES:
    r = all_k_results[k]
    n = NUM_ITERATIONS
    print(f"{k:>4}  {r['errors'].mean():>10.4f}  "
          f"{r['errors'].std()/np.sqrt(n):>8.4f}  "
          f"{r['rewards'].mean():>10.2f}  "
          f"{r['rewards'].std()/np.sqrt(n):>8.2f}")
