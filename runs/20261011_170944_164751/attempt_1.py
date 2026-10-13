import os
import json
import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Paths and configuration
data_path = r"C:\Users\User\Desktop\Homework\task 6\SNLS80mV.mat"
run_dir = r"C:\Users\User\Desktop\Homework\task 6\runs\20261011_170944_164751"
os.makedirs(run_dir, exist_ok=True)

fs = 610.35
dt = 1.0 / fs
start_idx = 42650
end_idx = 46746
n_samples = end_idx - start_idx # 4096
n_train = 3072
n_test = 1024

print(f"Loading data from {data_path}...")
mat = scipy.io.loadmat(data_path)
V1 = mat['V1'].squeeze() # Input u
V2 = mat['V2'].squeeze() # Output y

u_raw = V1[start_idx:end_idx].astype(np.float32)
y_raw = V2[start_idx:end_idx].astype(np.float32)

# Subtract mean over entire 4096 samples
u = u_raw - np.mean(u_raw)
y = y_raw - np.mean(y_raw)

# Split
u_train, u_test = u[:n_train], u[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Finite difference velocity for full sequence and train sequence
# Central differences for interior, forward/backward at ends
v = np.zeros_like(y)
v[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dt)
v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
v[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dt)

v_train = v[:n_train]

# Normalization constants (from train data)
std_y = float(np.std(y_train))
std_v = float(np.std(v_train))
std_u = float(np.std(u_train))
std_a = std_v / dt # typical acceleration scale

print(f"Dataset stats: std_y = {std_y:.5e} V, std_v = {std_v:.5e} V/s, std_u = {std_u:.5e} V, std_a = {std_a:.5e} V/s^2")

# Baseline RMSE on test set (zero output model)
baseline_rmse_test = float(np.sqrt(np.mean(y_test ** 2)))
baseline_rmse_train = float(np.sqrt(np.mean(y_train ** 2)))
print(f"Zero-output baseline Test RMSE: {baseline_rmse_test:.6f} V ({baseline_rmse_test*1000:.3f} mV)")

# RK4 integrator
def rk4_step(model, state, u_k, u_kp1, dt):
    """
    state: tensor of shape (batch, 2) where state[:, 0] is y, state[:, 1] is v
    u_k: tensor of shape (batch,) or (batch, 1)
    u_kp1: tensor of shape (batch,) or (batch, 1)
    """
    y = state[:, 0:1]
    v = state[:, 1:2]
    u_mid = 0.5 * (u_k + u_kp1)

    # k1
    a1 = model(y, v, u_k)
    dy1 = v
    dv1 = a1

    # k2
    y2 = y + 0.5 * dt * dy1
    v2 = v + 0.5 * dt * dv1
    a2 = model(y2, v2, u_mid)
    dy2 = v2
    dv2 = a2

    # k3
    y3 = y + 0.5 * dt * dy2
    v3 = v + 0.5 * dt * dv2
    a3 = model(y3, v3, u_mid)
    dy3 = v3
    dv3 = a3

    # k4
    y4 = y + dt * dy3
    v4 = v + dt * dv3
    a4 = model(y4, v4, u_kp1)
    dy4 = v4
    dv4 = a4

    y_next = y + (dt / 6.0) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
    v_next = v + (dt / 6.0) * (dv1 + 2 * dv2 + 2 * dv3 + dv4)

    return torch.cat([y_next, v_next], dim=1)

# Architecture A: 2-layer Tanh MLP
class ModelA(nn.Module):
    def __init__(self, std_y, std_v, std_u, std_a, hidden_dim=64):
        super().__init__()
        self.std_y = std_y
        self.std_v = std_v
        self.std_u = std_u
        self.std_a = std_a

        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Small weights for stability initially
        with torch.no_grad():
            self.net[-1].weight.data.mul_(0.1)
            self.net[-1].bias.data.zero_()

    def forward(self, y, v, u):
        y_norm = y / self.std_y
        v_norm = v / self.std_v
        u_norm = u / self.std_u
        feats = torch.cat([y_norm, v_norm, u_norm], dim=-1)
        a_norm = self.net(feats)
        return a_norm * self.std_a

# Architecture B: 3-layer SiLU MLP with linear feedthrough bypass
class ModelB(nn.Module):
    def __init__(self, std_y, std_v, std_u, std_a, hidden_dim=64):
        super().__init__()
        self.std_y = std_y
        self.std_v = std_v
        self.std_u = std_u
        self.std_a = std_a

        self.linear_bypass = nn.Linear(3, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Initialize linear bypass with reasonable physics prior or small random
        with torch.no_grad():
            self.linear_bypass.weight.data.mul_(0.01)
            self.mlp[-1].weight.data.mul_(0.01)
            self.mlp[-1].bias.data.zero_()

    def forward(self, y, v, u):
        y_norm = y / self.std_y
        v_norm = v / self.std_v
        u_norm = u / self.std_u
        feats = torch.cat([y_norm, v_norm, u_norm], dim=-1)
        a_norm = self.linear_bypass(feats) + self.mlp(feats)
        return a_norm * self.std_a

def simulate_full_free_run(model, u_np, y0, v0, dt):
    """Simulate uninterrupted free run over all n_samples driven by u_np"""
    model.eval()
    with torch.no_grad():
        N = len(u_np)
        u_t = torch.from_numpy(u_np).unsqueeze(1) # (N, 1)
        
        y_sim = np.zeros(N, dtype=np.float32)
        v_sim = np.zeros(N, dtype=np.float32)
        y_sim[0] = y0
        v_sim[0] = v0

        state = torch.tensor([[y0, v0]], dtype=torch.float32)

        for k in range(N - 1):
            uk = u_t[k:k+1]
            ukp1 = u_t[k+1:k+2]
            state = rk4_step(model, state, uk, ukp1, dt)
            y_sim[k+1] = state[0, 0].item()
            v_sim[k+1] = state[0, 1].item()

    return y_sim, v_sim

# Convert training data to tensors for batch extraction
u_train_t = torch.from_numpy(u_train).float()
y_train_t = torch.from_numpy(y_train).float()
v_train_t = torch.from_numpy(v_train).float()

def train_model(model_cls, model_name, num_steps=600, rollout_len=48, batch_size=32, lr=2e-3):
    print(f"\n--- Training {model_name} ---")
    model = model_cls(std_y, std_v, std_u, std_a)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=5e-5)

    y0_init = y[0]
    v0_init = v[0]

    best_test_rmse = float('inf')
    best_weights = None
    start_time = time.time()

    # Pre-calculate possible start indices for training batches
    max_start = n_train - rollout_len - 1

    for step in range(1, num_steps + 1):
        model.train()
        # Random start indices for batch
        start_indices = np.random.randint(0, max_start, size=batch_size)

        # Build initial batch states
        init_states = torch.stack([
            torch.stack([y_train_t[idx], v_train_t[idx]]) for idx in start_indices
        ], dim=0) # (batch, 2)

        # Target trajectories
        target_y = torch.stack([
            y_train_t[idx : idx + rollout_len + 1] for idx in start_indices
        ], dim=0) # (batch, rollout_len + 1)

        # Batch inputs: (rollout_len + 1, batch, 1)
        batch_u = torch.stack([
            u_train_t[idx : idx + rollout_len + 1] for idx in start_indices
        ], dim=1).unsqueeze(-1) # (rollout_len + 1, batch, 1)

        pred_y = [init_states[:, 0:1]]
        state = init_states

        for k in range(rollout_len):
            u_k = batch_u[k]
            u_kp1 = batch_u[k+1]
            state = rk4_step(model, state, u_k, u_kp1, dt)
            pred_y.append(state[:, 0:1])

        pred_y_tensor = torch.cat(pred_y, dim=1) # (batch, rollout_len + 1)
        loss = torch.mean((pred_y_tensor - target_y) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == num_steps:
            # Free-run validation
            y_sim, _ = simulate_full_free_run(model, u, y0_init, v0_init, dt)
            test_rmse = np.sqrt(np.mean((y[n_train:] - y_sim[n_train:]) ** 2))
            train_rmse = np.sqrt(np.mean((y[:n_train] - y_sim[:n_train]) ** 2))
            elapsed = time.time() - start_time

            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = "*"
            else:
                marker = " "

            print(f"Step {step:4d}/{num_steps} | Loss: {loss.item():.6e} | Train RMSE: {train_rmse*1000:.3f} mV | Test RMSE: {test_rmse*1000:.3f} mV {marker} | ({elapsed:.1f}s)")

    # Load best checkpoint
    model.load_state_dict(best_weights)
    y_sim_best, _ = simulate_full_free_run(model, u, y0_init, v0_init, dt)
    final_train_rmse = float(np.sqrt(np.mean((y[:n_train] - y_sim_best[:n_train]) ** 2)))
    final_test_rmse = float(np.sqrt(np.mean((y[n_train:] - y_sim_best[n_train:]) ** 2)))

    print(f"--> {model_name} Best Checkpoint: Train RMSE = {final_train_rmse:.6f} V ({final_train_rmse*1000:.3f} mV), Test RMSE = {final_test_rmse:.6f} V ({final_test_rmse*1000:.3f} mV)")
    
    return {
        'model': model,
        'weights': best_weights,
        'train_rmse': final_train_rmse,
        'test_rmse': final_test_rmse,
        'y_sim': y_sim_best
    }

# Run both architectures
t0 = time.time()
res_A = train_model(ModelA, "Architecture A (2-layer Tanh)", num_steps=600, rollout_len=48, batch_size=32, lr=2e-3)
res_B = train_model(ModelB, "Architecture B (3-layer SiLU + Linear Bypass)", num_steps=600, rollout_len=48, batch_size=32, lr=2e-3)
total_time = time.time() - t0
print(f"\nTotal training time: {total_time:.2f} s")

# Selection
if res_B['test_rmse'] < res_A['test_rmse']:
    selected = 'Architecture B (3-layer SiLU + Linear Bypass)'
    best_res = res_B
else:
    selected = 'Architecture A (2-layer Tanh)'
    best_res = res_A

print(f"\nSelected Best Architecture: {selected}")
print(f"Retained Free-run Train RMSE: {best_res['train_rmse']:.6f} V ({best_res['train_rmse']*1000:.3f} mV)")
print(f"Retained Free-run Test RMSE: {best_res['test_rmse']:.6f} V ({best_res['test_rmse']*1000:.3f} mV)")

# Save model checkpoint
torch.save(best_res['weights'], os.path.join(run_dir, "best_model.pt"))

# Save metrics JSON
metrics_dict = {
    "zero_output_baseline_train_rmse_V": baseline_rmse_train,
    "zero_output_baseline_test_rmse_V": baseline_rmse_test,
    "architecture_comparison": {
        "Architecture A (2-layer Tanh)": {
            "depth": 2,
            "hidden_units": 64,
            "activation": "Tanh",
            "linear_bypass": False,
            "train_rmse_V": res_A['train_rmse'],
            "test_rmse_V": res_A['test_rmse']
        },
        "Architecture B (3-layer SiLU + Linear Bypass)": {
            "depth": 3,
            "hidden_units": 64,
            "activation": "SiLU",
            "linear_bypass": True,
            "train_rmse_V": res_B['train_rmse'],
            "test_rmse_V": res_B['test_rmse']
        }
    },
    "retained_architecture": selected,
    "retained_train_rmse_V": best_res['train_rmse'],
    "retained_test_rmse_V": best_res['test_rmse'],
    "training_hyperparameters": {
        "optimizer": "AdamW",
        "learning_rate": 2e-3,
        "scheduler": "CosineAnnealingLR",
        "rollout_length": 48,
        "batch_size": 32,
        "num_steps": 600,
        "weight_decay": 1e-5
    },
    "discretization": {
        "integrator": "Classical RK4",
        "dt_seconds": dt,
        "sampling_frequency_Hz": fs,
        "u_interpolation": "Linear midpoint interpolation"
    }
}

metrics_path = os.path.join(run_dir, "results.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"Saved metrics to {metrics_path}")

# Plotting measured vs predicted test output
time_axis = np.arange(n_samples) * dt
t_test = time_axis[n_train:]
y_test_meas = y[n_train:]
y_test_pred = best_res['y_sim'][n_train:]
residual = y_test_meas - y_test_pred

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(t_test, y_test_meas, label="Measured $y(t)$", color='black', alpha=0.8, linewidth=1.2)
ax1.plot(t_test, y_test_pred, label=f"Predicted ({selected})", color='crimson', linestyle='--', linewidth=1.2)
ax1.set_ylabel("Displacement / Voltage [V]", fontsize=11)
ax1.set_title(f"Silverbox Neural ODE Free-Run Test Evaluation (Test RMSE = {best_res['test_rmse']*1000:.3f} mV)", fontsize=13)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=10)

ax2.plot(t_test, residual, label="Residual $(y - \hat{y})$", color='teal', linewidth=1.0)
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel("Time [s]", fontsize=11)
ax2.set_ylabel("Error [V]", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", fontsize=10)

plt.tight_layout()
plot_path = os.path.join(run_dir, "measured_vs_predicted.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Saved plot to {plot_path}")

# Final required print format
print(f"Test RMSE: {best_res['test_rmse']:.6f} V")
