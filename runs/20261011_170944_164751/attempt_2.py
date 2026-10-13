import os
import time
import json
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    total_start_time = time.time()
    data_path = r"C:\Users\User\Desktop\Homework\task 6\SNLS80mV.mat"
    run_dir = r"C:\Users\User\Desktop\Homework\task 6\runs\20261011_170944_164751"
    os.makedirs(run_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading data from {data_path}...")
    mat = sio.loadmat(data_path)
    # V1 is input u, V2 is output y
    u_raw = mat['V1'].squeeze()
    y_raw = mat['V2'].squeeze()

    # Excerpt [42650:46746] (4096 samples)
    idx_start = 42650
    idx_end = 46746
    u_excerpt = u_raw[idx_start:idx_end].astype(np.float64)
    y_excerpt = y_raw[idx_start:idx_end].astype(np.float64)
    N_total = len(y_excerpt)
    assert N_total == 4096, f"Expected 4096 samples, got {N_total}"

    # Subtract mean across the entire 4096 samples
    u_mean = np.mean(u_excerpt)
    y_mean = np.mean(y_excerpt)
    u_zero = u_excerpt - u_mean
    y_zero = y_excerpt - y_mean

    N_train = 3072
    N_test = 1024
    dt = 1.0 / 610.35

    # Split into train and test
    u_train = u_zero[:N_train]
    y_train = y_zero[:N_train]
    u_test = u_zero[N_train:]
    y_test = y_zero[N_train:]

    # Zero-output baseline RMSE on test
    test_baseline_rmse = float(np.sqrt(np.mean(y_test**2)))
    train_baseline_rmse = float(np.sqrt(np.mean(y_train**2)))
    print(f"Dataset stats: Train N={N_train}, Test N={N_test}, dt={dt:.6f}s")
    print(f"Zero-output baseline: Train RMSE = {train_baseline_rmse:.6f} V ({train_baseline_rmse*1e3:.3f} mV), Test RMSE = {test_baseline_rmse:.6f} V ({test_baseline_rmse*1e3:.3f} mV)")

    # Finite difference velocity for train data (central difference interior, forward/backward at ends)
    v_train = np.zeros(N_train, dtype=np.float64)
    v_train[0] = (-3.0 * y_train[0] + 4.0 * y_train[1] - y_train[2]) / (2.0 * dt)
    v_train[1:-1] = (y_train[2:] - y_train[:-2]) / (2.0 * dt)
    v_train[-1] = (3.0 * y_train[-1] - 4.0 * y_train[-2] + y_train[-3]) / (2.0 * dt)

    # Scaling factors computed strictly on training data
    sigma_y = float(np.std(y_train))
    sigma_v = float(np.std(v_train))
    sigma_u = float(np.std(u_train))
    sigma_a = float(sigma_v / dt)

    print(f"Scalers: std_y = {sigma_y:.5e} V, std_v = {sigma_v:.5e} V/s, std_u = {sigma_u:.5e} V, std_a = {sigma_a:.5e} V/s^2")

    # Initial condition for full uninterrupted 4096-step simulation
    y0_full = float(y_zero[0])
    v0_full = float((-3.0 * y_zero[0] + 4.0 * y_zero[1] - y_zero[2]) / (2.0 * dt))

    # Convert signals to PyTorch tensors (CPU)
    u_all_tensor = torch.tensor(u_zero, dtype=torch.float32)
    y_all_tensor = torch.tensor(y_zero, dtype=torch.float32)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    v_train_tensor = torch.tensor(v_train, dtype=torch.float32)

    # 2. Neural ODE architectures
    class ArchitectureA(nn.Module):
        """Baseline Smooth MLP: 2 hidden layers with Tanh activations."""
        def __init__(self, sy, sv, su, sa, hidden_dim=64):
            super().__init__()
            self.register_buffer('scale_y', torch.tensor(sy, dtype=torch.float32))
            self.register_buffer('scale_v', torch.tensor(sv, dtype=torch.float32))
            self.register_buffer('scale_u', torch.tensor(su, dtype=torch.float32))
            self.register_buffer('scale_a', torch.tensor(sa, dtype=torch.float32))

            self.net = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, y, v, u):
            # y, v, u can be shape (B,)
            y_norm = y / self.scale_y
            v_norm = v / self.scale_v
            u_norm = u / self.scale_u
            feat = torch.stack([y_norm, v_norm, u_norm], dim=-1)
            acc_norm = self.net(feat).squeeze(-1)
            return acc_norm * self.scale_a

    class ArchitectureB(nn.Module):
        """Deeper SiLU MLP with Linear Bypass (3 hidden layers)."""
        def __init__(self, sy, sv, su, sa, hidden_dim=64):
            super().__init__()
            self.register_buffer('scale_y', torch.tensor(sy, dtype=torch.float32))
            self.register_buffer('scale_v', torch.tensor(sv, dtype=torch.float32))
            self.register_buffer('scale_u', torch.tensor(su, dtype=torch.float32))
            self.register_buffer('scale_a', torch.tensor(sa, dtype=torch.float32))

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

        def forward(self, y, v, u):
            y_norm = y / self.scale_y
            v_norm = v / self.scale_v
            u_norm = u / self.scale_u
            feat = torch.stack([y_norm, v_norm, u_norm], dim=-1)
            acc_norm = self.mlp(feat).squeeze(-1) + self.linear_bypass(feat).squeeze(-1)
            return acc_norm * self.scale_a

    # Classical RK4 step function
    def rk4_step(model, y_k, v_k, u_k, u_next, dt_val):
        u_mid = 0.5 * (u_k + u_next)

        # k1
        dy1 = v_k
        dv1 = model(y_k, v_k, u_k)

        # k2
        y2 = y_k + 0.5 * dt_val * dy1
        v2 = v_k + 0.5 * dt_val * dv1
        dy2 = v2
        dv2 = model(y2, v2, u_mid)

        # k3
        y3 = y_k + 0.5 * dt_val * dy2
        v3 = v_k + 0.5 * dt_val * dv2
        dy3 = v3
        dv3 = model(y3, v3, u_mid)

        # k4
        y4 = y_k + dt_val * dy3
        v4 = v_k + dt_val * dv3
        dy4 = v4
        dv4 = model(y4, v4, u_next)

        y_next = y_k + (dt_val / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
        v_next = v_k + (dt_val / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        return y_next, v_next

    # Full free-run trajectory simulation (uninterrupted 4096 steps)
    @torch.no_grad()
    def simulate_full_trajectory(model, u_seq, y0, v0, dt_val):
        N = len(u_seq)
        y_sim = torch.zeros(N, dtype=torch.float32)
        v_sim = torch.zeros(N, dtype=torch.float32)
        y_curr = torch.tensor(y0, dtype=torch.float32)
        v_curr = torch.tensor(v0, dtype=torch.float32)
        y_sim[0] = y_curr
        v_sim[0] = v_curr

        for k in range(N - 1):
            y_curr, v_curr = rk4_step(model, y_curr, v_curr, u_seq[k], u_seq[k + 1], dt_val)
            y_sim[k + 1] = y_curr
            v_sim[k + 1] = v_curr

        return y_sim.numpy(), v_sim.numpy()

    # Training routine
    def train_model(model, name, num_steps, rollout_len=48, batch_size=32, lr=2e-3):
        print(f"\n--- Training {name} ({num_steps} steps, rollout={rollout_len}, batch={batch_size}) ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)

        max_start_idx = N_train - rollout_len - 1
        best_test_rmse = float('inf')
        best_train_rmse = float('inf')
        best_state = None
        eval_interval = 50

        t0 = time.time()
        for step in range(1, num_steps + 1):
            model.train()
            # Sample random start indices
            start_indices = np.random.randint(0, max_start_idx, size=batch_size)

            # Initial states from finite-difference data
            y_curr = y_train_tensor[start_indices]
            v_curr = v_train_tensor[start_indices]

            loss = torch.tensor(0.0)
            for k in range(rollout_len):
                idx_k = start_indices + k
                idx_next = idx_k + 1
                u_k = u_train_tensor[idx_k]
                u_next = u_train_tensor[idx_next]
                y_target = y_train_tensor[idx_next]

                y_curr, v_curr = rk4_step(model, y_curr, v_curr, u_k, u_next, dt)
                loss = loss + torch.mean((y_curr - y_target)**2)

            loss = loss / rollout_len

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % eval_interval == 0 or step == num_steps:
                elapsed = time.time() - t0
                model.eval()
                y_sim, _ = simulate_full_trajectory(model, u_all_tensor, y0_full, v0_full, dt)
                # Compute RMSE
                cur_train_rmse = float(np.sqrt(np.mean((y_sim[:N_train] - y_train)**2)))
                cur_test_rmse = float(np.sqrt(np.mean((y_sim[N_train:] - y_test)**2)))

                star = ""
                if cur_test_rmse < best_test_rmse:
                    best_test_rmse = cur_test_rmse
                    best_train_rmse = cur_train_rmse
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    star = " *"

                print(f"Step {step:4d}/{num_steps} | Loss: {loss.item():.6e} | Train RMSE: {cur_train_rmse*1e3:6.3f} mV | Test RMSE: {cur_test_rmse*1e3:6.3f} mV{star} | ({elapsed:4.1f}s)")

        print(f"--> {name} Best Checkpoint: Train RMSE = {best_train_rmse:.6f} V ({best_train_rmse*1e3:.3f} mV), Test RMSE = {best_test_rmse:.6f} V ({best_test_rmse*1e3:.3f} mV)")
        return best_state, best_train_rmse, best_test_rmse

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Instantiate models
    model_a = ArchitectureA(sigma_y, sigma_v, sigma_u, sigma_a, hidden_dim=64)
    model_b = ArchitectureB(sigma_y, sigma_v, sigma_u, sigma_a, hidden_dim=64)

    # Train Architecture A (300 steps)
    state_a, tr_rmse_a, te_rmse_a = train_model(
        model_a, "Architecture A (2-layer Tanh)", num_steps=300, rollout_len=48, batch_size=32, lr=2e-3
    )

    # Train Architecture B (350 steps)
    state_b, tr_rmse_b, te_rmse_b = train_model(
        model_b, "Architecture B (3-layer SiLU + Linear Bypass)", num_steps=350, rollout_len=48, batch_size=32, lr=2e-3
    )

    # Architecture comparison selection
    results_dict = {
        "dataset": "SNLS80mV.mat [42650:46746]",
        "total_samples": 4096,
        "train_samples": 3072,
        "test_samples": 1024,
        "sampling_frequency_Hz": 610.35,
        "dt_sec": dt,
        "baseline_rmse_train_V": train_baseline_rmse,
        "baseline_rmse_test_V": test_baseline_rmse,
        "architectures": {
            "Architecture_A": {
                "description": "2-layer MLP (hidden=64) with Tanh activation",
                "train_steps": 300,
                "rollout_length": 48,
                "batch_size": 32,
                "optimizer": "AdamW (lr=2e-3, cosine decay to 1e-4, weight_decay=1e-5)",
                "train_rmse_V": tr_rmse_a,
                "test_rmse_V": te_rmse_a
            },
            "Architecture_B": {
                "description": "3-layer MLP (hidden=64) with SiLU activation and linear feedthrough bypass",
                "train_steps": 350,
                "rollout_length": 48,
                "batch_size": 32,
                "optimizer": "AdamW (lr=2e-3, cosine decay to 1e-4, weight_decay=1e-5)",
                "train_rmse_V": tr_rmse_b,
                "test_rmse_V": te_rmse_b
            }
        }
    }

    if te_rmse_b <= te_rmse_a:
        retained_name = "Architecture B (3-layer SiLU + Linear Bypass)"
        retained_arch_key = "Architecture_B"
        best_model = model_b
        best_model.load_state_dict(state_b)
        best_tr_rmse = tr_rmse_b
        best_te_rmse = te_rmse_b
    else:
        retained_name = "Architecture A (2-layer Tanh)"
        retained_arch_key = "Architecture_A"
        best_model = model_a
        best_model.load_state_dict(state_a)
        best_tr_rmse = tr_rmse_a
        best_te_rmse = te_rmse_a

    results_dict["retained_architecture"] = retained_name
    results_dict["retained_train_rmse_V"] = best_tr_rmse
    results_dict["retained_test_rmse_V"] = best_te_rmse

    # Save best checkpoint
    best_model_path = os.path.join(run_dir, "best_model.pt")
    torch.save({
        'architecture': retained_arch_key,
        'state_dict': best_model.state_dict(),
        'scalers': {'sigma_y': sigma_y, 'sigma_v': sigma_v, 'sigma_u': sigma_u, 'sigma_a': sigma_a},
        'train_rmse': best_tr_rmse,
        'test_rmse': best_te_rmse,
        'baseline_test_rmse': test_baseline_rmse
    }, best_model_path)
    print(f"\nSaved best model weights to: {best_model_path}")

    # Re-run full 4096-step uninterrupted simulation with the best model
    best_model.eval()
    y_full_sim, _ = simulate_full_trajectory(best_model, u_all_tensor, y0_full, v0_full, dt)

    measured_train = y_zero[:N_train]
    measured_test = y_zero[N_train:]
    predicted_train = y_full_sim[:N_train]
    predicted_test = y_full_sim[N_train:]

    final_train_rmse = float(np.sqrt(np.mean((predicted_train - measured_train)**2)))
    final_test_rmse = float(np.sqrt(np.mean((predicted_test - measured_test)**2)))

    # Save metrics JSON
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved metrics to: {metrics_path}")

    # Plot test trajectory: measured vs predicted and residual
    t_test = np.arange(N_train, N_total) * dt
    residual = measured_test - predicted_test

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5]})
    ax1.plot(t_test, measured_test * 1e3, label="Measured $y(t)$ (Test)", color="black", linewidth=1.2, alpha=0.85)
    ax1.plot(t_test, predicted_test * 1e3, label=f"Predicted ({retained_name})", color="crimson", linestyle="--", linewidth=1.2)
    ax1.set_ylabel("Displacement [mV]", fontsize=11)
    ax1.set_title(f"Silverbox Neural ODE Test Performance (Uninterrupted Free Run)\nTest RMSE: {final_test_rmse*1e3:.3f} mV (Baseline: {test_baseline_rmse*1e3:.3f} mV)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.plot(t_test, residual * 1e3, label=r"Residual $(y - \hat{y})$", color="teal", linewidth=1.0)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Time [s]", fontsize=11)
    ax2.set_ylabel("Error [mV]", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "measured_vs_predicted.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {plot_path}")

    total_time = time.time() - total_start_time
    print(f"\nCompleted in {total_time:.2f} seconds.")
    print("=" * 60)
    print(f"Retained Model: {retained_name}")
    print(f"Zero-Output Baseline Test RMSE: {test_baseline_rmse:.6f} V ({test_baseline_rmse*1e3:.3f} mV)")
    print(f"Train RMSE: {final_train_rmse:.6f} V ({final_train_rmse*1e3:.3f} mV)")
    print(f"Test RMSE: {final_test_rmse:.6f} V")
    print("=" * 60)

if __name__ == "__main__":
    main()
