import numpy as np
import importlib
import os
import yaml
from pathlib import Path
from typing import Optional, Dict
from ctf4science.eval_module import evaluate_custom
import torch
import torch.nn as nn
from torchdiffeq import odeint

class Normalizer:
    def __init__(self, y, normalize=True):
        self.norm = normalize
        self.mean = y.mean(dim=0, keepdim=True)
        # +eps for numerical stability
        self.std  = y.std(dim=0, keepdim=True) + 1e-8

    def normalize(self, y):
        if not self.norm:
            return y
        return (y - self.mean.to(y.device)) / self.std.to(y.device)

    def denormalize(self, y):
        if not self.norm:
            return y
        return y * self.std.to(y.device) + self.mean.to(y.device)

class ODEF(nn.Module):
    def __init__(self, hidden_dim=512, input_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(hidden_dim*4, hidden_dim),
            # nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, y):
        return self.net(y)


class NeuralOde:
    """
    A class representing a NeuralODE baseline
    """

    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, init_data: Optional[np.ndarray] = None,
                 prediction_timesteps=None, training_timesteps=None, pair_id: Optional[int] = None):
        """
        Initialize the NeuralODE model with the provided configuration.

        Args:
            config (Dict): Configuration dictionary containing method and parameters.
            train_data (Optional[np.ndarray]): Training data for 'average' method.
        """
        self.method = config['model']['method']
        self.dataset_name = config['dataset']['name']
        self.lr = float(config['model']['lr'])
        self.hidden_size = int(config['model']['hidden_dim'])
        self.batch_size = int(config['model']['batch_size'])
        self.seq_len = int(config['model']['seq_len'])
        self.pair_id = pair_id
        self.train_data = train_data
        self.init_data = init_data
        self.prediction_timesteps = torch.tensor(prediction_timesteps)
        self.training_timesteps = torch.tensor(training_timesteps)
        self.prediction_horizon_steps = len(prediction_timesteps)
        self.num_epochs = int(config['model']['epochs'])
        self.batch_id = config['model'].get('batch_id', None)
    def predict(self) -> np.ndarray:
        """
        Generate predictions based on the specified model method.
        """
        print(self.training_timesteps[0][-1], len(self.training_timesteps[0]))
        print("timesteps",self.prediction_timesteps[-1], self.prediction_timesteps[0], len(self.prediction_timesteps))
        print(len(self.train_data))
        print(self.train_data[0].shape)
        print("init_data",self.init_data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t_train = []
        y_train = []
        t_eval = []
        y_eval = []

        # --- Interleaved split over sliding windows ---
        seq_len = self.seq_len
        train_ratio = 0.98

        for traj_idx, traj in enumerate(self.train_data):
            traj = torch.tensor(traj, dtype=torch.float32)  # [T, D]
            t_total = torch.tensor(self.training_timesteps[traj_idx], dtype=torch.float32)  # [T]
            T = traj.shape[0]

            if T < seq_len:
                continue

            starts = torch.arange(0, T - seq_len + 1)
            num_windows = starts.numel()
            perm = torch.randperm(num_windows)
            split = int(train_ratio * num_windows)
            train_starts = starts[perm[:split]]
            eval_starts  = starts[perm[split:]]

            # TRAIN windows
            for s in train_starts.tolist():
                y_train.append(traj[s: s + seq_len])       # [L, D]
                t_train.append(t_total[s: s + seq_len])    # [L]

            # EVAL windows
            for s in eval_starts.tolist():
                y_eval.append(traj[s: s + seq_len])        # [L, D]
                t_eval.append(t_total[s: s + seq_len])     # [L]

        print(t_train[-1].shape, y_train[-1].shape, t_eval[-1].shape, y_eval[-1].shape)

        def train_neural_ode(t_train_list, y_train_list, t_eval_list, y_eval_list,
                             niters=100, lr=1e-3, batch_size=40, seq_len=40):
            print(f'CUDA: {torch.cuda.is_available()}', flush=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Device: {device}', flush=True)

            # === Concatenate all training data for normalization ===
            y_train_all = torch.cat(y_train_list, dim=0)
            normalizer = Normalizer(y_train_all, normalize=self.method == "normalize")

            # === Normalize all trajectories (keep on CPU to save GPU memory) ===
            y_train_list = [normalizer.normalize(y) for y in y_train_list]
            y_eval_list  = [normalizer.normalize(y) for y in y_eval_list]

            dim = y_train_list[0].shape[1]
            ode_func = ODEF(input_dim=dim, hidden_dim=self.hidden_size).to(device)
            # small weight decay helps stability
            optimizer = torch.optim.Adam(ode_func.parameters(), lr=lr, weight_decay=1e-4)
            loss_fn = nn.MSELoss()

            # === Build dataset on CPU, batch to GPU on-the-fly ===
            X = torch.stack(y_train_list)  # [N, L, D] on CPU
            dataset = torch.utils.data.TensorDataset(X)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

            # Shared normalized time grid [0,1] with L points
            L = seq_len
            t_common = torch.linspace(0., 1., steps=L, device=device)

            best_eval_loss = float('inf')
            best_model_state = None

            for epoch in range(niters):  # epochs (kept your variable name)
                epoch_loss = 0.0
                for (y_batch,) in loader:
                    y_batch = y_batch.to(device)        # [B, L, D]
                    y0 = y_batch[:, 0, :]               # [B, D]

                    # === Single batched odeint call (fast) on shared grid, fixed-step RK4 ===
                    pred_y = odeint(
                        ode_func, y0, t_common,
                        method="rk4",
                        options={"step_size": 1.0 / (L - 1)}
                    ).transpose(0, 1)                   # [B, L, D]

                    # Drop trivial t0
                    loss = loss_fn(pred_y[:, 1:, :], y_batch[:, 1:, :])

                    optimizer.zero_grad()
                    loss.backward()
                    # gentle grad clipping
                    torch.nn.utils.clip_grad_norm_(ode_func.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()

                # === Vectorized evaluation on all eval windows using the same t_common ===
                with torch.no_grad():
                    if len(y_eval_list) > 0:
                        Y = torch.stack(y_eval_list).to(device)   # [Ne, L, D]
                        y0v = Y[:, 0, :]                          # [Ne, D]
                        pred_eval = odeint(
                            ode_func, y0v, t_common,
                            method="rk4",
                            options={"step_size": 1.0 / (L - 1)}
                        ).transpose(0, 1)                         # [Ne, L, D]
                        # compute eval in normalized space (consistent selection)
                        avg_eval_loss = loss_fn(pred_eval[:, 1:, :], Y[:, 1:, :]).item()
                    else:
                        avg_eval_loss = float('inf')

                    if avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        # deep copy the best state
                        best_model_state = {k: v.detach().cpu().clone() for k, v in ode_func.state_dict().items()}

                print(f"Epoch {epoch+1:4d} | Train Loss: {epoch_loss / max(1,len(loader)):.6f} | Eval Loss: {avg_eval_loss:.6f}")

                # Write progress for distributed worker
                if self.batch_id:
                    progress_file = Path(__file__).parent / f"progress_{self.batch_id}.yaml"
                    with open(progress_file, 'w') as pf:
                        yaml.dump({'epoch': epoch + 1, 'total_epochs': niters, 'pair_id': self.pair_id}, pf)

            # === Load best model ===
            if best_model_state is not None:
                ode_func.load_state_dict(best_model_state)

            # === Final prediction ===
            # Use provided init if present; else use the last eval window’s last state
            if self.init_data is None:
                if self.prediction_timesteps[0] == self.training_timesteps[0][0]:
                    print("reconstruction")
                    init_state = y_train_list[0][0].to(device)
                else:
                    print("prediction")
                    init_state = y_eval_list[-1][-1].to(device)
            else:
                init_tensor = torch.tensor(self.init_data, dtype=torch.float32, device=device)
                norm_init   = normalizer.normalize(init_tensor)
                init_state  = norm_init if norm_init.ndim == 1 else norm_init[-1]  # [D]

            # Prediction grid: send to device, scale for RK4 step sizing
            t_pred = self.prediction_timesteps.to(device).float()
            # FIX: Normalize by training time span (not prediction span) for consistent dynamics
            T_train = self.training_timesteps[0][-1] - self.training_timesteps[0][0]
            t_pred = (t_pred - t_pred[0]) / (T_train + 1e-12)

            pred_traj = odeint(
                ode_func, init_state, t_pred,
                method="rk4",
                # FIX: Use same step_size as training for consistent integration
                options={"step_size": 1.0 / (seq_len - 1)}
            )  # [Lp, D]
            pred_traj = normalizer.denormalize(pred_traj)
            return np.array(pred_traj.detach().cpu())

        pred = train_neural_ode(
            t_train, y_train, t_eval, y_eval,
            niters=self.num_epochs, lr=self.lr, batch_size=self.batch_size, seq_len=self.seq_len
        )
        return pred