# -*- coding: utf-8 -*-
"""
6G BEAM PREDICTION - REVISION TRAIN/TEST TOOLBOX (revision_train_test_func.py)
=============================================================================
Module: Revision_Evaluation (Tier 4)
Features:
- Continuous Trigonometric Geometric Features (sin AoA, cos AoA, Range, Δx, Δy)
- Leakage-Free Normalization (StandardScaler fit strictly on training split)
- Flexible Partitioning: 'chronological' (extrapolation) & 'random' (interpolation)
- Circular Moving Block Bootstrap (Politis & Romano 1992)
- Isotropic 2D Cartesian Gaussian GPS Noise
- Deep Neural Network Architecture with BatchNorm & Dropout
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utm
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# 1. KINEMATIC & GEOMETRIC CONTINUOUS FEATURE EXTRACTION
# ==============================================================================
def extract_kinematic_features(pos_bs, pos_veh):
    """
    Computes 7 real-world physical features in meters and radians using UTM:
    - Feature 0: Vehicle UTM Easting (x)
    - Feature 1: Vehicle UTM Northing (y)
    - Feature 2: Relative Easting (dx = x_veh - x_bs)
    - Feature 3: Relative Northing (dy = y_veh - y_bs)
    - Feature 4: Euclidean Distance (range in meters)
    - Feature 5: sin(Azimuth Angle) [continuous circular encoding]
    - Feature 6: cos(Azimuth Angle) [continuous circular encoding]
    """
    veh_x, veh_y, zone_num, zone_let = utm.from_latlon(pos_veh[:, 0], pos_veh[:, 1])
    bs_x, bs_y, _, _ = utm.from_latlon(pos_bs[:, 0], pos_bs[:, 1])

    dx = veh_x - bs_x
    dy = veh_y - bs_y
    distance = np.sqrt(dx**2 + dy**2)
    azimuth = np.arctan2(dy, dx)

    features = np.column_stack([
        veh_x,
        veh_y,
        dx,
        dy,
        distance,
        np.sin(azimuth),
        np.cos(azimuth)
    ])
    return features, (zone_num, zone_let)


# ==============================================================================
# 2. GPS NOISE INJECTION IN UTM (METERS)
# ==============================================================================
def add_pos_noise_utm(pos_veh, noise_std_m=1.0):
    """
    Applies isotropic 2D Gaussian perturbation directly in UTM metric coordinates:
    dx_noise ~ N(0, noise_std_m)
    dy_noise ~ N(0, noise_std_m)
    """
    veh_x, veh_y, zone_num, zone_let = utm.from_latlon(pos_veh[:, 0], pos_veh[:, 1])
    n_samples = len(veh_x)

    dx_noise = np.random.normal(0, noise_std_m, n_samples)
    dy_noise = np.random.normal(0, noise_std_m, n_samples)

    noisy_x = veh_x + dx_noise
    noisy_y = veh_y + dy_noise

    noisy_lat, noisy_lon = utm.to_latlon(noisy_x, noisy_y, zone_num, zone_let)
    return np.column_stack([noisy_lat, noisy_lon])


# ==============================================================================
# 3. LEAKAGE-FREE TRAIN / TEST SPLIT
# ==============================================================================
def split_and_scale_data(features, labels, split_mode='chronological', test_size=0.2, random_state=42):
    """
    Splits features and applies StandardScaler fitted STRICTLY on the training split.
    - 'chronological': First (1 - test_size) for train, last test_size for test (deployment extrapolation)
    - 'random': Standard random shuffle split (spatial interpolation)
    """
    n_samples = len(features)
    indices = np.arange(n_samples)

    if split_mode == 'chronological':
        split_point = int(n_samples * (1 - test_size))
        idx_train, idx_test = indices[:split_point], indices[split_point:]
        x_tr, x_te = features[:split_point], features[split_point:]
        y_tr, y_te = labels[:split_point], labels[split_point:]
    else:
        from sklearn.model_selection import train_test_split
        x_tr, x_te, y_tr, y_te, idx_train, idx_test = train_test_split(
            features, labels, indices,
            test_size=test_size, random_state=random_state
        )

    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr)
    x_te_scaled = scaler.transform(x_te)

    return x_tr_scaled, x_te_scaled, y_tr, y_te, idx_train, idx_test, scaler


# ==============================================================================
# 4. CIRCULAR MOVING BLOCK BOOTSTRAP (Politis & Romano 1992)
# ==============================================================================
def circular_mbb_indices(n, block_size):
    """
    Circular Moving Block Bootstrap: wraps indices modulo n to eliminate
    finite-sample boundary underweighting on the extrapolation tail.
    """
    n_blocks = int(np.ceil(n / block_size))
    starts = np.random.randint(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block_size)) % n for s in starts])
    return idx[:n]


# ==============================================================================
# 5. NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class Advanced_NN_FCN(nn.Module):
    def __init__(self, num_features=7, num_output=64, nodes_per_layer=256, n_layers=5, dropout_rate=0.2):
        super(Advanced_NN_FCN, self).__init__()
        self.layer_in = nn.Linear(num_features, nodes_per_layer)
        self.bn_in = nn.BatchNorm1d(nodes_per_layer)

        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.hidden_layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            self.bn_layers.append(nn.BatchNorm1d(nodes_per_layer))

        self.layer_out = nn.Linear(nodes_per_layer, num_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn_in(self.layer_in(x)))
        for layer, bn in zip(self.hidden_layers, self.bn_layers):
            x = self.relu(bn(layer(x)))
            x = self.dropout(x)
        x = self.layer_out(x)
        return x


class BeamDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_net(x_test, model, top_k=1):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_test).float().to(device)
        outputs = model(x_tensor)
        top_k_preds = torch.topk(outputs, top_k, dim=1).indices.cpu().numpy()
    return top_k_preds


def train_net(x_train, y_train, x_val, y_val, run_folder, num_epochs=50, model=None,
              batch_size=32, lr=0.005, weight_decay=1e-5, backup_best_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = Advanced_NN_FCN(num_features=x_train.shape[1], num_output=64, nodes_per_layer=256, n_layers=5, dropout_rate=0.2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    train_loader = DataLoader(BeamDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BeamDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    best_acc = 0.0
    os.makedirs(run_folder, exist_ok=True)
    best_model_path = os.path.join(run_folder, 'best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            if backup_best_model:
                torch.save(model.state_dict(), best_model_path)

    if not backup_best_model or not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    return best_model_path


# ==============================================================================
# 6. UNIVERSAL BACKWARD-COMPATIBLE MODEL CHECKPOINT LOADER
# ==============================================================================
def load_torch_checkpoint(path, map_location):
    """
    Loads a PyTorch checkpoint with backward and forward compatibility across PyTorch versions.
    Supplies weights_only=True if supported by the PyTorch runtime to avoid CVE-2024-33828 / FutureWarning.
    """
    import inspect
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def load_nn_model_checkpoint(ckpt_path, device, num_features=7, num_output=64, nodes_per_layer=256, n_layers=5, dropout_rate=0.2):
    """
    Instantiates and loads an Advanced_NN_FCN model from a checkpoint path.
    Raises FileNotFoundError if the checkpoint does not exist.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Pretrained model checkpoint not found at: {ckpt_path}\n"
            f"Please verify that saved_folder/Advanced_ML_Viz_Seeded_1789723135/ is tracked and available, "
            f"or specify a valid directory via --checkpoint-dir."
        )
    model = Advanced_NN_FCN(
        num_features=num_features,
        num_output=num_output,
        nodes_per_layer=nodes_per_layer,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    )
    model.load_state_dict(load_torch_checkpoint(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ==============================================================================
# 7. LINK OUTAGE PROBABILITY & PAIRED MCNEMAR TEST EVALUATOR
# ==============================================================================
def compute_outage_and_mcnemar(nn_loss, xgb_loss, thresh=3.0):
    """
    Computes paired outage counts, rates, and exact two-sided McNemar binomial test
    p-value across matched test frames (zero pseudoreplication).

    Parameters:
        nn_loss (np.ndarray): Per-frame beamforming power loss for NN (shape: N_test)
        xgb_loss (np.ndarray): Per-frame beamforming power loss for XGBoost (shape: N_test)
        thresh (float): Outage threshold in dB (e.g., 3.0 dB or 6.0 dB)

    Returns:
        dict: Outage statistics including nn_cnt, xgb_cnt, nn_pct, xgb_pct, b_disc, c_disc, p_mcnemar
    """
    import scipy.stats as stats
    nn_loss = np.asarray(nn_loss)
    xgb_loss = np.asarray(xgb_loss)
    n_test = len(nn_loss)

    nn_out = (nn_loss > thresh)
    xgb_out = (xgb_loss > thresh)

    nn_cnt = int(np.sum(nn_out))
    xgb_cnt = int(np.sum(xgb_out))
    nn_pct = (nn_cnt / n_test) * 100.0 if n_test > 0 else 0.0
    xgb_pct = (xgb_cnt / n_test) * 100.0 if n_test > 0 else 0.0

    # Discordant frame pairs:
    # b: XGBoost in outage, NN NOT in outage (NN succeeds where XGB fails)
    # c: NN in outage, XGBoost NOT in outage (XGB succeeds where NN fails)
    b_disc = int(np.sum((~nn_out) & xgb_out))
    c_disc = int(np.sum(nn_out & (~xgb_out)))

    if (b_disc + c_disc) > 0:
        p_mcnemar = float(stats.binomtest(b_disc, b_disc + c_disc, 0.5).pvalue)
    else:
        p_mcnemar = 1.0

    return {
        "n_test": n_test,
        "nn_cnt": nn_cnt,
        "xgb_cnt": xgb_cnt,
        "nn_pct": nn_pct,
        "xgb_pct": xgb_pct,
        "b_disc": b_disc,
        "c_disc": c_disc,
        "p_mcnemar": p_mcnemar
    }



