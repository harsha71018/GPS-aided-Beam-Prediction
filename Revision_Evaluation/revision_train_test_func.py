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
