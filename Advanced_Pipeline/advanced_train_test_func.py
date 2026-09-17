# -*- coding: utf-8 -*-
"""
6G BEAM PREDICTION TOOLBOX (advanced_train_test_func.py)
-------------------------------------------------------
Key Advancements:
1. Geometric & Kinematic Feature Extraction (Azimuth AoA, Distance, Relative Δx/Δy).
2. Leakage-Free Preprocessing (Fit scaler strictly on train split).
3. Flexible Splitting: 'random' shuffle or 'chronological' trajectory split.
4. Multi-Level Noise Injection (0.5m to 5.0m).
5. Upgraded Neural Network with BatchNorm & CosineAnnealingLR.
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
# 1. KINEMATIC & GEOMETRIC FEATURE EXTRACTION
# ==============================================================================
def extract_kinematic_features(pos_bs, pos_veh):
    """
    Computes real-world physical features in meters and radians using UTM:
    - Feature 0: Vehicle UTM Easting (x)
    - Feature 1: Vehicle UTM Northing (y)
    - Feature 2: Relative Easting (dx = x_veh - x_bs)
    - Feature 3: Relative Northing (dy = y_veh - y_bs)
    - Feature 4: Euclidean Distance (range in meters)
    - Feature 5: Line-of-Sight Azimuth Angle (bearing in radians [-pi, pi])
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
        azimuth
    ])
    return features, (zone_num, zone_let)


# ==============================================================================
# 2. GPS NOISE INJECTION IN UTM (METERS)
# ==============================================================================
def add_pos_noise_utm(pos_veh, noise_std_m=1.0):
    """
    Applies isotropic Gaussian perturbation directly in UTM metric coordinates,
    then transforms back to lat/lon.
    """
    if noise_std_m <= 0:
        return pos_veh.copy()

    n_samples = pos_veh.shape[0]
    r = np.abs(np.random.normal(0, noise_std_m, n_samples))
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    dx_noise = r * np.cos(theta)
    dy_noise = r * np.sin(theta)

    x, y, zn, zl = utm.from_latlon(pos_veh[:, 0], pos_veh[:, 1])
    noisy_x = x + dx_noise
    noisy_y = y + dy_noise

    lat, lon = utm.to_latlon(noisy_x, noisy_y, zn, zl)
    return np.column_stack((lat, lon))


# ==============================================================================
# 3. LEAKAGE-FREE TRAIN / TEST SPLIT
# ==============================================================================
def split_and_scale_data(features, labels, split_mode='random', test_size=0.2, random_state=42):
    """
    Splits features and applies StandardScaler fitted STRICTLY on the training split.
    - 'random': Standard random stratified shuffle
    - 'chronological': First (1 - test_size) for train, last test_size for test
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

    # Scaler is fitted ONLY on training data
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr)
    x_te_scaled = scaler.transform(x_te)

    return x_tr_scaled, x_te_scaled, y_tr, y_te, idx_train, idx_test, scaler


# ==============================================================================
# 4. NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class Advanced_NN_FCN(nn.Module):
    def __init__(self, num_features, num_output, nodes_per_layer=256, n_layers=5, dropout_rate=0.2):
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


def train_net(x_train, y_train, x_val, y_val, run_folder, num_epochs, model,
              batch_size=32, lr=0.005, weight_decay=1e-5, top_stats=5, backup_best_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    train_loader = DataLoader(BeamDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(BeamDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    best_acc = 0.0
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

        # Validation
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


def test_net(x_test, model, top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    test_tensor = torch.from_numpy(x_test).float().to(device)
    with torch.no_grad():
        outputs = model(test_tensor)
        _, top_k_preds = torch.topk(outputs, top_k, dim=1)

    return top_k_preds.cpu().numpy()