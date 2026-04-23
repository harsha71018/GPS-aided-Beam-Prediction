# -*- coding: utf-8 -*-
"""
6G BEAM PREDICTION TOOLBOX (train_test_func.py)
-----------------------------------------------
Contains:
- Neural Network Class (NN_FCN)
- Training/Testing Functions (train_net, test_net)
- UTM-based GPS Noise (Your Logic)
- Normalization
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utm

# ==========================================
# 🧠 NEURAL NETWORK CLASS
# ==========================================
class NN_FCN(nn.Module):
    def __init__(self, num_features, num_output, nodes_per_layer, n_layers):
        super(NN_FCN, self).__init__()
        self.layer_in = nn.Linear(num_features, nodes_per_layer)
        self.layers = nn.ModuleList()
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
        self.layer_out = nn.Linear(nodes_per_layer, num_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer_in(x))
        for layer in self.layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.layer_out(x)
        return x

# ==========================================
# 🏋️ TRAINING FUNCTIONS
# ==========================================
class Data_set(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def train_net(x_train, y_train, x_val, y_val, run_folder, num_epochs, model, batch_size, lr, decay_L2, top_stats, rnd_seed=0, fixed_GPU=False, backup_best_model=True, save_all_pred_labels=False, make_plots=False):
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay_L2)
    
    train_loader = DataLoader(Data_set(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Data_set(x_val, y_val), batch_size=batch_size, shuffle=False)
    
    best_acc = 0
    best_model_path = os.path.join(run_folder, 'best_model.pth')
    
    print(f"   > Training NN for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            if backup_best_model:
                torch.save(model.state_dict(), best_model_path)
                
    if not backup_best_model: 
        torch.save(model.state_dict(), best_model_path)
        
    return best_model_path

def test_net(x_test, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    test_tensor = torch.from_numpy(x_test).float().to(device)
    with torch.no_grad():
        outputs = model(test_tensor)
        # Get Top-5 predictions
        _, top_k_preds = torch.topk(outputs, 5, dim=1)
    
    return top_k_preds.cpu().numpy()

# ==========================================
# 🛠️ UTILS (Noise, Normalization)
# ==========================================
def get_experiment_name(scen_idx, n_beams, norm_type, noise):
    return f'scenario {scen_idx} beams {n_beams} norm {norm_type} noise {noise}'

def min_max(arr, ax=0):
    return (arr - arr.min(axis=ax)) / (arr.max(axis=ax) - arr.min(axis=ax))

def add_pos_noise(pos, noise_variance_in_m=1):
    if noise_variance_in_m == 0: return pos
    n_samples = pos.shape[0]
    dist = np.random.normal(0, noise_variance_in_m, n_samples)
    ang = np.random.uniform(0, 2*np.pi, n_samples)
    xy_noise = np.stack((dist * np.cos(ang), dist * np.sin(ang)), axis=1)
    
    x, y, zn, zl = utm.from_latlon(pos[:,0], pos[:,1])
    xy_pos = np.stack((x,y), axis=1) + xy_noise
    lat, long = utm.to_latlon(xy_pos[:,0], xy_pos[:,1], zn, zl)
    return np.stack((lat, long), axis=1)

def normalize_pos(pos1, pos2, norm_type=1):
    return min_max(pos2)