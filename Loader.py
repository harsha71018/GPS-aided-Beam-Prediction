# -*- coding: utf-8 -*-
"""
6G BEAM PREDICTION - FINAL PROJECT PIPELINE
--------------------------------------------
Algorithms  : KNN, NN, RF, XGB, NB
Scenarios   : 3 (Day-A, Night, Day-B) at 60 GHz
Outputs (13):
   1.  Top-1 Accuracy Bar Chart
   2.  Top-5 Accuracy Bar Chart
   3.  Allocation Gain Bar Chart
   4.  Jain Fairness Index Table
   5.  Power Loss Comparison Table (per Scenario)
   6.  Correlation Heatmap
   7-10. Accuracy vs Allocation Gain (4 Strategies)
   11. Latency (Log Scale)
   12. Robustness Drop Bar Chart
   13. Radar Chart
   + Final_Project_Results_Full.csv
"""

# ==============================================================================
# IMPORTS — torch must be imported here, before set_global_seeds() is called
# ==============================================================================
import os
import time
import random
import numpy as np
import torch                          
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import pi
import itertools
import train_test_func as func

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# BULLETPROOF REPRODUCIBILITY (GLOBAL SEEDS)
# ------------------------------------------------------------------------------
# By setting these global seeds, we force Python, NumPy, and PyTorch's cuDNN
# backend to use the exact same pseudo-random number sequence every single time.
# This ensures our Neural Network initializes with the exact same starting
# weights, making the metrics reported in our IEEE paper 100% reproducible
# on any machine.
# ==============================================================================
def set_global_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Lock the environment before doing any math
set_global_seeds(42)

# ==========================================#
#         METRICS & ALLOCATION             #
# ==========================================#
def run_resource_allocation(pred_beams, true_pwr_matrix, strategy="Max-SNR", n_users=10):
    n_samples    = len(pred_beams)
    n_slots      = n_samples // n_users
    slot_rates   = []
    users_covered      = 0
    total_users_count  = 0

    for slot in range(n_slots):
        u_start   = slot * n_users
        user_snrs = []
        for i in range(n_users):
            u_idx  = u_start + i
            p_beam = pred_beams[u_idx]
            if isinstance(p_beam, (list, np.ndarray)):
                p_beam = p_beam[0]
            pwr = true_pwr_matrix[u_idx, int(p_beam)]
            snr = pwr / 1e-9
            user_snrs.append(snr)
            if snr > 10:
                users_covered += 1
            total_users_count += 1

        user_snrs = np.array(user_snrs)

        if strategy == "Round-Robin":
            rates = np.log2(1 + user_snrs) / n_users
        elif strategy == "Max-SNR":
            rates = np.log2(1 + user_snrs)
        elif strategy == "Max-Min":
            rates = np.full(n_users, np.log2(1 + np.min(user_snrs)))
        elif strategy == "Proportional-Fair":
            
            
            snr_sum = np.sum(user_snrs) + 1e-9
            weights = user_snrs / snr_sum
            rates   = weights * np.log2(1 + user_snrs)

        slot_rates.extend(rates)

    rates          = np.array(slot_rates)
    fairness       = (np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2) + 1e-9)
    avg_throughput = np.mean(rates)
    coverage       = users_covered / (total_users_count + 1e-9)

    return avg_throughput, fairness, coverage

# ==========================================#
#           CONTROL PANEL                  #
# ==========================================#
gathered_data_folder = os.path.join(os.getcwd(), 'Gathered_data_DEV')
save_folder          = os.path.join(os.getcwd(), f'saved_folder/Final_ML_Viz_{int(time.time())}')

ai_strategies = ['KNN', 'NN', 'RF', 'XGB', 'NB']
scen_idxs     = [1, 2, 3]

SCENARIO_NAMES = {
    1: "Scenario 1 (V2I - Day - Location A)",
    2: "Scenario 2 (V2I - Night)",
    3: "Scenario 3 (V2I - Day - Location B)"
}

n_beams_list    = [64]
norm_types      = [1]
n_reps          = 1
nodes_per_layer = 256
layers          = 3 + 2
train_batch_size = 32
lr              = 0.01
decay_L2        = 1e-5
num_epochs      = 60

combinations      = list(itertools.product(scen_idxs, n_beams_list, norm_types, [0], [1] * n_reps))
results_db        = []
power_loss_dist_db = []

# ==========================================#
#             MAIN LOOP                    #
# ==========================================#
for scen_idx, n_beams, norm_type, _, rep in combinations:

    scen_name = SCENARIO_NAMES.get(scen_idx, f"Scenario {scen_idx}")
    print(f'\n' + '=' * 40)
    print(f"EXECUTING: {scen_name}")
    print('=' * 40)

    experiment_name = func.get_experiment_name(scen_idx, n_beams, norm_type, 0)
    saved_path      = os.path.join(save_folder, experiment_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    try:
        scen_str = f'scenario{scen_idx}'
        files    = os.listdir(gathered_data_folder)
        p1_f     = [f for f in files if f"{scen_str}_unit1_loc" in f][0]
        pwr_f    = [f for f in files if f"{scen_str}_unit1_pwr" in f][0]
        p2_cand  = [f for f in files if f"{scen_str}_unit2_loc_cal" in f]
        if not p2_cand:
            p2_cand = [f for f in files if f"{scen_str}_unit2_loc" in f]
        p2_f = p2_cand[0]

        pos1 = np.load(os.path.join(gathered_data_folder, p1_f))[:, :2]
        pos2 = np.load(os.path.join(gathered_data_folder, p2_f))[:, :2]
        pwr1 = np.load(os.path.join(gathered_data_folder, pwr_f))
    except Exception as e:
        print(f"Skipping {scen_name}: {e}")
        continue

    # Preprocess
    pos_norm_clean = func.normalize_pos(pos1, pos2, norm_type)
    pos2_noisy     = func.add_pos_noise(pos2, 1.0)
    pos_norm_noisy = func.normalize_pos(pos1, pos2_noisy, norm_type)

    beam_idxs   = np.arange(0, pwr1.shape[-1], pwr1.shape[-1] // n_beams)
    beam_pwrs   = pwr1[:, beam_idxs]
    beam_labels = np.argmax(beam_pwrs, axis=1)

    # ---- Loop Algorithms ----
    for algo in ai_strategies:
        if algo != 'NN':
            print(f"   > Training {algo}...")

        run_folder = os.path.join(saved_path, algo)
        if algo == 'NN':
            if not os.path.exists(run_folder):
                os.makedirs(run_folder)

        indices = np.arange(len(pos_norm_clean))
        x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
            pos_norm_clean, beam_labels, indices,
            test_size=0.2, random_state=42
        )

        pwr_test      = beam_pwrs[idx_test]
        x_test_noisy  = pos_norm_noisy[idx_test]

        start_t          = time.time()
        pred_probs       = []
        pred_beams_noisy = []

        if algo == 'KNN':
            clf        = KNeighborsClassifier(n_neighbors=5)
            clf.fit(x_train, y_train)
            pred_probs = clf.predict_proba(x_test)
            pred_beams = np.argsort(pred_probs, axis=1)[:, ::-1]
            pred_beams_noisy = clf.predict(x_test_noisy)

        elif algo == 'RF':
            clf        = RandomForestClassifier(n_estimators=50, n_jobs=-1)
            clf.fit(x_train, y_train)
            pred_probs = clf.predict_proba(x_test)
            pred_beams = np.argsort(pred_probs, axis=1)[:, ::-1]
            pred_beams_noisy = clf.predict(x_test_noisy)

        elif algo == 'XGB':
            le           = LabelEncoder()
            y_train_enc  = le.fit_transform(y_train)
            clf          = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(le.classes_),
                n_jobs=-1
            )
            clf.fit(x_train, y_train_enc)
            pred_probs   = clf.predict_proba(x_test)
            pred_beams   = np.argsort(pred_probs, axis=1)[:, ::-1]
            pred_beams_noisy = clf.predict(x_test_noisy)

        elif algo == 'NB':
            clf        = GaussianNB()
            clf.fit(x_train, y_train)
            pred_probs = clf.predict_proba(x_test)
            pred_beams = np.argsort(pred_probs, axis=1)[:, ::-1]
            pred_beams_noisy = clf.predict(x_test_noisy)

        elif algo == 'NN':
            x_tr, x_val, y_tr, y_val = train_test_split(
                x_train, y_train, test_size=0.25
            )
            model        = func.NN_FCN(x_train.shape[1], n_beams, nodes_per_layer, layers)
            trained_path = func.train_net(
                x_tr, y_tr, x_val, y_val, run_folder,
                num_epochs, model, train_batch_size, lr, decay_L2,
                top_stats=5, backup_best_model=True
            )
            model.load_state_dict(torch.load(trained_path))
            pred_beams       = func.test_net(x_test, model)
            pred_beams_noisy = func.test_net(x_test_noisy, model)[:, 0]

        train_time = time.time() - start_t

        # FIX 3: measure inference time for ALL algorithms including KNN
        t0 = time.time()
        if algo == 'NN':
            # NN inference: single forward pass on 100 samples
            func.test_net(x_test[:100], model)
        else:
            clf.predict(x_test[:100])
        infer_time = (time.time() - t0) / 100

        # Accuracy metrics
        top1_hits = 0
        top5_hits = 0
        for i in range(len(y_test)):
            if y_test[i] == pred_beams[i][0]:
                top1_hits += 1
            if y_test[i] in pred_beams[i][:5]:
                top5_hits += 1

        top1_acc = (top1_hits / len(y_test)) * 100
        top5_acc = (top5_hits / len(y_test)) * 100

        acc_noisy = np.mean(pred_beams_noisy == y_test) * 100
        rob_drop  = max(0, top1_acc - acc_noisy)

        # Power loss
        losses = []
        for i in range(len(y_test)):
            loss = 10 * np.log10(
                (pwr_test[i, y_test[i]] + 1e-12) /
                (pwr_test[i, int(pred_beams[i][0])] + 1e-12)
            )
            losses.append(loss)
            power_loss_dist_db.append({
                "Algorithm":    algo,
                "Scenario":     scen_name,
                "PowerLoss_dB": loss
            })

        avg_loss   = np.mean(losses)
        strategies = ["Max-SNR", "Round-Robin", "Proportional-Fair", "Max-Min"]

        for strat in strategies:
            thru, fair, cov = run_resource_allocation(
                pred_beams[:, 0], pwr_test, strategy=strat
            )
            results_db.append({
                "Algorithm":      algo,
                "Scenario":       scen_name,
                "Strategy":       strat,
                "Top1_acc":       top1_acc,
                "Top5_acc":       top5_acc,
                "PowerLoss_dB":   avg_loss,
                "AllocGain":      thru,
                "Fairness":       fair,
                "Coverage":       cov,
                "TrainTime":      train_time,
                "InferTime":      infer_time,
                "RobustnessDrop": rob_drop
            })

# ==========================================#
#         FINAL PLOTTING SUITE             #
# ==========================================#
if len(results_db) > 0:
    df      = pd.DataFrame(results_db)
    df_loss = pd.DataFrame(power_loss_dist_db)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width',       2000)
    pd.set_option('display.max_rows',    None)

    df.to_csv(os.path.join(save_folder, "Final_Project_Results_Full.csv"), index=False)

    print("\n" + "=" * 80)
    print("FINAL PROJECT RESULTS (Grouped by Scenario)")
    print("=" * 80)

    for scen in df['Scenario'].unique():
        print(f"\n{scen}")
        print("-" * 80)
        subset = df[df['Scenario'] == scen].drop(columns=['Scenario'])
        cols_to_print = [
            "Algorithm", "Strategy",
            "Top1_acc", "Top5_acc", "PowerLoss_dB",
            "AllocGain", "Fairness", "Coverage",
            "TrainTime", "InferTime", "RobustnessDrop"
        ]
        print(subset[cols_to_print].round(4).to_string(index=False))

    print("\nGENERATING FINAL VISUALIZATIONS...")

    # 1. Top-1 Accuracy
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Algorithm", y="Top1_acc", hue="Scenario", palette="viridis")
    plt.title("Top-1 Accuracy per Scenario")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(save_folder, "1_Top1_Accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Top-5 Accuracy
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Algorithm", y="Top5_acc", hue="Scenario", palette="plasma")
    plt.title("Top-5 Accuracy per Scenario")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(save_folder, "2_Top5_Accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Allocation Gain
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Strategy", y="AllocGain", hue="Algorithm", palette="rocket")
    plt.title("Allocation Gain by Strategy (Avg across Scenarios)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "3_Alloc_Gain.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Fairness Index TABLE
    plt.figure(figsize=(12, 6))
    ax_table = plt.subplot(111, frame_on=False)
    ax_table.xaxis.set_visible(False)
    ax_table.yaxis.set_visible(False)
    fair_table = df.pivot_table(
        index="Algorithm", columns="Strategy", values="Fairness"
    ).round(5)
    tab = pd.plotting.table(ax_table, fair_table, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(14)
    tab.scale(1.2, 1.5)
    plt.title("Jain's Fairness Index Table (Higher is Better)", fontsize=16, weight='bold')
    plt.savefig(os.path.join(save_folder, "4_Fairness_Table.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Power Loss Comparison TABLE — per Scenario (faculty requirement)
    # FIX 4: simple per-scenario mean power loss table replacing the old stats table
    plt.figure(figsize=(14, 5))
    ax_table = plt.subplot(111, frame_on=False)
    ax_table.xaxis.set_visible(False)
    ax_table.yaxis.set_visible(False)

    s1 = "Scenario 1 (V2I - Day - Location A)"
    s2 = "Scenario 2 (V2I - Night)"
    s3 = "Scenario 3 (V2I - Day - Location B)"

    loss_table = df_loss.groupby(
        ["Algorithm", "Scenario"]
    )["PowerLoss_dB"].mean().unstack()
    loss_table = loss_table[[s1, s2, s3]].round(2)
    loss_table.columns = ["Scenario 1 (Day-A)", "Scenario 2 (Night)", "Scenario 3 (Day-B)"]
    loss_table["Average (dB)"] = loss_table.mean(axis=1).round(2)

    tab = pd.plotting.table(
        ax_table, loss_table, loc='center', cellLoc='center',
        colWidths=[0.22, 0.22, 0.22, 0.22]
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(13)
    tab.scale(1.2, 1.8)
    plt.title("Mean Power Loss (dB) per Algorithm and Scenario", fontsize=15, weight='bold')
    plt.savefig(os.path.join(save_folder, "5_Power_Loss_Table.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    cols = ["Top1_acc", "AllocGain", "Fairness", "Coverage", "TrainTime", "InferTime"]
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(save_folder, "6_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 7–10. Accuracy vs Allocation Gain (one chart per strategy)
    strategies = ["Max-SNR", "Round-Robin", "Proportional-Fair", "Max-Min"]
    for i, strat in enumerate(strategies):
        fig, ax1 = plt.subplots(figsize=(12, 8))
        subset   = df[df["Strategy"] == strat].groupby("Algorithm")[
            ["Top1_acc", "AllocGain"]
        ].mean()
        width = 0.4
        x     = np.arange(len(subset))

        ax1.bar(x - width / 2, subset["Top1_acc"], width=width,
                label='Accuracy', color='#3498db', alpha=0.9)
        ax1.set_ylabel('Top-1 Accuracy (%)', color='#3498db', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#3498db')
        ax1.set_xlabel('Algorithm', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(subset.index, fontsize=11)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax1.grid(False, axis='x')

        ax2 = ax1.twinx()
        ax2.bar(x + width / 2, subset["AllocGain"], width=width,
                label='Gain', color='#e67e22', alpha=0.9)
        ax2.set_ylabel('Allocation Gain', color='#e67e22', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#e67e22')
        ax2.grid(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper left', bbox_to_anchor=(1.1, 1))

        plt.title(f"Accuracy vs Allocation Gain ({strat})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{7 + i}_Acc_vs_Gain_{strat}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 11. Latency (Log Scale)
    plt.figure(figsize=(10, 6))
    df.groupby("Algorithm")[["TrainTime", "InferTime"]].mean().plot(
        kind="bar", logy=True
    )
    plt.title("Latency (Log Scale)")
    plt.savefig(os.path.join(save_folder, "11_Latency.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 12. Robustness Drop
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df, x="Algorithm", y="RobustnessDrop",
        hue="Scenario", palette="viridis", errorbar=None
    )
    plt.title("Accuracy Drop (Noise 1m) by Scenario")
    plt.savefig(os.path.join(save_folder, "12_Robustness.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 13. Radar Chart
    metrics = [
        "Top1_acc", "AllocGain", "Fairness", "Coverage",
        "TrainTime", "InferTime", "RobustnessDrop"
    ]
    norm_df = df.groupby("Algorithm")[metrics].mean()
    norm_df["TrainTime"]       = 1 / (norm_df["TrainTime"]       + 1e-9)
    norm_df["InferTime"]       = 1 / (norm_df["InferTime"]       + 1e-9)
    norm_df["RobustnessDrop"]  = 1 / (norm_df["RobustnessDrop"]  + 1e-9)
    norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min() + 1e-9)

    N      = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for algo in norm_df.index:
        values  = norm_df.loc[algo].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=algo)
        ax.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], ["Acc", "Gain", "Fair", "Cov", "Train", "Infer", "Robust"])
    plt.title("Algorithm Performance Radar (Avg)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(save_folder, "13_Radar_Chart.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nDONE! All 13 outputs saved to: {save_folder}")

else:
    print("\nFAILED — No results to process.")