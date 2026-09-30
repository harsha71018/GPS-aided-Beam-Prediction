# GPS-Based Beam Prediction for 6G Vehicular Networks

**Using the DeepSense 6G Real-World Dataset**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Target Journal](https://img.shields.io/badge/Submitted%20to-Wiley%20IJCS-00416A.svg)](https://onlinelibrary.wiley.com/journal/10991131)
[![Dataset](https://img.shields.io/badge/Dataset-DeepSense%206G-green.svg)](https://deepsense6g.net)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-black.svg)](https://pep8.org/)

This repository contains the source code, evaluation framework, and empirical benchmarks for the research paper:

> *GPS-Based Beam Prediction for 6G Vehicular Networks Using the DeepSense 6G Real-World Dataset*
>
> **Submitted to:** International Journal of Communication Systems (IJCS), Wiley

> [!NOTE]
> **For Peer Reviewers (Wiley IJCS):**  
> To reproduce the exact models, tables, and 13 figures published in the submitted manuscript, please refer directly to **[Tier 1: Original Submitted Baseline](#run-original-submitted-baseline-tier-1)** (`python Loader.py` and `saved_folder/Final_ML_Viz_1776919650/`). The 13 figures correspond 1:1 to the submitted paper; the 14th figure (continuous GPS noise curve) belongs to the post-submission Advanced Pipeline.

### Based On

This project builds upon the foundational work by **Morais et al.** and their original codebase:
- **Paper**: J. Morais et al., *"Position-aided beam prediction in the real world: how useful GPS locations actually are?"*, arXiv:2205.09054, 2022.
- **Original Repository**: [github.com/jmoraispk/Position-Beam-Prediction](https://github.com/jmoraispk/Position-Beam-Prediction)

We extended their KNN and NN baselines by adding Random Forest, XGBoost, and Naive Bayes, and introduced new multi-dimensional evaluation metrics: multi-user resource allocation sum-rate, isotropic GPS noise robustness sweeps, and Jain's Fairness analysis.

---

## Repository Navigation (Three Research Tiers)

This repository documents the chronological progression of our research across three distinct frameworks:

| Tier | Directory | Primary Script | Description | Best NN Top-1 | Best Overall Top-1 |
|:---|:---|:---|:---|:---:|:---:|
| 🚀 **Tier 3: Advanced Pipeline (v2)** *(Recommended)* | [`Advanced_Pipeline/`](Advanced_Pipeline/) | `python advanced_loader.py` | Metric UTM kinematics, zero-leakage scaler, CosineAnnealingLR, continuous noise sweep, 14 standardized plots | **43.48%** *(53.20% S1)* | **43.48%** (NN) |
| 🔧 **Tier 2: Further Tuning** | [`Further tuning/`](Further%20tuning/) | `python deepseekv4pro_loader.py` | Corrected classical ML beam-ID mapping bug (`clf.classes_`) | **37.28%** | **41.89%** (KNN) |
| 🎓 **Tier 1: Original Submitted Baseline** | Root `/` | `python Loader.py` | Historical college baseline submitted to Wiley IJCS | **37.28%** | **37.28%** (NN) |

---

## Visual Highlights (Advanced Pipeline v2)

The figures below are generated directly by the Advanced Pipeline and are stored in [`Advanced_Pipeline/Advanced_ML_Viz_1789648029/`](Advanced_Pipeline/Advanced_ML_Viz_1789648029/):

<p align="center">
  <img src="Advanced_Pipeline/Advanced_ML_Viz_1789648029/13_Radar_Chart.png" width="48%" alt="Algorithm Performance Radar" />
  <img src="Advanced_Pipeline/Advanced_ML_Viz_1789648029/14_Noise_Robustness_Curve.png" width="48%" alt="Continuous GPS Noise Sensitivity Curve" />
</p>
<p align="center">
  <img src="Advanced_Pipeline/Advanced_ML_Viz_1789648029/1_Top1_Accuracy.png" width="48%" alt="Top-1 Accuracy by Scenario" />
  <img src="Advanced_Pipeline/Advanced_ML_Viz_1789648029/6_Heatmap.png" width="48%" alt="Feature Correlation Heatmap" />
</p>

---

## Overview

We compare five machine learning models for GPS-only beam prediction in 60 GHz vehicle-to-infrastructure (V2I) systems:

| Model | Type | Key Configuration |
|---|---|---|
| **KNN** | Instance-based | $k=5$, Euclidean distance, distance-weighted |
| **Random Forest** | Ensemble | 100 trees, bootstrap, max depth 20 |
| **XGBoost** | Gradient Boosting | `multi:softprob`, 64 classes, learning rate 0.08 |
| **Naive Bayes** | Probabilistic | Gaussian likelihood |
| **Neural Network** | Deep Learning | 5-layer FCN (256 nodes/layer), BatchNorm1d, Dropout 0.2, AdamW, 60 epochs |

### Key Results Across Research Tiers (Averaged across 3 scenarios)

| Model | Tier 1 Baseline Top-1 (%) | Tier 2 Further Tuning Top-1 (%) | Tier 3 Advanced v2 Top-1 (%) | Tier 3 Top-5 Acc (%) | Tier 3 Power Loss (dB) | Tier 3 1m Noise Drop (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Neural Network (NN)** | **37.28%** | **37.28%** | **43.48%** *(53.20% S1)* | **87.81%** | **0.62 dB** | **7.79%** |
| **XGBoost** | 5.42% | 41.03% | **43.17%** | 84.10% | 0.67 dB | 9.57% |
| **Random Forest (RF)** | 4.73% | 41.62% | **41.92%** | 84.67% | 0.65 dB | 10.14% |
| **KNN** | 5.87% | 41.89% | **41.09%** | 76.06% | 0.69 dB | 10.49% |
| **Naive Bayes (NB)** | 6.64% | 23.81% | **23.31%** | 68.38% | 1.41 dB | **1.70%** |

---

## Dataset

This project uses the **DeepSense 6G** position-aided beam prediction dataset:
- **Source**: [https://deepsense6g.net](https://deepsense6g.net)
- **Frequency**: 60 GHz mmWave
- **Codebook**: 64 beams (uniform sector array)
- **Scenarios**: 3 V2I outdoor scenarios:
  - Scenario 1: V2I Day - Location A
  - Scenario 2: V2I Night
  - Scenario 3: V2I Day - Location B

### Data Setup

1. Download the position-aided subset from [DeepSense 6G](https://deepsense6g.net).
2. Place the `.npy` files in a folder called `Gathered_data_DEV/` in the project root:

```
Gathered_data_DEV/
├── scenario1_unit1_loc.npy
├── scenario1_unit1_pwr.npy
├── scenario1_unit2_loc_cal.npy
├── scenario2_unit1_loc.npy
├── scenario2_unit1_pwr.npy
├── scenario2_unit2_loc_cal.npy
├── scenario3_unit1_loc.npy
├── scenario3_unit1_pwr.npy
└── scenario3_unit2_loc.npy
```

*(Note: `Gathered_data_DEV/` is included in `.gitignore` to prevent committing large binary data).*

---

## Project Structure

```
├── Loader.py                        # Tier 1: Original submitted baseline pipeline
├── train_test_func.py               # Tier 1: Original neural network & baseline utilities
├── check_env_file.py                # Tier 1: Environment & CUDA verification script
├── requirements.txt                 # Project Python dependencies
├── Gathered_data_DEV/               # Dataset directory (.npy files — download separately)
├── Further tuning/                  # Tier 2: Bug-fixed classical ML framework
│   ├── deepseekv4pro_loader.py      # Refined main pipeline with clf.classes_ fix
│   ├── deepseekv4pro_train_test_func.py
│   ├── deepseelv4pro_check_env_file.py
│   └── Final_ML_Viz_1779380116/     # Post-fix benchmark outputs (13 plots + CSV)
├── Advanced_Pipeline/               # Tier 3: State-of-the-art geometric framework (v2)
│   ├── advanced_loader.py           # Upgraded execution pipeline (14 plots + 2 CSVs)
│   ├── advanced_train_test_func.py  # UTM kinematics, zero-leakage scaler, AdamW
│   ├── advanced_check_env.py        # Environment & GPU verification script
│   └── Advanced_ML_Viz_1789648029/  # Complete verified results package (14 plots + CSVs)
└── saved_folder/                    # Tier 1 output directory
    └── Final_ML_Viz_1776919650/     # Original submitted baseline outputs
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/harsha71018/GPS-aided-Beam-Prediction.git
cd GPS-aided-Beam-Prediction

# 2. Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage (Choose Your Tier)

### 🚀 Recommended: Run Advanced Pipeline (Tier 3)
```bash
cd Advanced_Pipeline

# 1. Verify dependencies, CUDA GPU & dataset
python advanced_check_env.py

# 2. Execute the full geometric pipeline
python advanced_loader.py
```
This generates all **14 standardized visual outputs**, including the Continuous GPS Noise Robustness Curve, and saves results to `saved_folder/Advanced_ML_Viz_<timestamp>/`.

### 🔧 Run Further Tuning (Tier 2)
```bash
cd "Further tuning"
python deepseekv4pro_loader.py
```

### 🎓 Run Original Submitted Baseline (Tier 1)
```bash
python check_env_file.py
python Loader.py
```

---

## Reproducibility

All experiments lock random seeds (`seed=42`) for bit-exact reproducibility across:
- Python `random` module
- NumPy random generator
- PyTorch CPU and CUDA engines
- cuDNN deterministic mode (`torch.backends.cudnn.deterministic = True`)

---

## Evaluation Dimensions

| Dimension | Metric | Telecom Significance |
|---|---|---|
| **Accuracy** | Top-1 and Top-5 beam prediction accuracy | Primary beam alignment rate |
| **Link Quality** | Average beamforming power loss (dB) | Direct measure of received SNR degradation |
| **Fairness** | Jain's Fairness Index across 4 schedulers | Fair resource distribution among multi-user UEs |
| **Robustness** | Continuous noise degradation ($0.5\text{ m} \to 5.0\text{ m}$) | Resilience against GPS jitter and hardware inaccuracies |
| **Efficiency** | Training time (s) and inference latency ($\mu\text{s}$) | Edge-readiness for sub-1 ms URLLC constraints |

---

## Further Tuning: The Classical ML Mapping Bugfix

The [`Further tuning/`](Further%20tuning/) folder documents an important validation and debugging milestone in this research:

### What Changed
The original college baseline (`Loader.py`) contained a **class index mapping bug** in the classical ML models (KNN, RF, XGB, NB). In scikit-learn / XGBoost, `predict_proba()` returns probabilities ordered by the classes present in the training set, not necessarily absolute beam indices $0..63$. The original baseline inadvertently treated probability column index $j$ as beam ID $j$, which produced severe mismatches whenever the training partition did not span all 64 classes.

**Fix Applied** (in `deepseekv4pro_loader.py`):
```python
# Before (incorrect):
pred_beams = np.argsort(pred_probs, axis=1)[:, ::-1]

# After (correct):
sorted_idx = np.argsort(pred_probs, axis=1)[:, ::-1]
pred_beams = clf.classes_[sorted_idx]  # Explicitly map internal indices to physical beam IDs
```

The Neural Network was completely unaffected by this bug as it directly outputs an unconstrained 64-dimensional logits vector.

### Results Comparison — Before vs After Bug Fix

#### Top-1 Accuracy (%) — Averaged across 3 scenarios

| Model | Before (Original) | After (Further Tuning) | Change |
|-------|:--:|:--:|:--:|
| **KNN** | 5.87% | **41.89%** | +36.02% |
| **RF** | 4.73% | **41.62%** | +36.89% |
| **XGB** | 5.42% | **41.03%** | +35.61% |
| **NB** | 6.64% | **23.81%** | +17.17% |
| **NN** | 37.28% | **37.28%** | 0.00% |

> **Key takeaway:** The classical models were severely underreported in the original run. After the fix, KNN, RF, and XGBoost all perform competitively with the Neural Network.

#### Power Loss (dB) — Averaged across 3 scenarios

| Model | Before | After |
|-------|:--:|:--:|
| KNN | 1.77 dB | **0.64 dB** |
| RF | 1.72 dB | **0.64 dB** |
| XGB | 1.80 dB | **0.67 dB** |
| NB | 2.39 dB | **1.87 dB** |
| NN | 0.85 dB | **0.85 dB** |

All models remain well below the 3 dB practical outage threshold.

#### GPS Noise Robustness Drop (%) — 1 m perturbation

| Model | Before | After |
|-------|:--:|:--:|
| KNN | 0.00% | **20.44%** |
| RF | 0.00% | **22.00%** |
| XGB | 1.97% | **21.85%** |
| NB | 0.00% | **3.78%** |
| NN | 11.23% | **11.23%** |

> **Note:** The original "zero degradation" for KNN/RF/NB was an artifact of the bug — incorrect beam IDs happened to be equally wrong with or without noise. With correct beam mapping, classical models show meaningful GPS sensitivity. Naive Bayes remains the most robust (3.78% avg drop).

---

## Advanced Pipeline (v2 — Geometric & Kinematic Framework)

The [`Advanced_Pipeline/`](Advanced_Pipeline/) folder represents the premier evolution of this research:

### Key Architectural Advancements
1. **Kinematic & Geometric Feature Engineering**:
   - Converts raw latitude/longitude coordinates to metric UTM space.
   - Derives Euclidean range ($d = \sqrt{\Delta x^2 + \Delta y^2}$) and Line-of-Sight Azimuth bearing angle ($\theta = \arctan2(\Delta y, \Delta x)$) relative to the base station tower.
2. **Zero-Leakage Preprocessing**:
   - `StandardScaler` is fitted strictly on the training partition and applied out-of-sample to test data.
3. **Continuous Multi-Level GPS Robustness Curve**:
   - Systematically sweeps isotropic noise across $\sigma \in [0.5\text{ m}, 1.0\text{ m}, 2.0\text{ m}, 3.0\text{ m}, 5.0\text{ m}]$ (`14_Noise_Robustness_Curve.png`).
4. **Upgraded Neural Network Architecture**:
   - 5-layer FCN with `BatchNorm1d`, `Dropout(0.2)`, `AdamW`, and `CosineAnnealingLR` scheduling.

### Performance Milestone (Avg across 3 Scenarios)

| Metric | Submitted Paper Baseline | Further Tuning (Bug-Fixed) | Advanced Pipeline (v2) | Milestone Gain |
| :--- | :---: | :---: | :---: | :---: |
| **NN Top-1 Accuracy** | 37.28% | 37.28% | **43.48%** | **+6.20%** (*53.20% in Scen 1*) |
| **NN Top-5 Accuracy** | 85.76% | 85.76% | **87.81%** | **+2.05%** (*96.49% in Scen 1*) |
| **XGBoost Top-1 Accuracy** | 5.42% | 41.03% | **43.17%** | Near parity with NN |
| **Random Forest Top-1** | 4.73% | 41.62% | **41.92%** | High-performance ensemble |
| **KNN Top-1 Accuracy** | 5.87% | 41.89% | **41.09%** | Robust distance-weighted baseline |
| **Average Power Loss (NN)** | 0.85 dB | 0.85 dB | **0.62 dB** | **0.23 dB in Scen 1** |
| **1m Noise Drop (NN)** | 11.23% | 11.23% | **7.79%** | **+3.44% noise resilience** |
| **Inference Latency** | $<1\text{ ms}$ | $<1\text{ ms}$ | **$20\text{ }\mu\text{s}$ (NN)** | Sub-millisecond edge ready |

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{harshavardhan2026gps,
  title={GPS-Based Beam Prediction for 6G Vehicular Networks Using the DeepSense 6G Real-World Dataset},
  author={Dasyapu, Harshavardhan and Danaboyina, Vamshi and Gaikwad, Prasenjith Kumar and Tangelapalli, Swapna},
  journal={International Journal of Communication Systems},
  year={2026},
  publisher={Wiley}
}
```

---

## Acknowledgments

- **Morais et al.** — original [Position-Beam-Prediction](https://github.com/jmoraispk/Position-Beam-Prediction) codebase that inspired this work.
- [DeepSense 6G Team](https://deepsense6g.net) — Wireless Intelligence Lab, Arizona State University, for collecting and hosting the real-world dataset.

---

## License & Usage

This repository is provided strictly for academic, peer-review, and research evaluation purposes. All rights are reserved by the authors.
