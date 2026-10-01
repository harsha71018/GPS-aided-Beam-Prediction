# Revision Evaluation: Chronological Vehicular Extrapolation & Outage Analysis

This module constitutes **Tier 4 (Manuscript Revision Protocol)** of the GPS-aided 6G beam prediction framework.

It provides a fully reproducible evaluation harness that models realistic vehicular deployment by shifting from **spatial interpolation** (random shuffle splits) to **temporal trajectory extrapolation** (chronological trajectory splitting).

---

## 1. Core Findings & Physical Insights

Under chronological trajectory extrapolation, the Deep Neural Network achieves **a robust +10.31 pp to +10.36 pp margin in night conditions (Scenario 2)**, with a **marginal and directionally consistent +6.94 pp in Day-B (Scenario 3)**.

In Scenario 2 (Night), the neural network maintains **35.23%** Top-1 accuracy across an expanded 10-seed empirical distribution (`Revision_10_Seed_Distribution.csv`, standard error $\pm 0.60\text{ pp}$) and **35.18%** across the 3 standard benchmark checkpoints, compared to XGBoost at **24.87%** (a **+10.31 pp** to **+10.36 pp** advantage).

Uncertainty is quantified along two independent dimensions:
1. **Circular Moving-Block Bootstrap (C-MBB)** over the test trajectory (**95% CI [+2.35, +20.17] pp**, $N_{\text{test}} = 595$, strictly excluding zero across all block lengths $L=25, 50, 100$ under paired within-seed resampling). This frame-sampling uncertainty dominates and constitutes the primary basis of statistical significance.
2. **Seed-to-Seed Optimization Variability** ($\text{SD} = 1.90\text{ pp}$ across 10 independent seeds, standard error $\pm 0.60\text{ pp}$, 95% $t$-interval $[33.87\%, 36.59\%]$). This confirms the performance advantage is robust against weight initialization stochasticity.

### Environmental Propagation Regimes
* **Scenario 1 (Day-A, $N_{\text{test}} = 485$) — Open Line-of-Sight (LOS)**:
  On simple straight daylight paths, coordinate-to-beam correspondence is geometrically direct. Both tree ensembles and neural networks achieve equivalent accuracy (**48.45% vs. 48.66%**, a $+0.21\text{ pp}$ tie).
* **Scenario 2 (Night, $N_{\text{test}} = 595$) — Multipath & Low-Light Dynamic Scattering (Robust Advantage)**:
  Under complex nocturnal multipath conditions without visual cues, non-linear reflections and shadowing degrade spatial bijectivity. Rigid orthogonal decision trees fail to extrapolate into unseen terminal coordinates ($24.87\%$), whereas the continuous inductive bias of deep neural networks preserves smooth spatial tracking ($35.23\%$, $35.18\%$), delivering a **robust +10.31 pp margin** and cutting 3 dB beam misalignment outages by **$3.00\times$** ($10.08\%$ down to $3.36\%$, $p = 4.19 \times 10^{-6}$ via two-sided Fisher's Exact Test; corroborated under matched-pairs McNemar test on discordant frames with $b=53, c=3 \implies p = 8.14 \times 10^{-13}$, with per-seed paired evaluations ranging from $p = 3.09 \times 10^{-8}$ to $p = 1.38 \times 10^{-7}$). C-MBB 95% confidence intervals strictly exclude zero across all block lengths ($L=25, 50, 100$).
* **Scenario 3 (Day-B, $N_{\text{test}} = 298$) — Urban Canyon & Blockage (Marginal Directional Trend)**:
  In the shorter Day-B trajectory, the neural network exhibits a $+6.94\text{ pp}$ margin ($30.76\%$ vs. $23.83\%$) and modest outage reduction ($17.79\%$ to $14.09\%$, $p = 0.2631$). While directionally consistent with Scenario 2, the advantage is statistically marginal due to smaller test sample size ($N_{\text{test}} = 298$), with C-MBB intervals touching or crossing zero at two of the three block lengths ($L=25: [+0.00\%, +13.42\%]$; $L=100: [-0.34\%, +12.42\%]$, and significant only at $L=50: [+0.34\%, +13.09\%]$). We report this result transparently as a marginal, directionally consistent trend rather than a conclusive margin.

---

## 2. Summary Benchmark Results

### Table 1: Model Accuracy and Outage Reduction
*(Generated from `Revision_Accuracy_and_Outage_Summary.csv`)*

| Scenario | Test Frames | XGBoost Top-1 | Neural Network Top-1 | Margin | 3 dB Outage (XGB vs NN) | Outage Reduction Ratio | Two-Sided Fisher $p$ | Paired McNemar $p$ ($b, c$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scenario 1 (Day-A)** | 485 | 48.45% | 48.66% | +0.21 pp | 7 vs 5 (1.44% vs 1.03%) | 1.40x reduction | $p = 0.7730$ | $p = 0.6250$ ($b=3, c=1$) |
| **Scenario 2 (Night)** | 595 | **24.87%** | **35.23%** *(10 seeds)* | **+10.36 pp** | **60 vs 20 (10.08% vs 3.36%)** | **3.00x reduction** | **$p = 4.19 \times 10^{-6}$** | **$p = 8.14 \times 10^{-13}$ ($b=53, c=3$)** |
| **Scenario 3 (Day-B)** | 298 | 23.83% | 30.76% | +6.94 pp | 53 vs 42 (17.79% vs 14.09%) | 1.26x reduction | $p = 0.2631$ | $p = 2.32 \times 10^{-3}$ ($b=23, c=6$) |

> **Note on Scenario 2 Seed Reporting & Training Regimes**: The 3-checkpoint benchmark mean is 35.18% (+10.31 pp margin, trained for 60 epochs with cosine annealing), and the expanded 10-seed distribution mean is 35.23% ($\text{SD} = 1.90\text{ pp}$, $\text{SE} = \pm 0.60\text{ pp}$, +10.36 pp margin, trained for 50 epochs), demonstrating that the NN performance advantage remains remarkably consistent across independent weight initializations and training configurations (50 vs. 60 epochs). Per-seed 3 dB outage counts are: NN = [20, 22, 18] (mean 20.00) vs. XGB = [60, 60, 60] (mean 60.00). In Scenario 1 at 6 dB threshold, zero NN outage events were recorded (`n/a (0 events)`). Data partitioning enforces strict out-of-sample holdout test isolation (the 20% test trajectory was never seen during feature scaling or model training).

### Table 2: Circular Moving Block Bootstrap 95% Confidence Intervals
*(Generated from `Revision_Bootstrap_CI_Summary.csv` — Paired Within-Seed Contrast)*

| Scenario | Block Length ($L$) | Mean Margin | 95% Confidence Interval | Standard Error | Excludes Zero? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Scenario 1 (Day-A)** | $L = 25$ frames | +0.21 pp | [-4.12%, +4.74%] | 2.26% | False |
| **Scenario 1 (Day-A)** | $L = 50$ frames | +0.21 pp | [-3.30%, +3.92%] | 1.86% | False |
| **Scenario 1 (Day-A)** | $L = 100$ frames | +0.21 pp | [-3.09%, +3.71%] | 1.76% | False |
| **Scenario 2 (Night)** | $L = 25$ frames | +10.31 pp | **[+1.34%, +20.50%]** | 4.92% | **True** |
| **Scenario 2 (Night)** | $L = 50$ frames | +10.31 pp | **[+1.68%, +20.67%]** | 4.97% | **True** |
| **Scenario 2 (Night)** | $L = 100$ frames | +10.31 pp | **[+2.35%, +20.17%]** | 4.58% | **True** |
| **Scenario 3 (Day-B)** | $L = 25$ frames | +6.94 pp | [+0.00%, +13.42%] | 3.30% | False |
| **Scenario 3 (Day-B)** | $L = 50$ frames | +6.94 pp | [+0.34%, +13.09%] | 3.29% | **True** |
| **Scenario 3 (Day-B)** | $L = 100$ frames | +6.94 pp | [-0.34%, +12.42%] | 3.23% | False |

---

## 3. How to Reproduce

You can run the entire evaluation in **under 50 seconds** from Spyder or command line:

```bash
python Revision_Evaluation/run_revision_evaluation.py
```

### Module Contents
* `revision_train_test_func.py`: Self-contained 7-feature continuous trigonometric feature extraction ($\sin \theta, \cos \theta, \text{Range}, \Delta x, \Delta y$), leakage-free scaler, neural network definition, and self-contained training/evaluation routines.
* `run_revision_evaluation.py`: Reproducible evaluation runner with within-seed paired C-MBB bootstrap and automated 10-seed generator.
* `Revision_Accuracy_and_Outage_Summary.csv`: Sample counts, accuracies, and Fisher exact tests.
* `Revision_Bootstrap_CI_Summary.csv`: Moving Block Bootstrap intervals across block lengths.
* `Revision_10_Seed_Distribution.csv`: 10-seed empirical distribution data ($N=10$), automatically verified or regenerated by `run_revision_evaluation.py`.
* `Power_Loss_CDF_Scenario_*.png`: 300 DPI publication-grade beam power loss CDF curves.
