# GPS-Based Beam Prediction for 6G Vehicular Networks

**Using the DeepSense 6G Real-World Dataset**

This repository contains the source code for the research paper:

> *GPS-Based Beam Prediction for 6G Vehicular Networks Using the DeepSense 6G Real-World Dataset*

### Based On

This project builds upon the work by **Morais et al.** and their original codebase:
- **Paper**: J. Morais et al., *"Position-aided beam prediction in the real world: how useful GPS locations actually are?"*, arXiv:2205.09054, 2022.
- **Original Repository**: [github.com/jmoraispk/Position-Beam-Prediction](https://github.com/jmoraispk/Position-Beam-Prediction)

We extended their KNN and NN baselines by adding Random Forest, XGBoost, and Naive Bayes, and introduced new evaluation dimensions: resource allocation, GPS noise robustness, and Jain's Fairness analysis.

---

## Overview

We compare five machine learning models for GPS-only beam prediction in 60 GHz vehicle-to-infrastructure (V2I) systems:

| Model | Type | Key Config |
|-------|------|------------|
| **KNN** | Instance-based | k=5, Euclidean distance |
| **Random Forest** | Ensemble | 50 trees, bootstrap |
| **XGBoost** | Boosting | multi:softprob, 64 classes |
| **Naive Bayes** | Probabilistic | Gaussian likelihood |
| **Neural Network** | Deep Learning | 3×256 hidden, Dropout 0.2, Adam, 60 epochs |

### Key Results (Averaged across 3 scenarios)

| Model | Top-1 Acc (%) | Top-5 Acc (%) | Power Loss (dB) | GPS Noise Drop (%) |
|-------|:---:|:---:|:---:|:---:|
| NN | **37.28** | **85.76** | **0.85** | 11.23 |
| XGBoost | 5.42 | 39.44 | 1.80 | 1.97 |
| NB | 6.64 | 30.25 | 2.39 | 0.00 |
| KNN | 5.87 | 25.93 | 1.77 | 0.00 |
| RF | 4.73 | 34.64 | 1.72 | 0.00 |

---

## Dataset

This project uses the **DeepSense 6G** position-aided beam prediction dataset:
- **Source**: [https://deepsense6g.net](https://deepsense6g.net)
- **Frequency**: 60 GHz
- **Codebook**: 64 beams
- **Scenarios**: 3 V2I outdoor scenarios (Day-Location A, Night, Day-Location B)

### Data Setup

1. Download the position-aided subset from [DeepSense 6G](https://deepsense6g.net)
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

---

## Project Structure

```
├── Loader.py              # Main pipeline: data loading, training, evaluation, visualization
├── train_test_func.py     # Neural network architecture, training/testing, GPS noise, normalization
├── check_env_file.py      # Environment checker (Python, PyTorch, CUDA)
├── requirements.txt       # Python dependencies
├── Gathered_data_DEV/     # Dataset (.npy files — download separately)
└── saved_folder/          # Output folder (created automatically)
    └── Final_ML_Viz_*/
        ├── Final_Project_Results_Full.csv
        ├── 1_Top1_Accuracy.png
        ├── 2_Top5_Accuracy.png
        ├── ...
        └── 13_Radar_Chart.png
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/harsha71018/GPS-aided-Beam-Prediction.git
cd GPS-aided-Beam-Prediction

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Check your environment
```bash
python check_env_file.py
```

### 2. Run the full pipeline
```bash
python Loader.py
```

This will:
- Train all 5 models on 3 scenarios
- Generate 13 visualization outputs
- Save results to `saved_folder/Final_ML_Viz_<timestamp>/`
- Export `Final_Project_Results_Full.csv` with all metrics

### Outputs Generated (13 total)
1. Top-1 Accuracy Bar Chart
2. Top-5 Accuracy Bar Chart
3. Allocation Gain Bar Chart
4. Jain's Fairness Index Table
5. Power Loss Comparison Table
6. Correlation Heatmap
7–10. Accuracy vs Allocation Gain (4 scheduling strategies)
11. Latency (Log Scale)
12. Robustness Drop Bar Chart
13. Radar Chart (overall model comparison)

---

## Reproducibility

All experiments use `seed=42` for full reproducibility. The `set_global_seeds()` function in `Loader.py` locks:
- Python's `random` module
- NumPy's random state
- PyTorch's manual seed (CPU + CUDA)
- cuDNN deterministic mode

---

## Evaluation Dimensions

| Dimension | Metric |
|-----------|--------|
| Accuracy | Top-1 and Top-5 beam prediction accuracy |
| Link Quality | Average power loss (dB) — threshold: 3 dB |
| Fairness | Jain's Fairness Index across 4 scheduling strategies |
| Robustness | Accuracy drop under 1 m isotropic GPS noise |
| Efficiency | Training time (s) and inference time (μs) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{harshavardhan2025gps,
  title={GPS-Based Beam Prediction for 6G Vehicular Networks Using the DeepSense 6G Real-World Dataset},
  author={Dasyapu, Harshavardhan and Danaboyina, Vamshi and Gaikwad, Prasenjith Kumar and Tangelapalli, Swapna},
  journal={Transactions on Emerging Telecommunications Technologies},
  year={2025},
  publisher={Wiley}
}
```

---

## Acknowledgments

- **Morais et al.** — original [Position-Beam-Prediction](https://github.com/jmoraispk/Position-Beam-Prediction) codebase that this project builds upon
- [DeepSense 6G Team](https://deepsense6g.net) — Wireless Intelligence Lab, Arizona State University, for the real-world dataset

---

## License

This project is released for academic and research purposes.
