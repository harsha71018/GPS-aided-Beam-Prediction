# -*- coding: utf-8 -*-
"""
UNIT TESTS: REVISION HELPER FUNCTIONS & EVALUATION LOGIC
========================================================
Lightweight, pure-function unit tests with zero dataset dependency.
Tests feature extraction, circular block bootstrapping, data splitting,
McNemar exact binomial testing, and PyTorch model loading.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import scipy.stats as stats
import torch

# Ensure repo root and Revision_Evaluation module are accessible
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVISION_DIR = os.path.join(REPO_ROOT, "Revision_Evaluation")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if REVISION_DIR not in sys.path:
    sys.path.insert(0, REVISION_DIR)

import revision_train_test_func as func
from run_revision_evaluation import load_torch_checkpoint


class TestRevisionHelpers(unittest.TestCase):
    """Test suite for pure revision evaluation helper functions."""

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_extract_kinematic_features_geometry(self):
        """Verify UTM projection, range, and continuous azimuth encoding (sin/cos)."""
        # Synthetic base station and vehicle GPS points near Tempe, AZ (DeepSense 6G location)
        pos_bs = np.array([
            [33.4255, -111.9400],
            [33.4255, -111.9400],
            [33.4255, -111.9400]
        ])
        pos_veh = np.array([
            [33.4260, -111.9390],
            [33.4270, -111.9380],
            [33.4250, -111.9410]
        ])

        features, zone = func.extract_kinematic_features(pos_bs, pos_veh)

        # 1. Output shape must be (N, 7)
        self.assertEqual(features.shape, (3, 7))

        # 2. UTM Zone tuple check
        self.assertIsInstance(zone, tuple)
        self.assertEqual(len(zone), 2)

        # 3. Trigonometric Pythagorean identity: sin(AoA)^2 + cos(AoA)^2 == 1.0
        sin_aoa = features[:, 5]
        cos_aoa = features[:, 6]
        trig_sum = sin_aoa**2 + cos_aoa**2
        np.testing.assert_allclose(trig_sum, 1.0, atol=1e-6)

        # 4. Euclidean range check: d = sqrt(dx^2 + dy^2) > 0
        dx = features[:, 2]
        dy = features[:, 3]
        distance = features[:, 4]
        expected_dist = np.sqrt(dx**2 + dy**2)
        np.testing.assert_allclose(distance, expected_dist, atol=1e-5)
        self.assertTrue(np.all(distance > 0))

    def test_circular_mbb_indices(self):
        """Verify circular moving block bootstrap indices wrap modulo N correctly."""
        n_samples = 595  # Exact test sample count for Scenario 2
        for block_size in [25, 50, 100]:
            indices = func.circular_mbb_indices(n_samples, block_size)

            # Resample size must equal original sample size
            self.assertEqual(len(indices), n_samples)

            # All indices must be within [0, n_samples - 1]
            self.assertTrue(np.all(indices >= 0))
            self.assertTrue(np.all(indices < n_samples))

    def test_split_and_scale_data_chronological_no_leakage(self):
        """Verify chronological splitting preserves order and standardizes without data leakage."""
        n_samples = 100
        n_features = 7
        features = np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features)
        labels = np.random.randint(0, 64, size=n_samples)

        x_tr, x_te, y_tr, y_te, idx_tr, idx_te, scaler = func.split_and_scale_data(
            features, labels, split_mode="chronological", test_size=0.2, random_state=42
        )

        # 80% train, 20% test
        self.assertEqual(len(x_tr), 80)
        self.assertEqual(len(x_te), 20)
        self.assertEqual(len(y_tr), 80)
        self.assertEqual(len(y_te), 20)

        # Chronological order check
        np.testing.assert_array_equal(idx_tr, np.arange(80))
        np.testing.assert_array_equal(idx_te, np.arange(80, 100))

        # Train data must be standardized (mean ~ 0, std ~ 1)
        np.testing.assert_allclose(np.mean(x_tr, axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.std(x_tr, axis=0), 1.0, atol=1e-6)

        # Test data mean is NOT zero (verifying scaler was fit strictly on train)
        self.assertFalse(np.allclose(np.mean(x_te, axis=0), 0.0, atol=1e-2))

    def test_add_pos_noise_utm(self):
        """Verify isotropic GPS noise injection in UTM space."""
        pos_veh = np.array([
            [33.4260, -111.9390],
            [33.4270, -111.9380]
        ])

        # Zero noise test -> exact match
        noisy_zero = func.add_pos_noise_utm(pos_veh, noise_std_m=0.0)
        np.testing.assert_allclose(noisy_zero, pos_veh, atol=1e-7)

        # Non-zero noise test -> altered coordinates within reasonable threshold
        noisy_pert = func.add_pos_noise_utm(pos_veh, noise_std_m=3.0)
        self.assertEqual(noisy_pert.shape, (2, 2))
        self.assertTrue(np.all(np.abs(noisy_pert - pos_veh) > 0.0))


    def test_advanced_nn_architecture_forward(self):
        """Verify Advanced_NN_FCN forward pass produces expected logit dimensions."""
        model = func.Advanced_NN_FCN(
            num_features=7,
            num_output=64,
            nodes_per_layer=64,
            n_layers=4,
            dropout_rate=0.1
        )
        model.eval()

        dummy_x = torch.randn(16, 7)
        with torch.no_grad():
            outputs = model(dummy_x)

        self.assertEqual(outputs.shape, (16, 64))

    def test_mcnemar_exact_binomial_logic(self):
        """Verify paired McNemar exact binomial test calculation on matched discordant frames."""
        # Scenario 2 at 3 dB: b = 53 (XGB outage only), c = 3 (NN outage only)
        b_disc = 53
        c_disc = 3
        res = stats.binomtest(b_disc, b_disc + c_disc, 0.5)

        # Must reject null with extreme significance p < 1e-10 (matching 8.1371e-13)
        self.assertLess(res.pvalue, 1e-10)
        self.assertAlmostEqual(res.pvalue, 8.1371e-13, delta=1e-14)

        # Symmetric case: b = 10, c = 10 -> p = 1.0 (no difference)
        res_sym = stats.binomtest(10, 20, 0.5)
        self.assertEqual(res_sym.pvalue, 1.0)

    def test_safe_torch_checkpoint_loader(self):
        """Verify load_torch_checkpoint loads weights correctly across PyTorch versions."""
        model = func.Advanced_NN_FCN(num_features=7, num_output=64, nodes_per_layer=32, n_layers=3)
        state_dict = model.state_dict()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            torch.save(state_dict, tmp_path)
            loaded_sd = load_torch_checkpoint(tmp_path, map_location="cpu")

            self.assertEqual(set(loaded_sd.keys()), set(state_dict.keys()))
            for k in state_dict:
                self.assertTrue(torch.equal(loaded_sd[k], state_dict[k]))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
