#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Traditional SVM-Stat Baseline for p-Blend

This script evaluates the robustness of facial blendshape data against 
re-identification attacks using a Traditional Support Vector Machine (SVM) 
from scikit-learn, trained on time-domain statistical features.

Key Features:
- Replaces deep learning/gradient descent with a true analytical SVM (sklearn.svm.LinearSVC).
- Automatically applies StandardScaler (crucial for traditional SVM convergence).
- Extracts 5 statistical features per sequence (mean, min, max, median, std).
- Supports the segment-based dataset structure (e.g., session_one_Sword_5s_2000.txt).
- Seamlessly integrates with p-Blend perturbed datasets.
"""

import os
import sys
import pickle
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import joblib

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)

# ---------------------------------------------------------
# Dataset Parsing
# ---------------------------------------------------------
class BlendshapeDataset(Dataset):
    """
    Parses continuous blendshape data files where discrete samples 
    are separated by a single line containing only a digit.
    """
    def __init__(self, file_paths: List[str], path_to_label: Dict[str, int]):
        self.samples: List[np.ndarray] = []
        self.labels: List[int] = []
        self.label_map = path_to_label

        for file_path in file_paths:
            file_samples = self._load_samples(file_path)
            self.samples.extend(file_samples)
            self.labels.extend([self.label_map[file_path]] * len(file_samples))

    @staticmethod
    def _load_samples(file_path: str) -> List[np.ndarray]:
        samples: List[np.ndarray] = []
        current_sample: List[List[float]] = []

        if not os.path.exists(file_path):
            log_warn(f"File not found: {file_path}")
            return samples

        with open(file_path, "r") as f:
            lines = f.readlines()

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            
            if line.isdigit():  # Sequence boundary
                if current_sample:
                    samples.append(np.array(current_sample, dtype=np.float32))
                    current_sample = []
            else:
                try:
                    features = list(map(float, line.split(",")))
                    current_sample.append(features)
                except Exception as e:
                    log_warn(f"Parse error in {file_path}: '{line[:50]}' ({e})")

        if current_sample:
            samples.append(np.array(current_sample, dtype=np.float32))
            
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]  # Shape: (TimeSteps, Features)
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), label

# ---------------------------------------------------------
# Feature Extraction & Caching
# ---------------------------------------------------------
def extract_features(dataset: Dataset, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapses variable-length sequences into a fixed-size statistical vector.
    Calculates: [mean, min, max, median, std] over the time dimension.
    We still use PyTorch internally here because its tensor operations are very fast.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    X_list, y_list = [], []

    for i in range(len(dataset)):
        seq, label = dataset[i]  
        seq = seq.to(dev) 

        # Compute robust statistics handling potential NaN values
        mean   = torch.nan_to_num(seq.mean(dim=0), nan=0.0)
        min_v  = torch.nan_to_num(seq.min(dim=0).values, nan=0.0)
        max_v  = torch.nan_to_num(seq.max(dim=0).values, nan=0.0)
        median = torch.nan_to_num(seq.median(dim=0).values, nan=0.0)
        std    = torch.nan_to_num(seq.std(dim=0, unbiased=False), nan=0.0)

        feat = torch.cat([mean, min_v, max_v, median, std])
        X_list.append(feat.cpu().numpy())
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

def save_features(file_name: str, X: np.ndarray, y: np.ndarray) -> None:
    with open(file_name, "wb") as f:
        pickle.dump((X, y), f)

def load_features(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_name, "rb") as f:
        X, y = pickle.load(f)
    return X, y

# ---------------------------------------------------------
# Data Discovery & Path Management
# ---------------------------------------------------------
def build_filename(session: str, app: str, second: str, n_samples: str, perturb: str, noise_type: str, sigma: str) -> str:
    """Constructs the exact target filename based on experiment configuration."""
    base = f"session_{session}_{app}_{second}_{n_samples}.txt"
    if perturb != "clean":
        base = base.replace(".txt", f"_{perturb}_{noise_type}_{sigma}.txt")
    return base

def collect_data_paths(args: argparse.Namespace) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Scans the dataset directory to locate matching training and testing files."""
    train_paths, test_paths, labels = [], [], {}

    if not os.path.isdir(args.base_path):
        raise RuntimeError(f"Invalid dataset base path: {args.base_path}")

    for user_folder in os.listdir(args.base_path):
        user_folder_path = os.path.join(args.base_path, user_folder)
        if not (os.path.isdir(user_folder_path) and user_folder[0].isdigit()):
            continue

        try:
            # Extract label (e.g., folder '01_User' -> label 0)
            label = int(user_folder.split("_")[0]) - 1
        except ValueError:
            continue

        # Gather training files
        for t_sess in args.train_sessions:
            for app in args.apps_train:
                fn = build_filename(t_sess, app, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma)
                fp = os.path.join(user_folder_path, fn)
                if os.path.exists(fp):
                    train_paths.append(fp)
                    labels[fp] = label

        # Gather testing files
        for te_sess in args.test_sessions:
            for app in args.apps_test:
                fn = build_filename(te_sess, app, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma)
                fp = os.path.join(user_folder_path, fn)
                if os.path.exists(fp):
                    test_paths.append(fp)
                    labels[fp] = label

    if not train_paths: log_error("No training data found. Verify parameters.")
    if not test_paths: log_error("No testing data found. Verify parameters.")

    log_info(f"Discovered Train files: {len(train_paths)} | Test files: {len(test_paths)}")
    return train_paths, test_paths, labels

# ---------------------------------------------------------
# CLI & Execution Entry Point
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traditional SVM-Stat Baseline for VR Blendshape Re-identification")

    # Dataset parameters
    p.add_argument("--base_path", type=str, default="p_blend_dataset/", help="Dataset root directory")
    p.add_argument("--train_sessions", nargs="+", default=["one"], help="Training sessions (e.g., one two)")
    p.add_argument("--test_sessions", nargs="+", default=["two"], help="Testing sessions")
    p.add_argument("--apps_train", nargs="+", default=["Sword"], help="Training VR applications")
    p.add_argument("--apps_test", nargs="+", default=["Sword"], help="Testing VR applications")
    p.add_argument("--second", type=str, default="10s", help="Sequence length tag")
    p.add_argument("--n_samples", type=str, default="2000", help="Sample count suffix")

    # Perturbation (Defense) parameters
    p.add_argument("--perturb_method", type=str, default="clean", choices=["clean", "p-blend", "semi-random-only", "pure-random"], help="Perturbation defense variant")
    p.add_argument("--noise_type", type=str, default="laplace", help="Noise distribution")
    p.add_argument("--sigma", type=str, default="0.05", help="Noise scale/intensity")

    # Traditional SVM hyperparameters
    p.add_argument("--feature_dir", type=str, default=".", help="Directory for feature cache and models")
    p.add_argument("--C", type=float, default=1.0, help="Regularization parameter for SVM (Penalty cost)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()

def main() -> None:
    args = parse_args()
    
    # Enforce reproducibility
    np.random.seed(args.seed)

    # 1. Discover files
    train_paths, test_paths, labels = collect_data_paths(args)
    if not train_paths or not test_paths:
        sys.exit(1)

    # 2. Define cache & model names (preventing overrides across experiments)
    tag_train = f"{'_'.join(args.train_sessions)}_{'_'.join(args.apps_train)}_{args.second}_{args.n_samples}"
    tag_test  = f"{'_'.join(args.test_sessions)}_{'_'.join(args.apps_test)}_{args.second}_{args.n_samples}"
    noise_tag = f"_{args.perturb_method}_{args.noise_type}_{args.sigma}" if args.perturb_method != "clean" else ""
    
    feat_train_file = os.path.join(args.feature_dir, f"svm_feat_train_{tag_train}{noise_tag}.pkl")
    feat_test_file  = os.path.join(args.feature_dir, f"svm_feat_test_{tag_test}{noise_tag}.pkl")
    model_save_path = os.path.join(args.feature_dir, f"best_svm_{tag_train}_test_{tag_test}{noise_tag}.joblib")

    # Use CUDA just for fast feature extraction if available, SVM training will be on CPU
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Cache or Extract features
    if os.path.exists(feat_train_file) and os.path.exists(feat_test_file):
        X_train, y_train = load_features(feat_train_file)
        X_test, y_test = load_features(feat_test_file)
    else:
        log_info("Parsing sequences and extracting statistical features...")
        X_train, y_train = extract_features(BlendshapeDataset(train_paths, labels), device=device_str)
        X_test, y_test = extract_features(BlendshapeDataset(test_paths, labels), device=device_str)
        save_features(feat_train_file, X_train, y_train)
        save_features(feat_test_file, X_test, y_test)

    # 4. Standardize Features (Crucial for traditional SVM)
    # SVM is distance-based; unscaled features will ruin the hyperplane optimization
    log_info("Applying Standard Scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Traditional SVM
    # We use LinearSVC because it is highly optimized (liblinear) and extremely fast for datasets > 10,000 samples.
    log_info(f"Training Traditional Linear SVM (C={args.C}). This might take a moment...")
    svm_model = LinearSVC(C=args.C, random_state=args.seed, max_iter=5000, dual="auto")
    svm_model.fit(X_train_scaled, y_train)

    # 6. Evaluate
    log_info("Evaluating model on test set...")
    y_pred = svm_model.predict(X_test_scaled)
    final_accuracy = accuracy_score(y_test, y_pred) * 100.0

    # Save the pipeline (Scaler + Model)
    joblib.dump({'scaler': scaler, 'model': svm_model}, model_save_path)
    log_info(f"Saved traditional SVM model pipeline to {model_save_path}")

    # 7. Experiment Summary
    print("\n" + "="*40)
    print(" EXPERIMENT SUMMARY (Traditional SVM-Stat)")
    print("="*40)
    print(f" Perturbation : {args.perturb_method.upper()}")
    if args.perturb_method != "clean":
        print(f" Noise Config : {args.noise_type}, sigma={args.sigma}")
    print(f" Train Setup  : Sessions={args.train_sessions}, Apps={args.apps_train}")
    print(f" Test Setup   : Sessions={args.test_sessions}, Apps={args.apps_test}")
    print(f" Data Shape   : Train {X_train.shape} | Test {X_test.shape}")
    print(f" FINAL ACC    : {final_accuracy:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()