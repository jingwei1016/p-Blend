#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
knn_blendshape.py

K-Nearest Neighbors (KNN) classification on blendshape time-series data 
using time-domain statistical features (mean, min, max, median, std).

Features:
- Supports the new data naming convention (e.g., session_one_Sword_5s_2000.txt).
- Automatically handles p-Blend perturbed data variants.
- Extracts and caches time-domain features to accelerate repeated experiments.
- Utilizes GridSearchCV for optimal hyperparameter tuning.
"""

import os
import sys
import argparse
import pickle
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# -----------------------------
# Logging Utilities
# -----------------------------
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)

# -----------------------------
# Dataset Configuration
# -----------------------------
class BlendshapeDataset(Dataset):
    """
    Parses segment-based blendshape text files.
    Separator lines (digits only) divide the file into discrete samples.
    """
    def __init__(self, file_paths: List[str], path_to_label: Dict[str, int]) -> None:
        self.samples: List[np.ndarray] = []
        self.labels: List[int] = []
        self.path_to_label = path_to_label

        for fp in file_paths:
            cur_samples = self._load_samples(fp)
            self.samples.extend(cur_samples)
            self.labels.extend([self.path_to_label[fp]] * len(cur_samples))

    @staticmethod
    def _load_samples(file_path: str) -> List[np.ndarray]:
        samples: List[np.ndarray] = []
        current: List[List[float]] = []

        if not os.path.exists(file_path):
            log_warn(f"File not found: {file_path}")
            return samples

        with open(file_path, "r") as f:
            lines = f.readlines()

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            
            if line.isdigit():  # Sample boundary
                if current:
                    samples.append(np.array(current, dtype=np.float32))
                    current = []
            else:
                try:
                    feats = list(map(float, line.split(",")))
                    current.append(feats)
                except Exception as e:
                    log_warn(f"Parse error in {file_path}: '{line[:50]}' ({e})")

        if current:
            samples.append(np.array(current, dtype=np.float32))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]  # Shape: (T, D)
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), label

# -----------------------------
# Feature Extraction & Caching
# -----------------------------
def extract_features(dset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes [mean, min, max, median, std] over the time dimension for each sample.
    Resulting feature dimension: 5 * D (e.g., 5 * 52 = 260).
    """
    X, y = [], []
    for i in range(len(dset)):
        seq, label = dset[i]  
        
        # Compute statistics handling potential NaNs
        mean = torch.nan_to_num(torch.mean(seq, dim=0), nan=0.0)
        min_v = torch.nan_to_num(torch.min(seq, dim=0).values, nan=0.0)
        max_v = torch.nan_to_num(torch.max(seq, dim=0).values, nan=0.0)
        med  = torch.nan_to_num(torch.median(seq, dim=0).values, nan=0.0)
        std  = torch.nan_to_num(torch.std(seq, dim=0, unbiased=False), nan=0.0)
        
        feat = torch.cat([mean, min_v, max_v, med, std])
        X.append(feat.numpy())
        y.append(label)
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def save_features(file_name: str, X: np.ndarray, y: np.ndarray) -> None:
    with open(file_name, "wb") as f:
        pickle.dump((X, y), f)
    log_info(f"Cached features to {file_name}")

def load_features(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_name, "rb") as f:
        X, y = pickle.load(f)
    log_info(f"Loaded cached features from {file_name}")
    return X, y

# -----------------------------
# Data Discovery
# -----------------------------
def build_filename(session: str, app: str, second: str, n_samples: str, perturb: str, noise_type: str, sigma: str) -> str:
    """Constructs the exact filename based on experimental parameters."""
    base = f"session_{session}_{app}_{second}_{n_samples}.txt"
    if perturb != "clean":
        base = base.replace(".txt", f"_{perturb}_{noise_type}_{sigma}.txt")
    return base

def collect_data_paths(args: argparse.Namespace) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Scans the dataset directory for valid train/test combinations."""
    train_paths, test_paths, labels = [], [], {}

    if not os.path.isdir(args.base_path):
        raise RuntimeError(f"Invalid base path: {args.base_path}")

    for user_folder in os.listdir(args.base_path):
        user_folder_path = os.path.join(args.base_path, user_folder)
        if not (os.path.isdir(user_folder_path) and user_folder[0].isdigit()):
            continue

        try:
            label = int(user_folder.split("_")[0]) - 1
        except ValueError:
            continue

        # Collect training files
        for t_sess in args.train_sessions:
            for app in args.apps_train:
                fn = build_filename(t_sess, app, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma)
                fp = os.path.join(user_folder_path, fn)
                if os.path.exists(fp):
                    train_paths.append(fp)
                    labels[fp] = label

        # Collect testing files
        for te_sess in args.test_sessions:
            for app in args.apps_test:
                fn = build_filename(te_sess, app, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma)
                fp = os.path.join(user_folder_path, fn)
                if os.path.exists(fp):
                    test_paths.append(fp)
                    labels[fp] = label

    if not train_paths:
        log_error("No training data found. Verify apps, sessions, or perturbation args.")
    if not test_paths:
        log_error("No testing data found. Verify apps, sessions, or perturbation args.")

    log_info(f"Discovered {len(train_paths)} train files, {len(test_paths)} test files.")
    return train_paths, test_paths, labels

# -----------------------------
# Classification & Hyperparameter Tuning
# -----------------------------
def run_knn_grid_search(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, n_jobs: int = -1) -> GridSearchCV:
    """Performs grid search to find the optimal KNN parameters."""
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2], 
    }
    
    gs = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring="accuracy",
        verbose=1
    )
    
    log_info("Initiating GridSearchCV for KNN...")
    gs.fit(X_train, y_train)
    log_info(f"Optimal Parameters: {gs.best_params_}")
    log_info(f"Best Cross-Validation Accuracy: {gs.best_score_ * 100:.2f}%")
    return gs

# -----------------------------
# CLI & Execution
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KNN Re-identification on Blendshape Statistical Features")

    # Dataset arguments
    p.add_argument("--base_path", type=str, default="p_blend_dataset/", help="Dataset root directory")
    p.add_argument("--train_sessions", nargs="+", default=["one"], help="Training sessions (e.g., one two)")
    p.add_argument("--test_sessions", nargs="+", default=["two"], help="Testing sessions")
    p.add_argument("--apps_train", nargs="+", default=["Archery"], help="Training applications")
    p.add_argument("--apps_test", nargs="+", default=["Archery"], help="Testing applications")
    p.add_argument("--second", type=str, default="5s", help="Sequence length tag")
    p.add_argument("--n_samples", type=str, default="2000", help="Sample count suffix")

    # Perturbation arguments (aligns with p-Blend data generation)
    p.add_argument("--perturb_method", type=str, default="clean", choices=["clean", "p-blend", "semi-random-only", "pure-random"], help="Data variant to evaluate")
    p.add_argument("--noise_type", type=str, default="laplace")
    p.add_argument("--sigma", type=str, default="0.05")

    # Experiment settings
    p.add_argument("--feature_dir", type=str, default=".", help="Directory to store extracted feature caches")
    p.add_argument("--cv", type=int, default=5, help="GridSearchCV cross-validation folds")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic data loading")

    return p.parse_args()

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # 1. Collect data paths
    train_paths, test_paths, labels = collect_data_paths(args)
    if not train_paths or not test_paths:
        sys.exit(1)

    # 2. Build distinct cache names incorporating all parameters
    tag_train = f"{'_'.join(args.train_sessions)}_{'_'.join(args.apps_train)}_{args.second}_{args.n_samples}"
    tag_test  = f"{'_'.join(args.test_sessions)}_{'_'.join(args.apps_test)}_{args.second}_{args.n_samples}"
    noise_tag = f"_{args.perturb_method}_{args.noise_type}_{args.sigma}" if args.perturb_method != "clean" else ""
    
    feat_train_file = os.path.join(args.feature_dir, f"knn_feat_train_{tag_train}{noise_tag}.pkl")
    feat_test_file  = os.path.join(args.feature_dir, f"knn_feat_test_{tag_test}{noise_tag}.pkl")

    # 3. Load or Extract Features
    if os.path.exists(feat_train_file) and os.path.exists(feat_test_file):
        X_train, y_train = load_features(feat_train_file)
        X_test, y_test = load_features(feat_test_file)
    else:
        log_info("Extracting features from raw sequences...")
        X_train, y_train = extract_features(BlendshapeDataset(train_paths, labels))
        X_test, y_test = extract_features(BlendshapeDataset(test_paths, labels))
        save_features(feat_train_file, X_train, y_train)
        save_features(feat_test_file, X_test, y_test)

    # 4. Train & Evaluate KNN
    grid = run_knn_grid_search(X_train, y_train, cv=args.cv, n_jobs=args.n_jobs)
    best_knn = grid.best_estimator_
    
    y_pred = best_knn.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # 5. Output Summary
    print("\n" + "="*40)
    print(" EXPERIMENT SUMMARY")
    print("="*40)
    print(f" Perturbation : {args.perturb_method.upper()}")
    if args.perturb_method != "clean":
        print(f" Noise Config : {args.noise_type}, sigma={args.sigma}")
    print(f" Train Setup  : Sessions={args.train_sessions}, Apps={args.apps_train}")
    print(f" Test Setup   : Sessions={args.test_sessions}, Apps={args.apps_test}")
    print(f" Data Shape   : Train {X_train.shape} | Test {X_test.shape}")
    print(f" Best Params  : {grid.best_params_}")
    print(f" CV Accuracy  : {grid.best_score_ * 100:.2f}%")
    print(f" TEST ACCURACY: {test_acc * 100:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()


# python KNN-Stat.py --apps_train Sword --apps_test Sword --train_sessions one --test_sessions two --perturb_method clean
# python KNN-Stat.py --apps_train Sword --apps_test Sword --train_sessions one --test_sessions two --perturb_method p-blend --sigma 0.05
# python KNN-Stat.py --apps_train Archery --apps_test Sword --train_sessions one --test_sessions two