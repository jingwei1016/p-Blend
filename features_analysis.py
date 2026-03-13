
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Analysis & Tree-based Baseline for p-Blend

This script evaluates blendshape re-identification using tree-based models 
(XGBoost, Random Forest, Decision Tree) and generates feature importance rankings.

New Feature:
- Now calculates Global Statistical Importance (Mean vs Min vs Max vs Median vs Std)
  by aggregating the importance of all 52 blendshape dimensions per category.
"""

import os
import sys
import pickle
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# ---------------------------------------------------------
# Logging & Reproducibility
# ---------------------------------------------------------
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)

def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------
# Dataset & Feature Extraction
# ---------------------------------------------------------
class BlendshapeDataset(Dataset):
    def __init__(self, file_paths: List[str], path_to_label: Dict[str, int]):
        self.samples: List[np.ndarray] = []
        self.labels: List[int] = []
        self.label_map = path_to_label
        for fp in file_paths:
            seqs = self._load_samples(fp)
            self.samples.extend(seqs)
            self.labels.extend([self.label_map[fp]] * len(seqs))

    @staticmethod
    def _is_sep(line: str) -> bool:
        return line.strip().isdigit()

    def _load_samples(self, file_path: str) -> List[np.ndarray]:
        samples, current = [], []
        if not os.path.exists(file_path):
            return samples
        with open(file_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line: continue
                if self._is_sep(line):
                    if current:
                        samples.append(np.array(current, dtype=np.float32))
                        current = []
                else:
                    try:
                        feats = list(map(float, line.split(",")))
                        current.append(feats)
                    except Exception: pass
        if current: samples.append(np.array(current, dtype=np.float32))
        return samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return torch.tensor(self.samples[idx], dtype=torch.float32), self.labels[idx]

def extract_features(dset: Dataset, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    X_list, y_list = [], []
    for i in range(len(dset)):
        seq, label = dset[i] 
        seq = seq.to(dev)
        # Order: Mean, Min, Max, Median, Std
        mean = torch.nan_to_num(seq.mean(dim=0), nan=0.0)
        minv = torch.nan_to_num(seq.min(dim=0).values, nan=0.0)
        maxv = torch.nan_to_num(seq.max(dim=0).values, nan=0.0)
        med  = torch.nan_to_num(seq.median(dim=0).values, nan=0.0)
        std  = torch.nan_to_num(seq.std(dim=0, unbiased=False), nan=0.0)
        feat = torch.cat([mean, minv, maxv, med, std], dim=0) 
        X_list.append(feat.cpu().numpy())
        y_list.append(label)
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.int64)

# ---------------------------------------------------------
# Importance Analysis Logic
# ---------------------------------------------------------
def analyze_and_save_importance(model, out_dir: str, exp_tag: str, plot: bool):
    """
    Analyzes importance at two levels:
    1. Individual (Mean_0, Std_51, etc.)
    2. Global Category (Total Importance of Mean vs Total Importance of Std)
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = np.asarray(model.feature_importances_)
    num_dims = 52
    base_stats = ["mean", "min", "max", "median", "std"]
    
    # 1. Level 1: Individual Feature Importance
    feat_names = []
    for s in base_stats:
        feat_names += [f"{s}_{i}" for i in range(num_dims)]
    
    order = np.argsort(importances)[::-1]
    with open(os.path.join(out_dir, f"importance_detailed_{exp_tag}.txt"), "w") as f:
        f.write("Detailed Feature Importance (sorted descending)\n")
        for idx in order:
            f.write(f"{feat_names[idx]}: {importances[idx]:.6f}\n")

    # 2. Level 2: Statistical Category Importance (Aggregated)
    cat_importance = {}
    for i, stat in enumerate(base_stats):
        start_idx = i * num_dims
        end_idx = (i + 1) * num_dims
        cat_importance[stat] = np.sum(importances[start_idx:end_idx])

    # Save Category Importance TXT
    with open(os.path.join(out_dir, f"importance_categories_{exp_tag}.txt"), "w") as f:
        f.write("Global Statistical Category Importance\n")
        for stat in base_stats:
            f.write(f"{stat.upper()}: {cat_importance[stat]:.6f}\n")

    if plot:
        # Plot Detailed (Top 30)
        topk = min(30, len(importances))
        top_order = order[:topk]
        plt.figure(figsize=(12, 6))
        plt.bar(range(topk), importances[top_order], color='skyblue', edgecolor='black')
        plt.xticks(range(topk), [feat_names[i] for i in top_order], rotation=75, ha="right")
        plt.ylabel("Importance Score")
        plt.title(f"Top-{topk} Detailed Features")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_detailed_{exp_tag}.png"), dpi=200)
        plt.close()

        # Plot Categories (The 5 Stats)
        plt.figure(figsize=(8, 6))
        stats_labels = [s.upper() for s in base_stats]
        stats_values = [cat_importance[s] for s in base_stats]
        plt.bar(stats_labels, stats_values, color='salmon', edgecolor='black')
        plt.ylabel("Aggregated Importance Score")
        plt.title("Global Statistical Category Importance")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_categories_{exp_tag}.png"), dpi=200)
        plt.close()
    
    log_info(f"Importance analysis saved to {out_dir}")

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
def build_filename(session: str, app: str, second: str, n_samples: str, perturb: str, noise_type: str, sigma: str) -> str:
    base = f"session_{session}_{app}_{second}_{n_samples}.txt"
    if perturb != "clean":
        base = base.replace(".txt", f"_{perturb}_{noise_type}_{sigma}.txt")
    return base

def collect_data_paths(args) -> Tuple[List[str], List[str], Dict[str, int]]:
    train_paths, test_paths, labels = [], [], {}
    for user_folder in os.listdir(args.base_path):
        user_dir = os.path.join(args.base_path, user_folder)
        if not (os.path.isdir(user_dir) and user_folder[0].isdigit()): continue
        try: label = int(user_folder.split("_")[0]) - 1
        except ValueError: continue
        for t_sess in args.train_sessions:
            for app in args.apps_train:
                fp = os.path.join(user_dir, build_filename(t_sess, app, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma))
                if os.path.exists(fp): train_paths.append(fp); labels[fp] = label
        for te_sess in args.test_sessions:
            for app in args.apps_test:
                fp = os.path.join(user_dir, build_filename(te_sess, app, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma))
                if os.path.exists(fp): test_paths.append(fp); labels[fp] = label
    return train_paths, test_paths, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="p_blend_dataset/")
    parser.add_argument("--train_sessions", nargs="+", default=["one"])
    parser.add_argument("--test_sessions", nargs="+", default=["two"])
    parser.add_argument("--apps_train", nargs="+", default=["Sword"])
    parser.add_argument("--apps_test", nargs="+", default=["Sword"])
    parser.add_argument("--second", type=str, default="10s")
    parser.add_argument("--n_samples", type=str, default="2000")
    parser.add_argument("--perturb_method", type=str, default="clean", choices=["clean", "p-blend", "pure-random", "semi-random-only"])
    parser.add_argument("--noise_type", type=str, default="laplace")
    parser.add_argument("--sigma", type=str, default="0.05")
    parser.add_argument("--model", type=str, default="RF", choices=["XGB", "RF", "DT"])
    parser.add_argument("--val_ratio", type=float, default=0.4)
    parser.add_argument("--out_dir", type=str, default="feature_analysis_results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    set_seed()
    ensure_dir(args.out_dir)

    train_paths, test_paths, labels = collect_data_paths(args)
    if not train_paths: 
        log_error("No training data found."); sys.exit(1)

    X_train_full, y_train_full = extract_features(BlendshapeDataset(train_paths, labels))
    X_test, y_test = extract_features(BlendshapeDataset(test_paths, labels))

    X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
    for label in np.unique(y_train_full):
        idx = np.where(y_train_full == label)[0]
        split_point = int(len(idx) * (1.0 - args.val_ratio))
        X_train_list.append(X_train_full[idx[:split_point]]); y_train_list.append(y_train_full[idx[:split_point]])
        X_val_list.append(X_train_full[idx[split_point:]]); y_val_list.append(y_train_full[idx[split_point:]])

    X_train_sub, y_train_sub = np.concatenate(X_train_list), np.concatenate(y_train_list)
    X_val_sub, y_val_sub = np.concatenate(X_val_list), np.concatenate(y_val_list)
    X_tune, y_tune = np.concatenate([X_train_sub, X_val_sub]), np.concatenate([y_train_sub, y_val_sub])
    ps = PredefinedSplit(np.concatenate([np.full(len(X_train_sub), -1), np.zeros(len(X_val_sub))]))

    if args.model == "XGB":
        if xgb is None: raise RuntimeError("XGBoost not installed.")
        model = xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
        params = {"n_estimators": [100], "max_depth": [10]}
    elif args.model == "RF":
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        params = {"n_estimators": [100], "max_depth": [20]}
    else:
        model = DecisionTreeClassifier(random_state=42)
        params = {"max_depth": [20]}

    gs = GridSearchCV(model, params, cv=ps, n_jobs=-1).fit(X_tune, y_tune)
    best_model = gs.best_estimator_

    acc = accuracy_score(y_test, best_model.predict(X_test)) * 100
    log_info(f"Final Test Accuracy: {acc:.2f}%")

    noise_tag = f"_{args.perturb_method}_{args.noise_type}_{args.sigma}" if args.perturb_method != "clean" else ""
    # Fixed the typo here: changed app_train to apps_train
    app_tag = "_".join(args.apps_train)
    exp_tag = f"{args.model}_{app_tag}_{args.second}{noise_tag}"
    analyze_and_save_importance(best_model, args.out_dir, exp_tag, args.plot)

if __name__ == "__main__":
    main()