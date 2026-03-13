#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP-Stat Baseline for p-Blend

This script evaluates the effectiveness of re-identification attacks using 
a Multi-Layer Perceptron (MLP) trained on statistical features extracted 
from VR facial blendshape time-series data. 

Key Features:
- Replaces raw time-series with 5 statistical features (mean, min, max, median, std).
- Uses a strict chronological train-test split (first 60% train, last 40% validation) 
  per user to prevent data leakage in time-series evaluation.
- Seamlessly integrates with p-Blend perturbed datasets.
- Automatically handles caching and prevents file overwrites across different noise parameters.
"""

import os
import sys
import json
import pickle
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
# Configuration
# ---------------------------------------------------------
@dataclass
class Config:
    base_path: str = "p_blend_dataset/"

    # Default identifiers for data files
    second: str = "10s"
    app_train: str = " "
    app_test: str = " "
    train_session: str = "one"
    test_session: str = "two"
    n_samples: str = "2000"

    # Perturbation parameters (clean means original unperturbed data)
    perturb_method: str = "clean"  # Options: clean, p-blend, pure-random, semi-random-only
    noise_type: str = "laplace"
    sigma: str = "0.05"

    # Hyperparameters
    learning_rate: float = 5e-4
    num_epochs: int = 50
    batch_size: int = 32
    device: str = "cuda"  
    num_classes: int = 45 

    # File naming templates
    train_tpl: str = "session_{train}_{app}_{second}_{n_samples}.txt"
    test_tpl: str = "session_{test}_{app}_{second}_{n_samples}.txt"

    # Automation Application Sets
    second_list_same: Tuple[str, ...] = ("5s", "10s")
    app_list_same: Tuple[str, ...] = ("Immedu", "Parkour")

    second_list_diff: Tuple[str, ...] = ("5s", "10s")
    app_list_diff: Tuple[str, ...] = ("360pano", "Immedu", "Archery", "Sword", "Parkour")

    # Output artifact filenames
    summary_same_txt: str = "results_summary_same.txt"
    summary_same_all_tsv: str = "results_summary_all_same.txt"
    best_same_txt: str = "best_overall_same.txt"

    summary_diff_txt: str = "results_summary_diff.txt"
    summary_diff_all_tsv: str = "results_summary_all_diff.txt"
    best_diff_txt: str = "best_overall_diff.txt"

    model_tpl_same: str = "best_mlp_model_{train}_{app}_{second}.pth"
    model_tpl_diff: str = "best_mlp_model_{train}_{app_train}_{app_test}_{second}.pth"


# ---------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------
class BlendshapeDataset(Dataset):
    """
    Parses segment-based blendshape text files.
    Separator lines (digits only) divide the file into discrete samples.
    """
    def __init__(self, file_paths: List[str], path_to_label: Dict[str, int]) -> None:
        self.samples: List[np.ndarray] = []
        self.labels: List[int] = []
        self.path_to_label = path_to_label

        for p in file_paths:
            cur_samples = self._load_samples(p)
            self.samples.extend(cur_samples)
            self.labels.extend([self.path_to_label[p]] * len(cur_samples))

    @staticmethod
    def _is_numeric_separator(line: str) -> bool:
        s = line.strip()
        return s.isdigit()

    def _load_samples(self, file_path: str) -> List[np.ndarray]:
        samples: List[np.ndarray] = []
        current: List[List[float]] = []

        if not os.path.exists(file_path):
            log_warn(f"Missing file: {file_path}")
            return samples

        with open(file_path, "r") as f:
            lines = f.readlines()

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            if self._is_numeric_separator(line):
                if current:
                    samples.append(np.array(current, dtype=np.float32))
                    current = []
            else:
                try:
                    feats = list(map(float, line.split(",")))
                    current.append(feats)
                except Exception as e:
                    log_warn(f"Failed to parse line in {file_path}: '{line[:80]}' ({e})")

        if current:
            samples.append(np.array(current, dtype=np.float32))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        arr = self.samples[idx]
        label = self.labels[idx]
        return torch.tensor(arr, dtype=torch.float32), label


# ---------------------------------------------------------
# File Discovery & Path Management
# ---------------------------------------------------------
def make_train_test_filelists(cfg: Config, app_train: str, app_test: str, second: str) -> Tuple[str, str]:
    """Generates expected filenames for training and testing data."""
    train_name = cfg.train_tpl.format(train=cfg.train_session, app=app_train, second=second, n_samples=cfg.n_samples)
    test_name = cfg.test_tpl.format(test=cfg.test_session, app=app_test, second=second, n_samples=cfg.n_samples)
    
    # Append perturbation suffix if dealing with noise-injected data
    if cfg.perturb_method != "clean":
        noise_suffix = f"_{cfg.perturb_method}_{cfg.noise_type}_{cfg.sigma}.txt"
        train_name = train_name.replace(".txt", noise_suffix)
        test_name = test_name.replace(".txt", noise_suffix)
        
    return train_name, test_name

def collect_data_paths(cfg: Config, train_name: str, test_name: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Scans the base directory to locate available data files for all users."""
    train_paths: List[str] = []
    test_paths: List[str] = []
    labels: Dict[str, int] = {}

    if not os.path.isdir(cfg.base_path):
        log_error(f"Base path is not a directory: {cfg.base_path}")
        return train_paths, test_paths, labels

    for user_folder in os.listdir(cfg.base_path):
        user_path = os.path.join(cfg.base_path, user_folder)
        if not (os.path.isdir(user_path) and user_folder and user_folder[0].isdigit()):
            continue

        train_file = os.path.join(user_path, train_name)
        test_file = os.path.join(user_path, test_name)

        if os.path.exists(train_file) and os.path.exists(test_file):
            try:
                label = int(user_folder.split("_")[0]) - 1
            except Exception:
                log_warn(f"Folder name not in expected format for label: {user_folder}")
                continue

            labels[train_file] = label
            labels[test_file] = label
            train_paths.append(train_file)
            test_paths.append(test_file)

    return train_paths, test_paths, labels


# ---------------------------------------------------------
# Feature Extraction & Caching
# ---------------------------------------------------------
STATS_ORDER = ("mean", "min", "max", "median", "std")

def _compute_stat(seq: torch.Tensor, name: str) -> torch.Tensor:
    """Computes a specific time-domain statistic handling potential NaNs."""
    if name == "mean":
        return torch.nan_to_num(seq.mean(dim=0), nan=0.0)
    if name == "min":
        return torch.nan_to_num(seq.min(dim=0).values, nan=0.0)
    if name == "max":
        return torch.nan_to_num(seq.max(dim=0).values, nan=0.0)
    if name == "median":
        return torch.nan_to_num(seq.median(dim=0).values, nan=0.0)
    if name == "std":
        return torch.nan_to_num(seq.std(dim=0, unbiased=False), nan=0.0)
    raise ValueError(f"Unknown stat: {name}")

def extract_features(dset: Dataset, stats: Tuple[str, ...], device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """Compresses sequences into static statistical vectors."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for i in range(len(dset)):
        seq, label = dset[i]
        seq = seq.clone().detach().to(dev)

        stat_vecs = [_compute_stat(seq, s) for s in stats]
        feat = torch.cat(stat_vecs, dim=0)
        X_list.append(feat.cpu().numpy())
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def feature_cache_names(train_name: str, test_name: str, stats: Tuple[str, ...]) -> Tuple[str, str]:
    """Constructs cache filenames ensuring isolation across perturbation parameters."""
    base_train = train_name.replace(".txt", "_features_n.pkl")
    base_test  = test_name.replace(".txt", "_features_n.pkl")

    if tuple(stats) == STATS_ORDER:
        return base_train, base_test

    suffix = "_stats_" + "_".join(stats) + ".pkl"
    train_cache = train_name.replace(".txt", suffix)
    test_cache  = test_name.replace(".txt", suffix)
    return train_cache, test_cache

def save_features(file_name: str, X: np.ndarray, y: np.ndarray) -> None:
    with open(file_name, "wb") as f:
        pickle.dump((X, y), f)

def load_features(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_name, "rb") as f:
        X, y = pickle.load(f)
    return X, y


# ---------------------------------------------------------
# Neural Network Architecture
# ---------------------------------------------------------
class MLP(nn.Module):
    """
    Multi-Layer Perceptron for sequence classification based on extracted statistics.
    Implements a deep architecture with Batch Normalization and Dropout.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )
            layers.append(block)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------
# Training and Evaluation Logic
# ---------------------------------------------------------
def evaluate_model(model: nn.Module, loader: DataLoader) -> float:
    """Computes accuracy over a dataset."""
    device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * (correct / max(1, total))

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    model_save_path: str
) -> float:
    """Standard training loop with validation-based checkpointing."""
    device = next(model.parameters()).device
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        val_acc = evaluate_model(model, val_loader)
        
        log_msg = f"[Epoch {epoch+1:03d}/{num_epochs}] Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%"

        if test_loader is not None:
            test_acc = evaluate_model(model, test_loader)
            log_msg += f" | Test Acc: {test_acc:.2f}%"

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            log_msg += "  ✅ New Best Model!"

        log_info(log_msg)

    log_info(f"Training completed. Best Validation Accuracy: {best_acc:.2f}%")

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    final_acc = evaluate_model(model, test_loader)
    log_info(f"🔥 Final Test Accuracy (using best val model): {final_acc:.2f}%")
    return final_acc


# ---------------------------------------------------------
# Core Experiment Pipeline
# ---------------------------------------------------------
def run_single_experiment(
    cfg: Config,
    app_train: str,
    app_test: str,
    second: str,
    model_path: str,
    stats: Tuple[str, ...]
) -> float:
    """Executes a full pipeline for a specific app combination."""
    train_name, test_name = make_train_test_filelists(cfg, app_train, app_test, second)
    train_paths, test_paths, labels = collect_data_paths(cfg, train_name, test_name)
    
    if not train_paths or not test_paths:
        log_warn(f"No valid user folders found for combination: {train_name} -> {test_name}; accuracy = 0.0")
        return 0.0

    cache_train, cache_test = feature_cache_names(train_name, test_name, stats)
    if os.path.exists(cache_train) and os.path.exists(cache_test):
        X_train_full, y_train_full = load_features(cache_train)
        X_test, y_test = load_features(cache_test)
    else:
        dset_train = BlendshapeDataset(train_paths, labels)
        dset_test = BlendshapeDataset(test_paths, labels)

        X_train_full, y_train_full = extract_features(dset_train, stats=stats, device=cfg.device)
        X_test, y_test = extract_features(dset_test, stats=stats, device=cfg.device)

        save_features(cache_train, X_train_full, y_train_full)
        save_features(cache_test, X_test, y_test)

    # Time-Series Validation Split: Chronologically splits 60% train / 40% val per user
    # Prevents adjacent frame leakage inherent to random shuffling
    X_train_list, X_val_list = [], []
    y_train_list, y_val_list = [], []
    
    unique_labels = np.unique(y_train_full)
    
    for label in unique_labels:
        idx = np.where(y_train_full == label)[0]
        total_samples_for_user = len(idx)
        split_point = int(total_samples_for_user * 0.6)  
        
        train_idx = idx[:split_point]
        val_idx = idx[split_point:]
        
        X_train_list.append(X_train_full[train_idx])
        y_train_list.append(y_train_full[train_idx])
        X_val_list.append(X_train_full[val_idx])
        y_val_list.append(y_train_full[val_idx])
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    # Tensor Conversion
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.long)
    Xval_t = torch.tensor(X_val, dtype=torch.float32)
    yval_t = torch.tensor(y_val, dtype=torch.long)
    Xte_t = torch.tensor(X_test, dtype=torch.float32)
    yte_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    val_ds   = torch.utils.data.TensorDataset(Xval_t, yval_t)
    test_ds  = torch.utils.data.TensorDataset(Xte_t, yte_t)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Model Initialization
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train_full))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.num_epochs,
        model_save_path=model_path
    )
    return acc


# ---------------------------------------------------------
# Evaluation Modes (Intra-App vs. Cross-App)
# ---------------------------------------------------------
def main_automation_same(cfg: Config, stats: Tuple[str, ...]) -> None:
    """Evaluates scenario where Train and Test data originate from the SAME VR application."""
    results: List[Tuple[str, str, str, str, float]] = [] 

    with open(cfg.summary_same_txt, "w") as f:
        f.write("train\ttest\tsecond\tapp\taccuracy\n")

    for second in cfg.second_list_same:
        for app in cfg.app_list_same:
            noise_tag = f"_{cfg.perturb_method}_{cfg.noise_type}_{cfg.sigma}" if cfg.perturb_method != "clean" else ""
            model_path = cfg.model_tpl_same.format(train=cfg.train_session, app=app, second=second).replace(".pth", f"{noise_tag}.pth")

            log_info(f"Running (SAME APP) | Train Session={cfg.train_session} | Test Session={cfg.test_session} | Second={second} | App={app} | Noise={cfg.perturb_method}")
            acc = run_single_experiment(cfg, app_train=app, app_test=app, second=second, model_path=model_path, stats=stats)

            results.append((cfg.train_session, cfg.test_session, second, app, acc))
            with open(cfg.summary_same_txt, "a") as f:
                f.write(f"{cfg.train_session}\t{cfg.test_session}\t{second}\t{app}\t{acc}\n")

    df = pd.DataFrame(results, columns=["train", "test", "second", "app", "accuracy"])
    df.to_csv(cfg.summary_same_all_tsv, sep="\t", index=False)
    
    if not df.empty:
        best_row = df.loc[df["accuracy"].idxmax()]
        with open(cfg.best_same_txt, "w") as f:
            f.write("Best overall combination (Intra-app):\n")
            f.write(best_row.to_string())
        log_info("Completed SAME-app automation. Best Configuration:")
        log_info(f"\n{best_row.to_string()}")

def main_automation_diff(cfg: Config, stats: Tuple[str, ...]) -> None:
    """Evaluates scenario where Train and Test data originate from DIFFERENT VR applications."""
    results: List[Tuple[str, str, str, str, str, float]] = []

    with open(cfg.summary_diff_txt, "w") as f:
        f.write("train\ttest\tsecond\tapp_train\tapp_test\taccuracy\n")

    for second in cfg.second_list_diff:
        for app_tr in cfg.app_list_diff:
            for app_te in cfg.app_list_diff:
                if app_tr == app_te:
                    continue

                noise_tag = f"_{cfg.perturb_method}_{cfg.noise_type}_{cfg.sigma}" if cfg.perturb_method != "clean" else ""
                model_path = cfg.model_tpl_diff.format(train=cfg.train_session, app_train=app_tr, app_test=app_te, second=second).replace(".pth", f"{noise_tag}.pth")

                log_info(f"Running (CROSS APP) | Train={app_tr} | Test={app_te} | Second={second} | Noise={cfg.perturb_method}")
                acc = run_single_experiment(cfg, app_train=app_tr, app_test=app_te, second=second, model_path=model_path, stats=stats)

                results.append((cfg.train_session, cfg.test_session, second, app_tr, app_te, acc))
                with open(cfg.summary_diff_txt, "a") as f:
                    f.write(f"{cfg.train_session}\t{cfg.test_session}\t{second}\t{app_tr}\t{app_te}\t{acc}\n")

    df = pd.DataFrame(results, columns=["train", "test", "second", "app_train", "app_test", "accuracy"])
    df.to_csv(cfg.summary_diff_all_tsv, sep="\t", index=False)
    
    if not df.empty:
        best_row = df.loc[df["accuracy"].idxmax()]
        with open(cfg.best_diff_txt, "w") as f:
            f.write("Best overall combination (Cross-app):\n")
            f.write(best_row.to_string())
        log_info("Completed CROSS-app automation. Best Configuration:")
        log_info(f"\n{best_row.to_string()}")


# ---------------------------------------------------------
# Command Line Interface (CLI)
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP-based Re-identification on Blendshape Statistical Features")
    
    # Core settings
    p.add_argument("--mode", type=str, default="diff", choices=["same", "diff"], help="Intra-app (same) or Cross-app (diff) attack evaluation")
    p.add_argument("--base_path", type=str, default=None, help="Root directory containing user data folders")
    p.add_argument("--device", type=str, default=None, help="Compute device (e.g., cuda, cpu)")
    p.add_argument("--n_samples", type=str, default=None, help="Sample count suffix for targeted files (e.g., 2000)")
    
    # Model parameters
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--num_classes", type=int, default=None, help="Number of target users (classes)")

    # Perturbation parameters
    p.add_argument("--perturb_method", type=str, default="clean", 
                   choices=["clean", "p-blend", "semi-random-only", "pure-random"], 
                   help="Specify the perturbation variant applied to the data")
    p.add_argument("--noise_type", type=str, default="laplace")
    p.add_argument("--sigma", type=str, default="0.05")

    # Feature selection
    p.add_argument("--stats", type=str, nargs="+", default=list(STATS_ORDER), choices=list(STATS_ORDER), 
                   help="Select one or more statistical metrics to extract")

    return p.parse_args()


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
def main() -> None:
    cfg = Config()
    args = parse_args()

    # Apply overrides
    if args.base_path is not None: cfg.base_path = args.base_path
    if args.device is not None: cfg.device = args.device
    if args.epochs is not None: cfg.num_epochs = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr is not None: cfg.learning_rate = args.lr
    if args.num_classes is not None: cfg.num_classes = args.num_classes
    if args.n_samples is not None: cfg.n_samples = args.n_samples

    cfg.perturb_method = args.perturb_method
    cfg.noise_type = args.noise_type
    cfg.sigma = args.sigma

    # Isolate summary artifact filenames if using perturbed data
    if cfg.perturb_method != "clean":
        noise_tag = f"_{cfg.perturb_method}_{cfg.noise_type}_{cfg.sigma}"
        cfg.summary_same_txt = cfg.summary_same_txt.replace(".txt", f"{noise_tag}.txt")
        cfg.summary_same_all_tsv = cfg.summary_same_all_tsv.replace(".txt", f"{noise_tag}.txt")
        cfg.best_same_txt = cfg.best_same_txt.replace(".txt", f"{noise_tag}.txt")
        cfg.summary_diff_txt = cfg.summary_diff_txt.replace(".txt", f"{noise_tag}.txt")
        cfg.summary_diff_all_tsv = cfg.summary_diff_all_tsv.replace(".txt", f"{noise_tag}.txt")

    stats = tuple(args.stats)
    for s in stats:
        if s not in STATS_ORDER:
            raise ValueError(f"Unsupported statistic metric: {s}")

    log_info(f"Loaded Configuration:\n{json.dumps(asdict(cfg), indent=2)}")
    log_info(f"Targeting Statistical Features: {stats}")

    if args.mode == "same":
        main_automation_same(cfg, stats)
    else:
        main_automation_diff(cfg, stats)

if __name__ == "__main__":
    main()