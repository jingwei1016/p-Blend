#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep Learning Runner for p-Blend

This script orchestrates the training and evaluation of various deep learning models 
(Transformer, CNN, LSTM, MLP) on VR facial blendshape time-series data.

Key Features:
- Supports multiple sequence models directly on raw VR tracking data.
- Implements strict chronological Train/Validation splitting to prevent time-series data leakage.
- Fully supports p-Blend perturbed datasets for evaluating defense mechanisms.
- Dynamically generates model checkpoint names to prevent overwrites across different experiments.
"""

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import BlendshapeDataset
from model import TransformerModel, CNNModel, LSTMModel, FCModel
from train import train_model, evaluate_model

# ---------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")

# ---------------------------------------------------------
# Data Discovery & Path Management
# ---------------------------------------------------------
def build_filename(session: str, app: str, second: str, n_samples: str, perturb: str, noise_type: str, sigma: str) -> str:
    """Constructs the target filename based on experiment configuration and perturbation."""
    base = f"session_{session}_{app}_{second}_{n_samples}.txt"
    if perturb != "clean":
        base = base.replace(".txt", f"_{perturb}_{noise_type}_{sigma}.txt")
    return base

def collect_data_paths(args: argparse.Namespace) -> tuple:
    """
    Scans the base directory for valid training and testing files.
    Returns: (train_paths, test_paths, labels_dict)
    """
    train_paths, test_paths, labels = [], [], {}
    
    train_name = build_filename(args.train_session, args.app, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma)
    test_name  = build_filename(args.test_session,  args.app, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma)

    for user_folder in os.listdir(args.base_path):
        user_dir = os.path.join(args.base_path, user_folder)
        if not (os.path.isdir(user_dir) and user_folder and user_folder[0].isdigit()):
            continue

        train_file = os.path.join(user_dir, train_name)
        test_file  = os.path.join(user_dir, test_name)

        if os.path.exists(train_file) and os.path.exists(test_file):
            try:
                # Extract label (e.g., folder '01_User' -> label 0)
                lab = int(user_folder.split("_")[0]) - 1
            except ValueError:
                continue
                
            labels[train_file] = lab
            labels[test_file]  = lab
            train_paths.append(train_file)
            test_paths.append(test_file)
        else:
            log_warn(f"Missing required files for user {user_folder}. Checked:\n  - {train_name}\n  - {test_name}")
            
    log_info(f"Discovered Train files: {len(train_paths)} | Test files: {len(test_paths)}")
    return train_paths, test_paths, labels

# ---------------------------------------------------------
# Model Factory
# ---------------------------------------------------------
def get_model(name: str, num_features: int, time_steps: int, num_classes: int) -> nn.Module:
    """Instantiates the selected deep learning architecture."""
    name = name.lower()
    if name == "transformer":
        return TransformerModel(num_features=num_features, time_steps=time_steps, num_classes=num_classes)
    elif name == "cnn":
        return CNNModel(num_features=num_features, time_steps=time_steps, num_classes=num_classes)
    elif name == "lstm":
        return LSTMModel(input_size=num_features, hidden_size=128, num_layers=2, num_classes=num_classes)
    elif name == "mlp":
        return FCModel(num_features=num_features, time_steps=time_steps, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture specified: {name}. Choose from [transformer, cnn, lstm, mlp].")

# ---------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Learning Runner for Blendshape Re-identification")
    
    # Dataset Parameters
    parser.add_argument("--base_path", type=str, required=True, help="Root directory containing user folders")
    parser.add_argument("--app", type=str, default="Sword", help="Target VR application")
    parser.add_argument("--second", type=str, default="5s", help="Sequence length suffix")
    parser.add_argument("--n_samples", type=str, default="2000", help="Sample count suffix")
    parser.add_argument("--train_session", type=str, default="one", help="Training session identifier")
    parser.add_argument("--test_session", type=str, default="two", help="Testing session identifier")
                    
    # Defense/Perturbation Parameters
    parser.add_argument("--perturb_method", type=str, default="clean", choices=["clean", "p-blend", "semi-random-only", "pure-random"])
    parser.add_argument("--noise_type", type=str, default="laplace")
    parser.add_argument("--sigma", type=str, default="0.05")

    # Model & Training Hyperparameters
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "cnn", "lstm", "mlp"])
    parser.add_argument("--time_steps", type=int, default=200, help="Number of frames per sequence")
    parser.add_argument("--num_features", type=int, default=52, help="Dimension of the raw blendshape vector")
    parser.add_argument("--num_classes", type=int, default=45, help="Number of target users")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of training data to use for validation")
    
    return parser.parse_args()

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main():
    args = parse_args()
    
    # 1. Print Experiment Configuration
    print("="*60)
    print(" DEEP LEARNING EXPERIMENT SETUP")
    print("="*60)
    print(f" Model        : {args.model.upper()}")
    print(f" Application  : {args.app} ({args.second}, {args.n_samples} samples)")
    print(f" Sessions     : Train={args.train_session} | Test={args.test_session}")
    print(f" Perturbation : {args.perturb_method.upper()} " + (f"(sigma={args.sigma})" if args.perturb_method != "clean" else ""))
    print("="*60)

    # 2. Discover Data Paths
    train_paths, test_paths, labels = collect_data_paths(args)
    if not train_paths:
        log_error("No training data found. Terminating.")
        return

    # 3. Instantiate Full Datasets
    log_info("Loading raw sequences into memory...")
    full_train_ds = BlendshapeDataset(train_paths, labels)
    test_ds       = BlendshapeDataset(test_paths, labels)

    # 4. Chronological Train-Validation Split (Prevents Time-Series Leakage)
    # Extracts labels to perform stratified and chronological splitting per user.
    log_info(f"Applying strict chronological Train/Val split (Val Ratio: {args.val_ratio})...")
    all_train_labels = np.array([int(full_train_ds[i][1]) for i in range(len(full_train_ds))])
    
    train_idx, val_idx = [], []
    unique_labels = np.unique(all_train_labels)
    
    for label in unique_labels:
        idx_for_label = np.where(all_train_labels == label)[0]
        # Chronological split: First (1 - val_ratio) for training, remaining for validation
        split_point = int(len(idx_for_label) * (1.0 - args.val_ratio))
        
        train_idx.extend(idx_for_label[:split_point])
        val_idx.extend(idx_for_label[split_point:])

    train_ds = Subset(full_train_ds, train_idx)
    val_ds   = Subset(full_train_ds, val_idx)

    log_info(f"Dataset Size -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 5. Initialize DataLoaders
    # Note: Shuffling is perfectly safe NOW because the Train and Val subsets 
    # are completely isolated in the time domain.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # 6. Initialize Model and Optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, args.num_features, args.time_steps, args.num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 7. Generate Dynamic Model Checkpoint Name
    noise_tag = f"_{args.perturb_method}_{args.noise_type}_{args.sigma}" if args.perturb_method != "clean" else ""
    model_save_path = f"best_{args.model}_{args.app}_{args.second}{noise_tag}.pth"

    # 8. Execute Training Loop
    log_info(f"Starting {args.model.upper()} training. Checkpoints will be saved to: {model_save_path}")
    train_model(
        model=model,
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, 
        criterion=criterion, 
        optimizer=optimizer,
        num_epochs=args.epochs, 
        model_save_path=model_save_path
    )

    # 9. Final Evaluation on Test Set
    print("\n" + "="*60)
    log_info("Restoring best validation checkpoint for FINAL TEST evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    try:
        evaluate_model(model, test_loader, prefix="FINAL TEST")
    except TypeError:
        # Fallback if train.py's evaluate_model doesn't support the 'prefix' arg
        evaluate_model(model, test_loader)
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

# =====================================================================
# Usage Examples
# =====================================================================
# 
# 1. Baseline Transformer (Clean Data)
#    python main.py --base_path p_blend_dataset/ \
#      --app Sword --second 10s --model transformer --time_steps 200
#
# 2. Evaluate CNN against p-Blend Defense
#    python main.py --base_path p_blend_dataset/ \
#      --app Sword --second 10s --model cnn --time_steps 200 \
#      --perturb_method p-blend --sigma 0.05
#
# 3. Cross-Session Evaluation with LSTM
#    python main.py --base_path p_blend_dataset/ \
#      --app Sword --second 10s --model lstm --time_steps 200 \
#      --train_session two --test_session one