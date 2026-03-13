#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WinVote-Stat Baseline for p-Blend

This script evaluates re-identification robustness using a Windowed-Statistics + Linear SVM 
baseline, incorporating majority voting at test time.

Key Features:
- Supports the segment-based dataset structure (e.g., session_one_Sword_5s_2000.txt).
- Automatically handles p-Blend perturbed data variants via command-line arguments.
- Extracts per-window statistical and kinematic features (mean, min, max, median, std, 
  velocity, acceleration).
- Trains a PyTorch-based Linear SVM using multi-class Hinge Loss.
- Employs a sliding window evaluation strategy, concluding with a majority vote for the final prediction.
"""

import os
import sys
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# Global Defaults
# =========================================================
DEFAULT_SECOND = "5s"
DEFAULT_N_SAMPLES = "2000"
DEFAULT_APP = "Sword"
DEFAULT_TRAIN = "one"
DEFAULT_TEST = "two"
DEFAULT_LR = 1e-4
DEFAULT_NUM_CLASSES = 45
DEFAULT_BASE_PATH = "p_blend_dataset/"

# Sliding-window parameters (Configured for 100Hz VR tracking)
DEFAULT_FPS = 100
DEFAULT_WINDOW_SEC = 2
DEFAULT_OVERLAP_SEC = 1

# =========================================================
# Logging Utilities
# =========================================================
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)

# =========================================================
# Dataset Parsing
# =========================================================
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

# =========================================================
# Windowed Feature Extraction
# =========================================================
def _window_indices(length: int, window_size: int, step_size: int):
    """Yields (start, end) indices for a sliding window."""
    if length < window_size:
        return 
    for start in range(0, length - window_size + 1, step_size):
        yield start, start + window_size

def _feature_from_window(window: torch.Tensor) -> torch.Tensor:
    """
    Computes statistical and kinematic features for a given temporal window.
    Returns a concatenated vector of shape (9 * D).
    """
    # 1st-order derivatives (Velocity)
    velocity = torch.diff(window, dim=0)                 
    
    # 2nd-order derivatives (Acceleration)
    acceleration = torch.diff(velocity, dim=0)           

    # Time-domain statistics
    mean = torch.mean(window, dim=0)
    min_val = torch.min(window, dim=0).values
    max_val = torch.max(window, dim=0).values
    median = torch.median(window, dim=0).values
    std_dev = torch.std(window, dim=0, unbiased=False)

    # Kinematic statistics
    velocity_mean = torch.mean(velocity, dim=0)
    velocity_abs_mean = torch.mean(torch.abs(velocity), dim=0)
    acceleration_mean = torch.mean(acceleration, dim=0)
    acceleration_abs_mean = torch.mean(torch.abs(acceleration), dim=0)

    feat = torch.cat([
        mean, min_val, max_val, median, std_dev,
        velocity_abs_mean, velocity_mean,
        acceleration_abs_mean, acceleration_mean
    ])
    return feat

def extract_features_with_windows(
    dataset: Dataset,
    device: str = "cuda",
    fps: int = DEFAULT_FPS,
    window_sec: int = DEFAULT_WINDOW_SEC,
    overlap_sec: int = DEFAULT_OVERLAP_SEC
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a sliding window over each sample sequence and extracts features.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    window_size = int(window_sec * fps)
    step_size = int((window_sec - overlap_sec) * fps)

    X_list, y_list = [], []
    for i in range(len(dataset)):
        seq, label = dataset[i]  
        seq = seq.to(dev)
        T = seq.shape[0]

        for s, e in _window_indices(T, window_size, step_size):
            win = seq[s:e]  
            feat = _feature_from_window(win)
            X_list.append(feat.cpu().numpy())
            y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

# =========================================================
# Model Definitions
# =========================================================
class LinearSVM(nn.Module):
    """A single linear transformation layer acting as an SVM hyperplane."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class HingeLoss(nn.Module):
    """Multi-class Hinge Loss (Crammer-Singer formulation)."""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
        correct_scores = (outputs * one_hot).sum(dim=1)
        max_others = (outputs - one_hot).max(dim=1)[0]
        loss = torch.clamp(self.margin - correct_scores + max_others, min=0.0)
        return loss.mean()

# =========================================================
# Data Discovery & Path Management
# =========================================================
def build_filename(session: str, app: str, second: str, n_samples: str, perturb: str, noise_type: str, sigma: str) -> str:
    """Constructs the exact filename based on experimental parameters."""
    base = f"session_{session}_{app}_{second}_{n_samples}.txt"
    if perturb != "clean":
        base = base.replace(".txt", f"_{perturb}_{noise_type}_{sigma}.txt")
    return base

def collect_data_paths(args: argparse.Namespace) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Scans the dataset directory to locate matching training and testing files."""
    train_paths, test_paths, labels = [], [], {}

    if not os.path.isdir(args.base_path):
        raise RuntimeError(f"Invalid dataset base path: {args.base_path}")

    train_name = build_filename(args.train_session, args.app_train, args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma)
    test_name  = build_filename(args.test_session,  args.app_test,  args.second, args.n_samples, args.perturb_method, args.noise_type, args.sigma)

    for user_folder in os.listdir(args.base_path):
        user_folder_path = os.path.join(args.base_path, user_folder)
        if not (os.path.isdir(user_folder_path) and user_folder[0].isdigit()):
            continue

        try:
            label = int(user_folder.split("_")[0]) - 1
        except ValueError:
            continue
            
        train_fp = os.path.join(user_folder_path, train_name)
        test_fp  = os.path.join(user_folder_path, test_name)
        
        if os.path.exists(train_fp) and os.path.exists(test_fp):
            labels[train_fp] = label
            labels[test_fp]  = label
            train_paths.append(train_fp)
            test_paths.append(test_fp)

    if not train_paths or not test_paths:
        log_error("No valid data found. Verify paths, sessions, and perturbation arguments.")

    log_info(f"Discovered Train files: {len(train_paths)} | Test files: {len(test_paths)}")
    return train_paths, test_paths, labels

# =========================================================
# Training & Voting Evaluation
# =========================================================
def evaluate_model_with_vote(
    model: nn.Module,
    dataset: Dataset,
    fps: int,
    window_sec: int,
    overlap_sec: int
) -> float:
    """
    Evaluates the model using a sliding window over the raw test sequences.
    The final sequence-level prediction is determined by majority voting across its windows.
    """
    model.eval()
    device = next(model.parameters()).device

    window_size = int(window_sec * fps)
    step_size = int((window_sec - overlap_sec) * fps)

    vote_dict = defaultdict(list)
    label_dict = {}

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for idx, (sample, label) in enumerate(loader):
            seq = sample.squeeze(0).to(device)  
            T = seq.shape[0]
            label_dict[idx] = label.item()

            has_vote = False
            for s, e in _window_indices(T, window_size, step_size):
                win = seq[s:e]  
                feat = _feature_from_window(win).unsqueeze(0).to(device)
                
                logits = model(feat)
                pred = torch.argmax(logits, dim=1).item()
                vote_dict[idx].append(pred)
                has_vote = True

            # Fallback if sequence is too short for a full temporal window
            if not has_vote and T >= window_size:
                mid = (T - window_size) // 2
                win = seq[mid : mid + window_size]
                feat = _feature_from_window(win).unsqueeze(0).to(device)
                pred = torch.argmax(model(feat), dim=1).item()
                vote_dict[idx].append(pred)

    correct_predictions = 0
    for idx, preds in vote_dict.items():
        if preds:
            majority_vote = Counter(preds).most_common(1)[0][0]
            correct_predictions += int(majority_vote == label_dict[idx])

    accuracy = 100.0 * correct_predictions / max(1, len(vote_dict))
    return accuracy

def train_feature_level(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int
) -> None:
    """Trains the SVM layer using the extracted windowed features."""
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / max(1, len(train_loader))
        log_info(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {avg_loss:.4f}")

# =========================================================
# CLI & Execution Entry Point
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WinVote-Stat Baseline Evaluation")

    # Dataset arguments
    p.add_argument("--base_path", type=str, default=DEFAULT_BASE_PATH, help="Dataset root directory")
    p.add_argument("--train_session", type=str, default=DEFAULT_TRAIN, help="Training session identifier")
    p.add_argument("--test_session", type=str, default=DEFAULT_TEST, help="Testing session identifier")
    p.add_argument("--app_train", type=str, default=DEFAULT_APP, help="Training application")
    p.add_argument("--app_test", type=str, default=DEFAULT_APP, help="Testing application")
    p.add_argument("--second", type=str, default=DEFAULT_SECOND, help="Sequence length tag")
    p.add_argument("--n_samples", type=str, default=DEFAULT_N_SAMPLES, help="Sample count suffix")

    # Perturbation (Defense) arguments
    p.add_argument("--perturb_method", type=str, default="clean", choices=["clean", "p-blend", "semi-random-only", "pure-random"], help="Perturbation defense variant")
    p.add_argument("--noise_type", type=str, default="laplace")
    p.add_argument("--sigma", type=str, default="0.05")

    # Training hyperparameters
    p.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="Number of classification targets")

    # Window configuration
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--window_sec", type=int, default=DEFAULT_WINDOW_SEC)
    p.add_argument("--overlap_sec", type=int, default=DEFAULT_OVERLAP_SEC)

    p.add_argument("--model_dir", type=str, default=".", help="Directory to save the trained model")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return p.parse_args()

def main() -> float:
    args = parse_args()
    
    # Enforce reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Discover files
    train_paths, test_paths, labels = collect_data_paths(args)
    if not train_paths or not test_paths:
        sys.exit(1)

    train_dataset = BlendshapeDataset(train_paths, labels)
    test_dataset  = BlendshapeDataset(test_paths, labels)

    # 2. Extract Windowed Features for Training
    log_info("Extracting sliding-window features for training...")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, y_train = extract_features_with_windows(
        train_dataset, device=device_str, fps=args.fps, window_sec=args.window_sec, overlap_sec=args.overlap_sec
    )

    # Prepare DataLoader
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)

    # 3. Initialize Model
    input_dim = X_train.shape[1]
    device = torch.device(device_str)
    
    model = LinearSVM(input_dim=input_dim, num_classes=args.num_classes).to(device)
    criterion = HingeLoss(margin=1.2)
    
    # Add weight_decay to properly enforce SVM L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Execute Training
    log_info("Initiating SVM Training (Feature-Level)...")
    train_feature_level(model, train_loader, criterion, optimizer, epochs=args.epochs)

    # 5. Evaluate via Majority Voting
    log_info("Evaluating Test Set via Sliding-Window Majority Vote...")
    final_acc = evaluate_model_with_vote(
        model, test_dataset, fps=args.fps, window_sec=args.window_sec, overlap_sec=args.overlap_sec
    )

    # 6. Save Model & Output Summary
    noise_tag = f"_{args.perturb_method}_{args.noise_type}_{args.sigma}" if args.perturb_method != "clean" else ""
    model_name = f"best_winvote_{args.train_session}_{args.app_train}_{args.second}_{args.n_samples}{noise_tag}.pth"
    model_path = os.path.join(args.model_dir, model_name)
    
    torch.save(model.state_dict(), model_path)
    log_info(f"Model saved to: {model_path}")

    print("\n" + "="*40)
    print(" EXPERIMENT SUMMARY (WinVote-Stat)")
    print("="*40)
    print(f" Perturbation : {args.perturb_method.upper()}")
    if args.perturb_method != "clean":
        print(f" Noise Config : {args.noise_type}, sigma={args.sigma}")
    print(f" Train Setup  : Session={args.train_session}, App={args.app_train}")
    print(f" Test Setup   : Session={args.test_session}, App={args.app_test}")
    print(f" FINAL VOTING ACCURACY: {final_acc:.2f}%")
    print("="*40 + "\n")

    return final_acc

if __name__ == "__main__":
    main()