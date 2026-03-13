#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correlation Analysis Pipeline for p-Blend (Production Version)

Key Features:
- Aggregates all blendshape frames from specified sessions/apps.
- Computes Pearson Correlation and P-value matrices.
- Extracts linear mappings (Y = aX + b) for high-correlation pairs.
- Generates high-resolution heatmaps with publication-ready formatting.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ---------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------
def read_raw_data(file_path):
    """Parses raw blendshape logs (3-line groups). Extracts data from the 2nd line."""
    features = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # raw format: i=timestamp, i+1=data, i+2=empty
        for i in range(0, len(lines) - 1, 3):
            line_content = lines[i+1].strip()
            if line_content:
                try:
                    parts = line_content.split(',')
                    # Validate 52-dim format
                    if len(parts) == 52:
                        vector = [float(x) for x in parts]
                        features.append(vector)
                except ValueError:
                    continue
                    
    if not features:
        return np.empty((0, 52))
        
    return np.array(features, dtype=np.float32)

def collect_aggregated_data(args):
    """Traverses dataset directory to aggregate frames from users/sessions/apps."""
    all_frames = []
    for user_folder in os.listdir(args.base_path):
        user_dir = os.path.join(args.base_path, user_folder)
        if not (os.path.isdir(user_dir) and user_folder[0].isdigit()):
            continue
            
        for session_name in args.sessions:
            session_dir = os.path.join(user_dir, f"session_{session_name}")
            for app in args.apps:
                file_path = os.path.join(session_dir, app, "blendshape_data.txt")
                if os.path.exists(file_path):
                    frames = read_raw_data(file_path)
                    if frames.shape[0] > 0:
                        all_frames.append(frames)
                    
    if not all_frames:
        return None
    return np.vstack(all_frames)
# ---------------------------------------------------------
# Mathematical Analysis
# ---------------------------------------------------------
def run_correlation_analysis(features):
    """Calculates Correlation and P-value matrices."""
    num_dims = features.shape[1]
    col_names = [str(i) for i in range(num_dims)]
    df = pd.DataFrame(features, columns=col_names)

    corr_matrix = pd.DataFrame(np.zeros((num_dims, num_dims)), index=df.columns, columns=df.columns)
    p_matrix = pd.DataFrame(np.zeros((num_dims, num_dims)), index=df.columns, columns=df.columns)

    print(f"[INFO] Calculating correlations for {num_dims} dimensions...")
    for i in range(num_dims):
        for j in range(i + 1, num_dims):
            corr, p_val = pearsonr(df.iloc[:, i], df.iloc[:, j])
            corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i] = corr
            p_matrix.iloc[i, j] = p_matrix.iloc[j, i] = p_val
            
    return corr_matrix, p_matrix

def save_analysis_results(corr_matrix, p_matrix, features, output_dir, threshold=0.8):
    """Saves matrices to TXT and extracts linear mappings."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Matrices
    corr_matrix.to_csv(os.path.join(output_dir, 'overall_correlation_matrix.txt'), sep='\t')
    p_matrix.to_csv(os.path.join(output_dir, 'overall_p_value_matrix.txt'), sep='\t')
    
    # Extract Linear Mappings (Y = aX + b)
    mapping_path = os.path.join(output_dir, 'high_correlation_linear_mappings.txt')
    num_dims = features.shape[1]
    df = pd.DataFrame(features, columns=[str(i) for i in range(num_dims)])
    
    with open(mapping_path, 'w') as f:
        f.write(f"Linear Mappings for Pairs with |R| > {threshold}\n\n")
        for i in range(num_dims):
            for j in range(i + 1, num_dims):
                r_val = corr_matrix.iloc[i, j]
                p_val = p_matrix.iloc[i, j]
                if abs(r_val) > threshold and p_val < 0.05:
                    x, y = df.iloc[:, i].values, df.iloc[:, j].values
                    a, b = np.polyfit(x, y, 1) # Linear regression
                    f.write(f"Index {i} <-> {j} | R={r_val:.4f} | Eq: Idx_{j} = {a:.4f}*Idx_{i} + {b:.4f}\n")
    print(f"[INFO] Analysis reports saved to {output_dir}")

# ---------------------------------------------------------
# Publication-Ready Plotting
# ---------------------------------------------------------
def generate_publication_heatmap(matrix, output_path, cmap='coolwarm', step=4):
    """Generates the high-res heatmap with specified font and tick settings."""
    plt.figure(figsize=(12, 12))
    
    ax = sns.heatmap(
        matrix, 
        annot=False, 
        cmap=cmap, 
        cbar_kws={
            'shrink': 1.0,
            'aspect': 30,
            'fraction': 0.02,
            'pad': 0.015
        }
    )

    num_features = matrix.shape[0]
    indices = np.arange(num_features)

    # Label Formatting
    ax.set_xticks(indices[::step] + 0.5)
    ax.set_yticks(indices[::step] + 0.5)
    ax.set_xticklabels(indices[::step], fontsize=26, fontweight='bold', rotation=0)
    ax.set_yticklabels(indices[::step], fontsize=26, fontweight='bold', rotation=0)

    ax.set_xlabel("Blendshape Index", fontweight='bold', fontsize=32)
    ax.set_ylabel("Blendshape Index", fontweight='bold', fontsize=32)

    # Colorbar Tick Formatting
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=26)
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Heatmap saved -> {output_path}")

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full Correlation Analysis Pipeline")
    parser.add_argument("--base_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--sessions", nargs="+", default=["one", "two"])
    parser.add_argument("--apps", nargs="+", default=["Sword", "Immedu", "Archery", "360pano", "Parkour"])
    parser.add_argument("--second", type=str, default="10s")
    parser.add_argument("--n_samples", type=str, default="2000")
    parser.add_argument("--out_dir", type=str, default="correlation_results")
    parser.add_argument("--thresh", type=float, default=0.8, help="Correlation threshold for mappings")
    
    args = parser.parse_args()

    # 1. Aggregate Data
    print("[INFO] Aggregating raw frames...")
    full_features = collect_aggregated_data(args)
    if full_features is None:
        print("[ERROR] No data found. Check paths.")
        return
    print(f"[INFO] Analyzed {full_features.shape[0]} frames across all users.")

    # 2. Run Statistics
    corr_matrix, p_matrix = run_correlation_analysis(full_features)

    # 3. Save Reports and Mappings
    save_analysis_results(corr_matrix, p_matrix, full_features, args.out_dir, threshold=args.thresh)

    # 4. Generate Heatmaps
    generate_publication_heatmap(corr_matrix, os.path.join(args.out_dir, "heatmap_correlation.png"))
    generate_publication_heatmap(p_matrix, os.path.join(args.out_dir, "heatmap_pvalue.png"), cmap='Reds')

if __name__ == "__main__":
    main()