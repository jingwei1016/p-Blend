#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
p-Blend: Privacy-preserving Blendshape Perturbation.
Implements 'p-blend', 'semi-random-only', and 'pure-random' strategies.
"""

import os
import argparse
import numpy as np

# Highly correlated pairs from paper Table 5 for noise sharing
HIGH_CORRELATION_PAIRS = [
    (0, 12), (1, 13), (2, 45), (3, 30), (3, 36),
    (4, 16), (6, 21), (6, 50), (8, 39), (11, 44),
    (19, 21), (23, 49), (27, 40), (28, 38), (29, 37),
    (30, 36), (31, 35), (33, 42), (46, 47)
]

def generate_noise(input_dim: int, noise_type: str, mu: float, sigma: float, consider_correlation: bool) -> np.ndarray:
    """Generate noise vector with optional correlation-aware sharing."""
    if noise_type == 'normal':
        noise = np.random.normal(mu, sigma, input_dim)
    elif noise_type == 'laplace':
        noise = np.random.laplace(mu, sigma, input_dim)
    else:
        raise ValueError("Use 'normal' or 'laplace'.")
    
    # Apply noise sharing for correlated features
    if consider_correlation:
        used_features = set()
        for (f1, f2) in HIGH_CORRELATION_PAIRS:
            if f1 not in used_features and f2 not in used_features:
                noise[f2] = noise[f1]  
                used_features.add(f1)
                used_features.add(f2)
                
    return noise

def process_file(in_path: str, out_path: str, args: argparse.Namespace):
    """Inject noise into blendshape file line by line."""
    is_fixed_noise = args.method in ['p-blend', 'semi-random-only']
    consider_corr = args.method == 'p-blend'

    # Generate fixed offset for the entire sequence if using semi-random/p-blend
    file_fixed_noise = None
    if is_fixed_noise:
        file_fixed_noise = generate_noise(args.input_dim, args.noise_type, args.mu, args.sigma, consider_corr)

    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
        for line in fin:
            s = line.strip()
            
            if not s:
                fout.write("\n")
                continue
                
            if s.isdigit():
                fout.write(line)
                continue
                
            try:
                feats = np.array(list(map(float, s.split(','))), dtype=np.float32)
                
                # Per-frame noise for pure-random, otherwise use fixed offset
                current_noise = file_fixed_noise if is_fixed_noise else generate_noise(
                    args.input_dim, args.noise_type, args.mu, args.sigma, consider_corr
                )
                
                noisy_feats = np.clip(feats + current_noise, 0.0, 1.0)
                fout.write(",".join(f"{v:.6f}" for v in noisy_feats) + "\n")
                
            except Exception as e:
                print(f"[WARN] Parsing error in {in_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Apply p-Blend perturbation.")
    
    # Path parameters
    parser.add_argument("--base_path", type=str, required=True, help="Root folder for users")
    parser.add_argument("--app", type=str, default="Sword", help="Application name")
    parser.add_argument("--second", type=str, default="10s", help="Time segment length")
    parser.add_argument("--n_samples", type=str, default="2000", help="Sample count suffix")
    parser.add_argument("--session", type=str, default="two", help="Session ID")
    
    # Perturbation parameters
    parser.add_argument("--method", type=str, default="p-blend", 
                        choices=["p-blend", "semi-random-only", "pure-random"])
    parser.add_argument("--noise_type", type=str, default="laplace", choices=["normal", "laplace"])
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.05, help="Noise intensity (b)")
    parser.add_argument("--input_dim", type=int, default=52)
    
    args = parser.parse_args()
    
    target_filename = f"session_{args.session}_{args.app}_{args.second}_{args.n_samples}.txt"
    out_filename = f"session_{args.session}_{args.app}_{args.second}_{args.n_samples}_{args.method}_{args.noise_type}_{args.sigma}.txt"
    
    print("-" * 30)
    print(f"Strategy: {args.method.upper()}")
    print(f"Intensity: {args.sigma}")
    print("-" * 30)

    processed_count = 0
    for user_folder in os.listdir(args.base_path):
        user_dir = os.path.join(args.base_path, user_folder)
        if not (os.path.isdir(user_dir) and user_folder[0].isdigit()):
            continue
            
        in_path = os.path.join(user_dir, target_filename)
        out_path = os.path.join(user_dir, out_filename)
        
        if os.path.exists(in_path):
            process_file(in_path, out_path, args)
            processed_count += 1
            print(f"Done: User {user_folder}")

    print(f"\n[Finished] Processed {processed_count} users.")

if __name__ == "__main__":
    main()
