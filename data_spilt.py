#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_split.py

Uniform & joint sample generator for VR facial blendshape datasets.
This script parses raw continuous time-series data and extracts fixed-length 
sequences (windows) evenly spread across the entire recording.

Key Updates:
- Adopts the new directory structure: session_{session_name}/{app}/blendshape_data.txt
- Outputs the new segment naming convention: session_one_Parkour_10s_2000.txt

Example Usage:
    python dataset_split.py \
      --base_path p_blend_dataset \
      --apps 360pano Immedu Sword Archery Parkour \
      --train_session one --test_session two \
      --raw_file blendshape_data.txt \
      --second 5s \
      --seq_len 100 --n_train 2000 --n_test 2000
"""

import os
import random
import argparse
import numpy as np

# ---------------------------------------------------------
# Data I/O
# ---------------------------------------------------------
def read_raw_blendshapes(file_path: str) -> np.ndarray:
    """
    Reads the raw data file.
    The original recording format assumes every 3 lines describe one time-step,
    where the 2nd line contains the CSV-formatted feature vector.
    
    Args:
        file_path: Path to the raw blendshape_data.txt file.
        
    Returns:
        A NumPy array of shape (Total_Frames, Features).
    """
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    for i in range(0, len(lines) - 1, 3):
        # Ensure the expected lines are not empty
        if len(lines[i].strip()) > 0 and len(lines[i + 1].strip()) > 0:
            feats = list(map(float, lines[i + 1].strip().split(",")))
            data.append(feats)
            
    return np.array(data, dtype=np.float32)

def save_uniform_samples(data: np.ndarray, output_file: str, num_samples: int, seq_len: int, rng: random.Random) -> None:
    """
    Uniformly spreads sample windows across the raw sequence to ensure diverse coverage.
    
    Args:
        data: The full contiguous raw sequence array.
        output_file: Path to save the extracted segments.
        num_samples: How many discrete sequences to extract.
        seq_len: The number of frames (time_steps) per extracted sequence.
        rng: Random number generator for deterministic jitter within bins.
    """
    total_steps = len(data) - seq_len
    if total_steps <= 0:
        print(f"[WARN] Not enough frames to cut windows of length={seq_len} in {output_file}")
        return
        
    # Calculate the size of the interval bin to ensure uniform spread
    interval = max(1, total_steps // num_samples)

    with open(output_file, "w") as f:
        for k in range(num_samples):
            # Define the temporal boundaries for this specific sample
            start_lo = k * interval
            start_hi = min((k + 1) * interval, total_steps)
            
            # Add a slight random jitter within the assigned interval
            start_idx = rng.randint(start_lo, start_hi) if start_hi > start_lo else start_lo
            sample = data[start_idx : start_idx + seq_len]
            
            # Write out in the standard 3-line format (ID, CSV Data, Empty Line)
            f.write(f"{k + 1}\n")
            for step in sample:
                f.write(",".join(map(str, step)) + "\n")
            f.write("\n")

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Segment raw blendshape recordings into fixed-length samples.")
    
    # Path configuration
    ap.add_argument("--base_path", required=True, type=str, help="Root directory containing user folders")
    ap.add_argument("--apps", nargs="+", required=True, help="List of applications (e.g., 360pano Immedu Sword Archery Parkour)")
    
    # Naming configuration (Matches the new 'session_one' paradigm)
    ap.add_argument("--train_session", default="one", type=str, help="Session identifier for training data (e.g., one)")
    ap.add_argument("--test_session", default="two", type=str, help="Session identifier for testing data (e.g., two)")
    ap.add_argument("--raw_file", default="blendshape_data.txt", type=str, help="Name of the raw data file inside the app folder")
    ap.add_argument("--second", default="5s", type=str, help="Time suffix appended to the output file (e.g., 5s, 10s)")
    
    # Sampling parameters
    ap.add_argument("--seq_len", default=100, type=int, help="Number of frames per sample (time_steps)")
    ap.add_argument("--n_train", default=2000, type=int, help="Number of training samples to extract")
    ap.add_argument("--n_test", default=2000, type=int, help="Number of testing samples to extract")
    ap.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("="*60)
    print(f"[INFO] Initializing Dataset Splitter")
    print(f"[INFO] Train Session : {args.train_session} ({args.n_train} samples)")
    print(f"[INFO] Test Session  : {args.test_session} ({args.n_test} samples)")
    print(f"[INFO] Sequence Len  : {args.seq_len} frames")
    print("="*60)

    for user_folder in os.listdir(args.base_path):
        user_dir = os.path.join(args.base_path, user_folder)
        if not (os.path.isdir(user_dir) and user_folder[0].isdigit()):
            continue

        for app in args.apps:
            # Construct paths to the raw data files
            # Expected pattern: base_path/1/session_one/Sword/blendshape_data.txt
            in_train = os.path.join(user_dir, f"session_{args.train_session}", app, args.raw_file)
            in_test  = os.path.join(user_dir, f"session_{args.test_session}",  app, args.raw_file)

            if not (os.path.exists(in_train) and os.path.exists(in_test)):
                print(f"  [SKIP] User {user_folder} | App: {app} -> Missing raw files.")
                continue

            # Read raw sequences into memory
            train_data = read_raw_blendshapes(in_train)
            test_data  = read_raw_blendshapes(in_test)

            # Construct target output filenames
            # Expected pattern: base_path/1/session_one_Sword_5s_2000.txt
            out_train = os.path.join(user_dir, f"session_{args.train_session}_{app}_{args.second}_{args.n_train}.txt")
            out_test  = os.path.join(user_dir, f"session_{args.test_session}_{app}_{args.second}_{args.n_test}.txt")

            # Extract and save samples
            save_uniform_samples(train_data, out_train, args.n_train, args.seq_len, rng)
            save_uniform_samples(test_data,  out_test,  args.n_test,  args.seq_len, rng)

            print(f"  [OK] Processed User {user_folder} | App: {app}")

    print("\n[DONE] Dataset splitting complete.")

if __name__ == "__main__":
    main()