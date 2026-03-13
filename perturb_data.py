# dataset_split.py
"""
Uniform & joint sample generator for all users and multiple app pairs.
- Evenly spread sample start indices across the full sequence (uniform intervals).
- Parse original raw files where each 3 lines describe one time step,
  and the 2nd line contains 52-dim CSV features (same rule as your script).
- Generate both train/test outputs for each (user, app) pair in one pass.

Example:
python dataset_split.py \
  --base_path /path/to/blendshape_dataset \
  --apps 360pano Immedu Sword Archery Parkour \
  --train_root pico_1 --test_root pico_2 \
  --face_file face/52.txt \
  --second 1s \
  --seq_len 20 --n_train 1000 --n_test 1000
"""

import os
import random
import argparse
import numpy as np

def read_data(file_path):
    """Read raw file: every 3 lines form a ‘time-step’; the 2nd line has 51-dim CSV features."""
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    for i in range(0, len(lines) - 1, 3):
        if len(lines[i].strip()) > 0 and len(lines[i + 1].strip()) > 0:
            feats = list(map(float, lines[i + 1].strip().split(",")))
            data.append(feats)
    return np.array(data, dtype=np.float32)

def save_uniform_samples(data, output_file, num_samples, seq_len, rng):
    """Uniformly spread sample windows across the sequence (interval bins)."""
    total_steps = len(data) - seq_len
    if total_steps <= 0:
        print(f"[WARN] Not enough steps to cut windows of length={seq_len}: {output_file}")
        return
    interval = max(1, total_steps // num_samples)

    with open(output_file, "w") as f:
        for k in range(num_samples):
            start_lo = k * interval
            start_hi = min((k + 1) * interval, total_steps)
            start_idx = rng.randint(start_lo, start_hi) if start_hi > start_lo else start_lo
            sample = data[start_idx : start_idx + seq_len]
            f.write(f"{k + 1}\n")
            for step in sample:
                f.write(",".join(map(str, step)) + "\n")
            f.write("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", required=True, type=str)
    ap.add_argument("--apps", nargs="+", required=True,
                    help="List of app folder names under each root (e.g., 360pano ... Sword...)")
    ap.add_argument("--train_root", default="", type=str)
    ap.add_argument("--test_root", default="pico_2", type=str)
    ap.add_argument("--face_file", default="face/52.txt", type=str,
                    help="relative file path under each (root/app)")
    ap.add_argument("--second", default="1s", type=str,
                    help="tag appended to output filenames")
    ap.add_argument("--seq_len", default=20, type=int)
    ap.add_argument("--n_train", default=1000, type=int)
    ap.add_argument("--n_test", default=1000, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    for user_folder in os.listdir(args.base_path):
        user_dir = os.path.join(args.base_path, user_folder)
        if not os.path.isdir(user_dir):
            continue

        for app in args.apps:
            in_train = os.path.join(user_dir, args.train_root, app, args.face_file)
            in_test  = os.path.join(user_dir, args.test_root,  app, args.face_file)

            if not (os.path.exists(in_train) and os.path.exists(in_test)):
                print(f"[SKIP] Missing train/test for user={user_folder}, app={app}")
                continue

            # Read raw data
            train_data = read_data(in_train)
            test_data  = read_data(in_test)

            # Output files (joint naming)
            out_train = os.path.join(user_dir, f"{args.train_root}_{app}_{args.second}_{args.n_train}.txt")
            out_test  = os.path.join(user_dir, f"{args.test_root}_{app}_{args.second}_{args.n_test}.txt")

            save_uniform_samples(train_data, out_train, args.n_train, args.seq_len, rng)
            save_uniform_samples(test_data,  out_test,  args.n_test,  args.seq_len, rng)

            print(f"[OK] user={user_folder} app={app} -> {os.path.basename(out_train)}, {os.path.basename(out_test)}")

if __name__ == "__main__":
    main()


