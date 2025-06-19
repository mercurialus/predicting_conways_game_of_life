import os
import numpy as np
from tqdm.auto import tqdm  # NEW

PATH = "/mnt/c/Users/Harshil/Documents/Codes/conways_game_of_life/out/build/x64-debug/dataset"
PATH_TEST = "/mnt/c/Users/Harshil/Documents/Codes/conways_game_of_life/out/build/x64-debug/dataset/test_samples"
PATH_TEST_T32 = "/mnt/c/Users/Harshil/Documents/Codes/conways_game_of_life/out/build/x64-debug/dataset/test_32_frame"


def load_bin_pair(path, shape=(32, 32)):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[:1024].reshape(shape), data[1024:].reshape(shape)


def load_all_pairs(folder):
    X, Y = [], []
    # tqdm shows overall file-loading progress
    for fname in tqdm(
        sorted(os.listdir(folder)), desc="Loading bin pairs", unit="file"
    ):
        if not fname.endswith(".bin"):
            continue
        x, y = load_bin_pair(os.path.join(folder, fname))
        X.append(x)
        Y.append(y)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y  # shape: (N, 32, 32)
