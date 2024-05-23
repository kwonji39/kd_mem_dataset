import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from typing import List
import os
import sys

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

INFL = np.load("/home/kwon/cs590_sp24/d/kd_memorization/src/dataset/loss_diff_score.npz")

# Make Data Subsets based on memorization values
def filter_subset(training_mem: np.ndarray, thresh_min: float, thresh_max: float):
    subset = []
    for i in tqdm(range(len(training_mem))):
        if training_mem[i] <= thresh_max and training_mem[i] >= thresh_min:
            subset.append(i)
    return subset

def save_subsets(dir_path: str, training_mem: np.ndarray, threshs: List[float], equal_size: bool = True, min_thresh: float = 0, max_thresh: float = None):
    if max_thresh is not None:
        # select from threshs[i] to max_thresh
        size = training_mem.shape[0]
        for min_thresh in sorted(threshs, reverse=True):
            subset = filter_subset(training_mem, min_thresh, max_thresh)
            size = min(size, len(subset))
            if equal_size:
                subset = sorted(random.sample(subset, size))
            subset_df = pd.DataFrame(subset, columns=["idx"])
            subset_df.to_csv(os.path.join(dir_path, "subset_{}-{}.csv".format(min_thresh, max_thresh)), index=False)

    else:
        # select from min_thresh to threshs[i]
        size = training_mem.shape[0]
        for max_thresh in sorted(threshs):
            subset = filter_subset(training_mem, min_thresh, max_thresh)
            size = min(size, len(subset))
            if equal_size:
                subset = sorted(random.sample(subset, size))
            subset_df = pd.DataFrame(subset, columns=["idx"])
            subset_df.to_csv(os.path.join(dir_path, "subset_{}-{}.csv".format(min_thresh, max_thresh)), index=False)

# Make Disjoint sets
def make_disjoint(orig_set: list, seed: int = random.randint(0, sys.maxsize)):
    random.seed(seed)
    set1 = random.sample(orig_set, len(orig_set)//2)
    set2 = list(set(orig_set) - set(set1))

    set1 = sorted(set1)
    set2 = sorted(set2)

    return set1, set2

# Load memorization values from the npz file
training_mem = INFL['arr_0']  # Replace 'your_key_here' with the actual key if needed ["tr_mem"] cifar 100 mem file
print(training_mem)
# Directory where the subset CSVs will be saved
output_directory = "/home/kwon/cs590_sp24/d/kd_memorization/src/temp"  # Update to your directory

# List of thresholds you want to use to create subsets
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

# Minimum and maximum thresholds (set these based on your specific requirements)
min_threshold = 0  # Example minimum threshold
max_threshold = 1  # Example maximum threshold

# Call the function to save subsets
save_subsets(output_directory, training_mem, thresholds, equal_size=True, min_thresh=min_threshold, max_thresh=max_threshold)
