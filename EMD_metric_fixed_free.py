# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:07:36 2025

@author: Ismail Huseynov
"""


import numpy as np
from scipy.spatial.distance import cdist
from mne import read_forward_solution, convert_forward_solution
from ot.emd2 import emd2  # Earth Mover's Distance (Wasserstein-2)

# -----------------------------------------------------
# Utility: Get path to the subject-specific forward model
# -----------------------------------------------------
def get_fwd_fname(subject):
    return f"calibrain/tests/data/{subject}-fwd.fif"

# -----------------------------------------------------
# Main Function: EMD Calculation (from BSI Zoo)
# -----------------------------------------------------
def emd(x, x_hat, orientation_type, subject):
    """
    Compute Earth Mover's Distance (EMD) between true and estimated source activations.

    Parameters:
    - x : (n_sources, n_times) or (n_sources, 3, n_times)
        Ground truth source time courses.
    - x_hat : same shape as x
        Estimated source time courses.
    - orientation_type : str
        'fixed' or 'free' for orientation modeling.
    - subject : str
        Subject ID used to locate the forward model.

    Returns:
    - float
        Earth Mover's Distance between normalized source distributions.
    """

    # Step 1: Compute spatial amplitudes per source
    if orientation_type == "fixed":
        a = np.linalg.norm(x, axis=1)
        b = np.linalg.norm(x_hat, axis=1)
    elif orientation_type == "free":
        a = np.linalg.norm(np.linalg.norm(x, axis=2), axis=1)
        b = np.linalg.norm(np.linalg.norm(x_hat, axis=2), axis=1)
    else:
        raise ValueError("orientation_type must be 'fixed' or 'free'")

    # Step 2: Keep only active (non-zero) sources
    a_mask = a != 0
    b_mask = b != 0
    a = a[a_mask]
    b = b[b_mask]

    # Step 3: Load the forward solution and extract source locations
    fwd_fname = get_fwd_fname(subject)
    fwd = read_forward_solution(fwd_fname)
    fwd = convert_forward_solution(fwd, force_fixed=True)
    src = fwd["src"]

    rr_a = np.r_[
        src[0]["rr"][a_mask[:len(src[0]["rr"])]],          # LH
        src[1]["rr"][a_mask[len(src[0]["rr"]):]]           # RH
    ]
    rr_b = np.r_[
        src[0]["rr"][b_mask[:len(src[0]["rr"])]],
        src[1]["rr"][b_mask[len(src[0]["rr"]):]]
    ]

    # Step 4: Compute ground distance matrix between active sources
    M = cdist(rr_a, rr_b, metric="euclidean")

    # Step 5: Normalize amplitude vectors to form valid probability distributions
    a /= a.sum()
    b /= b.sum()

    # Step 6: Compute EMD
    return emd2(a, b, M)

# -----------------------------------------------------
# Example Usage
# -----------------------------------------------------
subject_list = ["CC120166", "CC120313", "CC120264", "CC120313", "CC120309"]
subject = subject_list[0]

emd_score = emd(x, x_hat, orientation_type="fixed", subject=subject)
print("EMD Score:", emd_score)
