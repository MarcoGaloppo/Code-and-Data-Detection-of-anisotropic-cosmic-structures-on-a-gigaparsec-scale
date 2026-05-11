#!/usr/bin/env python3

import numpy as np
import glob
import os

from adpd_python import compute_adpd, angular_variance


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

ANG_BINS = 720
DIST_BINS = 50


# --------------------------------------------------
# LOCATE DIRECTORIES
# --------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
mock_dir   = os.path.join(script_dir, "LCDM_mock_samples")
out_dir    = os.path.join(script_dir, "LCDM_mock_variance")

os.makedirs(out_dir, exist_ok=True)

print("Script directory :", script_dir)
print("Mock directory   :", mock_dir)
print("Output directory :", out_dir)


# --------------------------------------------------
# FIND MOCK FILES
# --------------------------------------------------

files = sorted(glob.glob(os.path.join(mock_dir, "mock_*.dat")))

print("Mocks found:", len(files))

if len(files) == 0:
    raise RuntimeError("No mock samples found")


# --------------------------------------------------
# STORAGE
# --------------------------------------------------

var_list = []


# --------------------------------------------------
# LOOP OVER MOCKS
# --------------------------------------------------

for i, fname in enumerate(files):

    base = os.path.basename(fname).replace(".dat", "")
    variance_file = os.path.join(out_dir, base + "_variance.dat")

    if i % 20 == 0:
        print(f"Processing mock {i}/{len(files)}")

    # --------------------------------------------------
    # IF FILE EXISTS: READ
    # --------------------------------------------------

    if os.path.exists(variance_file):

        data = np.loadtxt(variance_file)

        r_centers = data[:, 0]
        var       = data[:, 1]

    else:

        # --------------------------------------------------
        # COMPUTE ADPD
        # --------------------------------------------------

        data = np.loadtxt(fname)

        x = data[:, 0]
        y = data[:, 1]

        r_centers, p, p_err, M, angle_edges, angle_centers = compute_adpd(
            x,
            y,
            ang_bins=ANG_BINS,
            dist_bins=DIST_BINS
        )

        var = angular_variance(p)

        np.savetxt(
            variance_file,
            np.column_stack([r_centers, var]),
            header="r  sigma_theta^2"
        )

    var_list.append(var)


# --------------------------------------------------
# CONVERT TO ARRAY
# --------------------------------------------------

var_list = np.array(var_list)


# --------------------------------------------------
# COMPUTE LCDM MEAN AND SCATTER
# --------------------------------------------------

var_mean = np.mean(var_list, axis=0)
var_std  = np.std(var_list, axis=0)


# --------------------------------------------------
# WRITE LCDM VARIANCE SUMMARY FILE
# --------------------------------------------------

np.savetxt(
    os.path.join(script_dir, "LCDM_variance.dat"),
    np.column_stack([r_centers, var_mean, var_std]),
    header="r  sigma_theta^2_mean  sigma_theta^2_std"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data_file = os.path.join(script_dir, "bootstrap_variance.dat")

r_data, data_var, data_err = np.loadtxt(data_file, unpack=True)


# --------------------------------------------------
# COVARIANCE FROM MOCKS
# --------------------------------------------------

cov = np.cov(var_list, rowvar=False)

Nmocks = var_list.shape[0]
Nbins  = var_list.shape[1]

hartlap = (Nmocks - Nbins - 2) / (Nmocks - 1)

cov_inv = hartlap * np.linalg.inv(cov)

delta = data_var - var_mean

T_data = delta @ cov_inv @ delta


# --------------------------------------------------
# MOCK DISTRIBUTION
# --------------------------------------------------

T_mock = []

for k in range(Nmocks):

    other = np.delete(var_list, k, axis=0)

    mean_k = np.mean(other, axis=0)
    cov_k  = np.cov(other, rowvar=False)

    hartlap_k = (Nmocks - 1 - Nbins - 2) / (Nmocks - 2)

    cov_k_inv = hartlap_k * np.linalg.inv(cov_k)

    delta_k = var_list[k] - mean_k

    T = delta_k @ cov_k_inv @ delta_k

    T_mock.append(T)

T_mock = np.array(T_mock)


# --------------------------------------------------
# SAVE STATISTICS FOR PLOTTING
# --------------------------------------------------

np.savetxt(
    os.path.join(script_dir, "mock_statistics.dat"),
    T_mock,
    header="T_mock statistics from LCDM mocks"
)

np.savetxt(
    os.path.join(script_dir, "data_statistic.dat"),
    np.array([T_data]),
    header="T_data statistic"
)


# --------------------------------------------------
# P-VALUE
# --------------------------------------------------

p_value = np.mean(T_mock >= T_data)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n=============================")
print("LCDM consistency test")
print("=============================")

print("Number of mocks:", Nmocks)
print("T_data =", T_data)
print("p-value =", p_value)


# --------------------------------------------------
# WRITE RESULTS TO FILE
# --------------------------------------------------

out_file = os.path.join(script_dir, "LCDM_consistency_test.txt")

with open(out_file, "w") as f:

    f.write("LCDM consistency test\n")
    f.write("=====================\n\n")

    f.write(f"Number of mocks : {Nmocks}\n")
    f.write(f"Number of bins  : {Nbins}\n")
    f.write(f"T_data          : {T_data:.6e}\n")
    f.write(f"p-value         : {p_value:.6e}\n")

    if p_value == 0:
        sig = np.sqrt(2*np.log(Nmocks))
        f.write(f"Significance    : > {sig:.2f} sigma\n")
    else:
        from scipy.special import erfcinv
        sigma = np.sqrt(2)*erfcinv(2*p_value)
        f.write(f"Significance    : {sigma:.2f} sigma\n")

print("\nResults written to:")
print(out_file)
print("\nSummary file written:")
print("LCDM_variance.dat")
