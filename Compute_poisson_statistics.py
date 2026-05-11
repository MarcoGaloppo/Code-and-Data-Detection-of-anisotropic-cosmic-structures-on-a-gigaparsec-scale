#!/usr/bin/env python3

import numpy as np
from adpd_python import compute_adpd, angular_variance


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

ANG_BINS = 720
DIST_BINS = 50

N_POISSON = 100


# --------------------------------------------------
# READ DATA
# --------------------------------------------------

data = np.loadtxt("points.dat")

x = data[:,0]
y = data[:,1]

N = len(x)

print("Number of galaxies =", N)

r = np.sqrt(x**2 + y**2)

R = np.max(r)

print("Sample radius =", R)


# --------------------------------------------------
# POISSON ENSEMBLE
# --------------------------------------------------

print("Generating Poisson ensemble...")

var_list = []


# --------------------------------------------------
# LOOP OVER POISSON REALIZATIONS
# --------------------------------------------------

for i in range(N_POISSON):

    print("Poisson realization", i + 1)

    theta = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(0, 1, N)

    r_p = R * np.sqrt(u)

    x_p = r_p * np.cos(theta)
    y_p = r_p * np.sin(theta)

    r_centers, p, p_err, M, angle_edges, angle_centers = compute_adpd(
        x_p,
        y_p,
        ang_bins=ANG_BINS,
        dist_bins=DIST_BINS
    )

    var = angular_variance(p)

    var_list.append(var)


# --------------------------------------------------
# CONVERT TO ARRAY
# --------------------------------------------------

var_list = np.array(var_list)


# --------------------------------------------------
# COMPUTE MEAN AND STANDARD DEVIATION
# --------------------------------------------------

var_p_mean = np.mean(var_list, axis=0)

var_p_std = np.std(var_list, axis=0)


# --------------------------------------------------
# WRITE OUTPUT FILE
# --------------------------------------------------

np.savetxt(
    "poisson_variance.dat",
    np.column_stack([
        r_centers,
        var_p_mean,
        var_p_std
    ]),
    header="r  sigma_theta^2_mean  sigma_theta^2_std"
)


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print("\n✔ Poisson variance statistics written:")
print("   poisson_variance.dat")
