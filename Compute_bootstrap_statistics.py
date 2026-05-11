#!/usr/bin/env python3

import numpy as np
import os

from adpd_python import read_points, angular_variance


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

INPUT_FILE = "points.dat"

ANG_BINS = 720
DIST_BINS = 50

N_BOOT = 200   # number of bootstrap realizations

OUTDIR = "bootstrap_samples"

os.makedirs(OUTDIR, exist_ok=True)


# --------------------------------------------------
# WEIGHTED ADPD
# --------------------------------------------------

def compute_adpd_weighted(x, y, w, ang_bins=720, dist_bins=50):

    n = len(x)

    r_point = np.sqrt(x*x + y*y)

    rmin = np.min(r_point)
    rmax = np.max(r_point)

    dbin = (rmax-rmin)/dist_bins

    counts = np.zeros((ang_bins, dist_bins))
    M = np.zeros(dist_bins)

    for i in range(n-1):

        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]

        dist = np.sqrt(dx*dx + dy*dy)

        d_bin = np.floor((dist-rmin)/dbin).astype(int)

        good = (d_bin>=0) & (d_bin<dist_bins)

        if not np.any(good):
            continue

        dx = dx[good]
        dy = dy[good]
        d_bin = d_bin[good]

        w_pair = w[i] * w[i+1:][good]

        ang = np.degrees(np.arctan2(dy,dx))

        ang[ang<0] += 360
        ang[ang>=180] -= 180

        a_bin = np.floor(ang/180*ang_bins).astype(int)

        for k in range(len(a_bin)):

            a = a_bin[k]
            d = d_bin[k]

            if a<0 or a>=ang_bins:
                continue

            counts[a,d] += w_pair[k]
            M[d] += w_pair[k]

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------

    p = np.zeros_like(counts)

    for d in range(dist_bins):

        if M[d] > 0:

            p[:,d] = counts[:,d] / M[d]

    r_centers = rmin + (
        np.arange(dist_bins)+0.5
    ) * dbin

    return r_centers, p, M


# --------------------------------------------------
# READ DATA
# --------------------------------------------------

x, y = read_points(INPUT_FILE)

N = len(x)

print("N galaxies =", N)


# --------------------------------------------------
# FULL SAMPLE
# --------------------------------------------------

print("Computing ADPD for full sample")

w = np.ones(N)

r_centers, p_full, M_full = compute_adpd_weighted(
    x,
    y,
    w,
    ANG_BINS,
    DIST_BINS
)

var_full = angular_variance(p_full)


# --------------------------------------------------
# STORAGE
# --------------------------------------------------

var_boot = []


# --------------------------------------------------
# BOOTSTRAP LOOP
# --------------------------------------------------

for b in range(N_BOOT):

    print("Bootstrap", b+1, "/", N_BOOT)

    # --------------------------------------------------
    # POISSON BOOTSTRAP WEIGHTS
    # --------------------------------------------------

    w = np.random.poisson(1, size=N)

    r_centers, p, M = compute_adpd_weighted(
        x,
        y,
        w,
        ANG_BINS,
        DIST_BINS
    )

    var = angular_variance(p)

    var_boot.append(var)

    # --------------------------------------------------
    # SAVE INDIVIDUAL REALIZATION
    # --------------------------------------------------

    np.savetxt(
        f"{OUTDIR}/bootstrap_{b:03d}_variance.dat",
        np.column_stack([r_centers, var]),
        header="r sigma_theta^2"
    )


# --------------------------------------------------
# CONVERT TO ARRAY
# --------------------------------------------------

var_boot = np.array(var_boot)


# --------------------------------------------------
# COMPUTE MEAN AND ERROR
# --------------------------------------------------

var_mean = np.mean(var_boot, axis=0)

var_err = np.std(var_boot, axis=0)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

np.savetxt(
    "bootstrap_variance.dat",
    np.column_stack([
        r_centers,
        var_mean,
        var_err
    ]),
    header="r sigma_theta^2 error"
)


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print("\n✔ Bootstrap variance statistics computed")
print("   bootstrap_variance.dat")
