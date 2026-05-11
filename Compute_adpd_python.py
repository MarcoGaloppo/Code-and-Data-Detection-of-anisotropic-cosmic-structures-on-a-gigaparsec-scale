#!/usr/bin/env python3
"""
Compute ADPD (Angular Distribution of Pairwise Distances) in 2D for
points inside a circular sample centered at (0,0).

The script:

- reads points from points.dat (columns: x y)
- computes pair separations
- bins separations into DIST_BINS radial shells
- bins pair angles into ANG_BINS angular bins over [0,180) degrees
- computes the normalized angular probability distribution
- computes the angular variance sigma_theta^2(r)

Output files:

    angular_dist_bin_<d>_normalized.dat
        (theta_deg, p, r_center, p_err, M)

    pairs_in_shell.dat
        (r_center, M)

    adpd_angular_variance.dat
        (r_center, variance)

    variance_adpd.jpeg

"""

import os
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# READ INPUT FILE
# --------------------------------------------------

def read_points(path):

    data = np.loadtxt(path)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least 2 columns (x y).")

    x = data[:, 0].astype(np.float64)
    y = data[:, 1].astype(np.float64)

    return x, y


# --------------------------------------------------
# COMPUTE ADPD
# --------------------------------------------------

def compute_adpd(
    x,
    y,
    ang_bins=720,
    dist_bins=50,
    alpha_min_deg=0.0,
    alpha_max_deg=180.0
):

    n = x.size

    if n < 2:
        raise ValueError("Need at least 2 points.")

    # --------------------------------------------------
    # RADIAL LIMITS
    # --------------------------------------------------

    r_point = np.sqrt(x*x + y*y)

    rmin = np.min(r_point)
    rmax = np.max(r_point)

    # --------------------------------------------------
    # ANGULAR BINNING
    # --------------------------------------------------

    sector_angle = alpha_max_deg - alpha_min_deg

    angle_edges = np.linspace(
        alpha_min_deg,
        alpha_max_deg,
        ang_bins + 1
    )

    angle_centers = 0.5 * (
        angle_edges[:-1] + angle_edges[1:]
    )

    # --------------------------------------------------
    # RADIAL BINNING
    # --------------------------------------------------

    dbin = (rmax - rmin) / dist_bins

    r_centers = rmin + (
        np.arange(dist_bins) + 0.5
    ) * dbin

    counts = np.zeros((ang_bins, dist_bins))
    M = np.zeros(dist_bins)

    # --------------------------------------------------
    # PAIR LOOP
    # --------------------------------------------------

    for i in range(n - 1):

        if (i + 1) % 5000 == 0:
            print(f"{i+1} / {n}")

        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]

        dist = np.sqrt(dx*dx + dy*dy)

        # radial bin
        d_bin = np.floor(
            (dist - rmin) / dbin
        ).astype(np.int64)

        good = (
            (d_bin >= 0) &
            (d_bin < dist_bins)
        )

        if not np.any(good):
            continue

        dx = dx[good]
        dy = dy[good]
        d_bin = d_bin[good]

        # --------------------------------------------------
        # ANGLES
        # --------------------------------------------------

        ang = np.degrees(np.arctan2(dy, dx))

        ang[ang < 0.0] += 360.0
        ang[ang >= 180.0] -= 180.0

        a_bin = np.floor(
            (ang - alpha_min_deg)
            / sector_angle
            * ang_bins
        ).astype(np.int64)

        a_bin[a_bin < 0] = -1
        a_bin[a_bin >= ang_bins] = ang_bins - 1

        # --------------------------------------------------
        # ACCUMULATE
        # --------------------------------------------------

        np.add.at(counts, (a_bin, d_bin), 1.0)
        np.add.at(M, d_bin, 1.0)

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------

    p = np.zeros_like(counts)
    p_err = np.zeros_like(counts)

    for d in range(dist_bins):

        if M[d] <= 0:
            continue

        p[:, d] = counts[:, d] / M[d]

        p_err[:, d] = np.sqrt(
            p[:, d] * (1.0 - p[:, d]) / M[d]
        )

    return (
        r_centers,
        p,
        p_err,
        M,
        angle_edges,
        angle_centers
    )


# --------------------------------------------------
# ANGULAR VARIANCE
# --------------------------------------------------

def angular_variance(p):

    ntheta = p.shape[0]

    p_mean = 1.0 / ntheta

    var = np.mean(
        (p - p_mean)**2,
        axis=0
    )

    return var


# --------------------------------------------------
# WRITE OUTPUT FILES
# --------------------------------------------------

def write_outputs(
    outdir,
    r_centers,
    p,
    p_err,
    M,
    angle_centers_deg,
    prefix="angular_dist_bin_",
    suffix="_normalized.dat"
):

    os.makedirs(outdir, exist_ok=True)

    dist_bins = r_centers.size

    # --------------------------------------------------
    # ANGULAR DISTRIBUTIONS
    # --------------------------------------------------

    for d in range(dist_bins):

        fname = os.path.join(
            outdir,
            f"{prefix}{d+1}{suffix}"
        )

        with open(fname, "w") as f:

            for a in range(p.shape[0]):

                f.write(
                    f"{angle_centers_deg[a]:12.6f}  "
                    f"{p[a,d]:18.8e}  "
                    f"{r_centers[d]:12.4f}  "
                    f"{p_err[a,d]:18.8e}  "
                    f"{M[d]:18.8e}\n"
                )

    # --------------------------------------------------
    # PAIRS PER SHELL
    # --------------------------------------------------

    with open(
        os.path.join(outdir, "pairs_in_shell.dat"),
        "w"
    ) as f:

        f.write("# r_center  M_pairs\n")

        for d in range(dist_bins):

            f.write(
                f"{r_centers[d]:12.4f}  "
                f"{M[d]:18.8e}\n"
            )


# --------------------------------------------------
# WRITE VARIANCE
# --------------------------------------------------

def write_variance(outdir, r_centers, var):

    with open(
        os.path.join(outdir, "adpd_angular_variance.dat"),
        "w"
    ) as f:

        f.write("# r   sigma_theta^2\n")

        for r, v in zip(r_centers, var):

            f.write(f"{r:10.4f}  {v:18.8e}\n")


# --------------------------------------------------
# QUICK PLOT
# --------------------------------------------------

def make_quick_plot(outdir, r_centers, var):

    plt.figure()

    plt.plot(r_centers, var)

    plt.xlabel("r")
    plt.ylabel(r"$\sigma_\theta^2(r)$")

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, "variance_adpd.jpeg"),
        dpi=200
    )

    plt.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    in_file = "points.dat"

    outdir = "ADPD_Output_Python"

    ANG_BINS = 720
    DIST_BINS = 50

    x, y = read_points(in_file)

    print(f"N = {x.size}")

    (
        r_centers,
        p,
        p_err,
        M,
        angle_edges,
        angle_centers
    ) = compute_adpd(
        x,
        y,
        ang_bins=ANG_BINS,
        dist_bins=DIST_BINS,
        alpha_min_deg=0.0,
        alpha_max_deg=180.0
    )

    var = angular_variance(p)

    write_outputs(
        outdir,
        r_centers,
        p,
        p_err,
        M,
        angle_centers
    )

    write_variance(
        outdir,
        r_centers,
        var
    )

    make_quick_plot(
        outdir,
        r_centers,
        var
    )

    print(f"\n✔ Wrote outputs to: {outdir}/")

    print("  - angular_dist_bin_*_normalized.dat")
    print("  - pairs_in_shell.dat")
    print("  - adpd_angular_variance.dat")
    print("  - variance_adpd.jpeg")


if __name__ == "__main__":

    main()
