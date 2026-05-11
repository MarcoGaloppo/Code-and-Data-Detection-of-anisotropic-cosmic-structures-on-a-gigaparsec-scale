#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# READ DATA
# --------------------------------------------------

r_var, var, var_err = np.loadtxt(
    "bootstrap_variance.dat",
    unpack=True
)

var_scaled = var
var_scaled_err =  var_err


# --------------------------------------------------
# READ POISSON
# --------------------------------------------------

r_p, var_p_mean, var_p_std = np.loadtxt(
    "poisson_variance.dat",
    unpack=True
)

var_p_scaled =  var_p_mean
var_p_scaled_std =  var_p_std


# --------------------------------------------------
# READ LCDM
# --------------------------------------------------

r_l, var_l_mean, var_l_std = np.loadtxt(
    "LCDM_variance.dat",
    unpack=True
)

var_l_scaled = var_l_mean
var_l_scaled_std =  var_l_std


# --------------------------------------------------
# FIGURE (single panel)
# --------------------------------------------------

plt.figure(figsize=(6,5))

# Data
plt.errorbar(
    r_var,
    var_scaled,
    yerr=var_scaled_err,
    fmt='o',
    markersize=5,
    label="Data",
    capsize=3
)

# Poisson
plt.plot(r_p, var_p_scaled, color='red', label="Poisson mean")

plt.fill_between(
    r_p,
    var_p_scaled - var_p_scaled_std,
    var_p_scaled + var_p_scaled_std,
    color='red',
    alpha=0.3
)

# LCDM
plt.plot(r_l, var_l_scaled, color='blue', label="ΛCDM mean")

plt.fill_between(
    r_l,
    var_l_scaled - var_l_scaled_std,
    var_l_scaled + var_l_scaled_std,
    color='blue',
    alpha=0.3
)

# --------------------------------------------------
# AXES
# --------------------------------------------------

plt.yscale("log")

plt.xlabel(r"$r\;[\mathrm{Mpc}/h]$", fontsize=20)
plt.ylabel(r"$\sigma_\theta^2(r)$", fontsize=20)

plt.xlim(0, 200)
plt.ylim(1e-10, 8e-7)



plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True)
plt.legend(fontsize=14, loc='upper left')

plt.tight_layout()

plt.savefig(
    "fig_variance_r_data_poisson_LCDM.jpeg",
    dpi=300
)

plt.show()

print("✔ Figure written: fig_variance_r_data_poisson_LCDM.jpeg")
