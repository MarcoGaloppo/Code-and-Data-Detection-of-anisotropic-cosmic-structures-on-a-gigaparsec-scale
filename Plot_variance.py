import numpy as np
import matplotlib.pyplot as plt
import os

# File name (same in both locations)
filename = "adpd_angular_variance.dat"

# Define file paths
file_paths = {
    "": filename}

# Plotting
plt.figure(figsize=(10, 6))

for label, path in file_paths.items():
    if os.path.exists(path):
        data = np.loadtxt(path)
        r = data[:, 0]
        G = data[:, 1]
        plt.plot(r, G, 'o', label=label, markersize=8)  # large dots
    else:
        print(f"⚠️ Warning: File {path} not found.")

# Plot styling
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r"$r$ [Mpc/$h$]", fontsize=22)
plt.ylabel(r"$\sigma^2_\theta(r)$", fontsize=22)
plt.xlim(1, 2500)
plt.ylim(5e-9, 1e-6)
plt.grid(True, which="both", ls="--")
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()

# Save and show
plt.savefig("adpd_angular_variance.jpeg", dpi=300)
plt.show()
