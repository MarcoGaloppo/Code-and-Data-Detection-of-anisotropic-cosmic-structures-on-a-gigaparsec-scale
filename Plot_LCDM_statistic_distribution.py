import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

# --------------------------------------------------
# LOAD STATISTICS
# --------------------------------------------------

T_mock = np.loadtxt("mock_statistics.dat")
T_data = np.loadtxt("data_statistic.dat")

Nmocks = len(T_mock)

# --------------------------------------------------
# COMPUTE p-value
# --------------------------------------------------

p_value = np.mean(T_mock >= T_data)

if p_value == 0:
    sigma = np.sqrt(2*np.log(Nmocks))
    label_sigma = f"> {sigma:.2f}σ"
else:
    sigma = np.sqrt(2)*erfcinv(2*p_value)
    label_sigma = f"{sigma:.2f}σ"

# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.figure(figsize=(6,5))

plt.hist(T_mock,
         bins=30,
         density=True,
         color="steelblue",
         alpha=0.7,
         label="ΛCDM mocks")

plt.axvline(T_data,
            color="red",
            linewidth=2.5,
            label="data")

plt.xscale("log")
plt.legend(fontsize=13)

plt.xlabel(r"$\cal{T}$", fontsize=18)
plt.ylabel("Probability density", fontsize=18)

# plt.title("Distribution of anisotropy statistic")

# annotation box
text = f"N mocks = {Nmocks}\n"
text += f"p-value < {1/Nmocks:.3f}\n"
text += f"significance > {label_sigma}"

plt.text(0.55,0.80,
         text,
         transform=plt.gca().transAxes,
         fontsize=12,
         bbox=dict(facecolor="white",alpha=0.8))

plt.grid(alpha=0.3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.legend(loc="lower right", fontsize=13)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(0.53,0.75),
    fontsize=13
)

from matplotlib.ticker import ScalarFormatter, MaxNLocator

ax = plt.gca()

# few ticks
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

# scientific notation with global factor
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))   # force scientific notation

ax.yaxis.set_major_formatter(formatter)

plt.tight_layout()

plt.savefig("anisotropy_statistic_distribution.jpeg",dpi=300)

plt.show()
