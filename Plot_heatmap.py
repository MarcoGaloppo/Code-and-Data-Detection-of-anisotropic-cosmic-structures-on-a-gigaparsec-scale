import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation, FuncFormatter, ScalarFormatter

def load_adpd_heatmap(n_files=10, file_pattern="angular_dist_bin_{}_normalized.dat"):
    adpd_list = []
    theta_vals = None
    r_vals = []

    for i in range(1, n_files + 1):
        filename = file_pattern.format(i)
        data = np.loadtxt(filename)

        theta = data[:, 0]         # Column 1: θ
        adpd = data[:, 1]          # Column 2: ADPD(θ)
        r = data[0, 2]             # Column 3: r (assumed constant within file)

        if theta_vals is None:
            theta_vals = theta     # Store θ grid once

        adpd_list.append(adpd)
        r_vals.append(r)

    adpd_array = np.array(adpd_list)  # Shape: (N_files, N_theta)
    r_vals = np.array(r_vals)

    return theta_vals, r_vals, adpd_array

def sci_two_decimals(x, pos):
    return f"{x:.2e}"  # scientific notation + 2 decimals

def plot_adpd_heatmap(theta, r, adpd_array, vmin=None, vmax=None, output_file="adpd_heatmap.jpeg"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    # --- 1. Compute scientific exponent manually ---
    data_max = np.nanmax(np.abs(adpd_array))
    exponent = int(np.floor(np.log10(data_max)))
    factor = 10.0**exponent

    # --- 2. Rescale array for plotting ---
    scaled_array = adpd_array / factor
    scaled_vmin = None if vmin is None else vmin / factor
    scaled_vmax = None if vmax is None else vmax / factor

    plt.figure(figsize=(10, 6))
    extent = [theta[0], theta[-1], r[0], r[-1]]

    im = plt.imshow(
        scaled_array,
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        vmin=scaled_vmin,
        vmax=scaled_vmax
    )

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(r"$p(\theta,r)$", fontsize=22)

    # --- 3. Tick formatting (exactly two decimals) ---
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=18)

    # --- 4. Add exponent ABOVE the colorbar (manual offset text) ---
    offset_text = cbar.ax.yaxis.get_offset_text()
    offset_text.set_visible(False)  # Hide Matplotlib auto-offset

    # Replace with our own text above the bar
    cbar.ax.text(
        -0.2, 1.02,                # x,y position relative to axis
        fr"$\times 10^{{{exponent}}}$",
        transform=cbar.ax.transAxes,
        fontsize=16,
        va='bottom'
    )

    plt.xlabel(r'$\theta$ [degrees]', fontsize=22)
    plt.ylabel(r'$r\, [\mathrm{Mpc}/h]$', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
   #   plt.ylim([10,800])
    
    plt.tight_layout()
    plt.savefig(output_file, format='jpeg', dpi=300)
    print(f"Figure saved to {output_file}")

    plt.show()

# Run it with custom color limits and JPEG output
theta, r, adpd = load_adpd_heatmap(n_files=50)
plot_adpd_heatmap(theta, r, adpd, vmin=0.0013, vmax=0.00145, output_file="fig_adpd_heatmap.jpeg")
