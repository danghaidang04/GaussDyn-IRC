import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.fft import fft, fftfreq

# Set high-quality styling matching IEEE / CIRP style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
})

def create_figure_3(save_path="figure_3_reproduced.png"):
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # ==========================================
    # (a) Bench Setup Schematic
    # ==========================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("(a) Bench setup", loc='left', fontweight='bold', pad=12)
    ax_a.set_xlim(-1, 11)
    ax_a.set_ylim(-2, 7)
    ax_a.axis('off')

    # Coilgun body
    rect_coil = patches.Rectangle((0, 2), 2.5, 2, facecolor='#e8eef5', edgecolor='black', linewidth=1.2)
    ax_a.add_patch(rect_coil)
    ax_a.text(1.25, 3, "coilgun", ha='center', va='center', fontsize=11)

    # Barrel
    rect_barrel = patches.Rectangle((2.5, 2.5), 1.5, 1, facecolor='#f5f5f5', edgecolor='black', linewidth=1.2)
    ax_a.add_patch(rect_barrel)
    ax_a.text(3.25, 4.2, "barrel", ha='center', va='bottom', fontsize=10)

    # Steel Projectile (Sphere)
    circle_proj = patches.Circle((4.5, 3), 0.25, facecolor='black', edgecolor='black')
    ax_a.add_patch(circle_proj)

    # Arrow indicating velocity v0
    ax_a.annotate('', xy=(6.3, 3), xytext=(4.9, 3),
                  arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=12))
    ax_a.text(5.6, 2.4, r"$v_0$", ha='center', va='top', fontsize=11, fontstyle='italic')

    # Chronograph (dashed box)
    rect_chrono = patches.Rectangle((4.8, 1.2), 1.2, 3.6, fill=False, edgecolor='black', linestyle='--', linewidth=1.2)
    ax_a.add_patch(rect_chrono)
    ax_a.text(5.4, 5.2, "chrono-\ngraph", ha='center', va='bottom', fontsize=10)

    # Pad (Shore A90 Polyurethane)
    rect_pad = patches.Rectangle((7.0, 1.8), 0.4, 2.4, facecolor='#2e7d32', edgecolor='black', linewidth=1.0)
    ax_a.add_patch(rect_pad)
    ax_a.text(7.2, 4.4, "pad", ha='center', va='bottom', color='#2e7d32', fontsize=10, fontweight='semibold')

    # Force Sensor (Dytran 5802A)
    rect_sensor = patches.Rectangle((7.4, 1.4), 1.3, 3.2, facecolor='#b0b0b0', edgecolor='black', linewidth=1.2)
    ax_a.add_patch(rect_sensor)
    ax_a.text(8.05, 3.0, "force\nsensor", ha='center', va='center', fontsize=10)

    # Bench setup annotation
    ax_a.text(5.0, -0.6, r"$\mathrm{V} \rightarrow v_0\ \mathrm{(chronograph)},\ \mathrm{F}(t)\ \mathrm{(Dytran\ 5802A)}$",
              ha='center', va='center', color='#666666', fontsize=9.5)

    # ==========================================
    # (b) v0 and F_max vs. Charging Voltage
    # ==========================================
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b1.set_title(r"(b) $v_0$ and $F_{\max}$ vs. voltage", loc='left', fontweight='bold', pad=12)

    # Experimental data
    V_data = np.array([20, 30, 40, 50, 60, 70, 80])
    v0_data = np.array([1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0])  # m/s
    Fmax_data = np.array([12.0, 17.8, 23.9, 30.1, 35.8, 42.0, 48.0])  # N

    # Twin axis for Fmax
    ax_b2 = ax_b1.twinx()

    # Theoretical Hertz curve: Fmax \propto v0^(6/5)
    V_fine = np.linspace(20, 80, 100)
    v0_fine = 0.075 * V_fine
    # Fitted Hertz curve to pass closely through data
    F_hertz = 12.0 * (v0_fine / 1.5)**(1.2) * 0.88 + 1.2 # scaled theoretical trend

    # Plots
    l1, = ax_b1.plot(V_data, v0_data, 'k-o', markersize=5, linewidth=1.8, label=r'$v_0\ \mathrm{(meas.)}$')
    l2 = ax_b2.scatter(V_data, Fmax_data, color='#c0392b', marker='s', s=30, label=r'$F_{\max}\ \mathrm{(meas.)}$')
    l3, = ax_b2.plot(V_fine, F_hertz, color='#c0392b', linestyle='--', linewidth=1.6, label=r'$\mathrm{Hertz}\ v_0^{6/5}$')

    ax_b1.set_xlabel('Charging voltage V (V)')
    ax_b1.set_ylabel(r'$v_0\ \mathrm{(m/s)}$', color='black')
    ax_b2.set_ylabel(r'$F_{\max}\ \mathrm{(N)}$', color='#c0392b')

    ax_b1.set_xlim(18, 82)
    ax_b1.set_ylim(1.2, 6.3)
    ax_b2.set_ylim(8, 50)
    ax_b2.tick_params(axis='y', labelcolor='#c0392b')

    # Combined Legend
    lines = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    ax_b1.legend(lines, labels, loc='upper left', frameon=False, handletextpad=0.5)

    # ==========================================
    # (c) Time-Domain Force Histories
    # ==========================================
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("(c) Force histories", loc='left', fontweight='bold', pad=12)

    t = np.linspace(-100, 900, 10000)  # microseconds
    dt = t[1] - t[0]

    def hertz_pulse(t_arr, t_start, tau_c, F_max):
        """Hertzian elastic contact pulse profile."""
        f = np.zeros_like(t_arr)
        idx = (t_arr >= t_start) & (t_arr <= t_start + tau_c)
        # Normalized Hertz contact pulse: (1 - ((2t - tau)/tau)^2)^(1.5) or sin^(1.5)
        tau_rel = (t_arr[idx] - t_start) / tau_c
        f[idx] = F_max * np.sin(np.pi * tau_rel)**1.5
        return f

    # Define 4 cases matching paper
    # 1. Bare Coilgun on Steel (tau ~ 14 us, Fmax ~ 298 N)
    f_coil_bare = hertz_pulse(t, t_start=5, tau_c=14, F_max=298)

    # 2. Manual Hammer, Metal Tip (tau ~ 250 us, Fmax ~ 70 N)
    f_ham_metal = hertz_pulse(t, t_start=10, tau_c=250, F_max=70)

    # 3. Coilgun + Soft Pad (tau ~ 400 us, Fmax ~ 50 N)
    f_coil_pad = hertz_pulse(t, t_start=10, tau_c=400, F_max=50)

    # 4. Manual Hammer, Plastic Tip (tau ~ 600 us, Fmax ~ 40 N)
    f_ham_plastic = hertz_pulse(t, t_start=10, tau_c=600, F_max=40)

    ax_c.plot(t, f_ham_plastic, color='#1f1f1f', linewidth=1.5, label='hammer, plastic tip')
    ax_c.plot(t, f_ham_metal, color='#c0392b', linewidth=1.5, label='hammer, metal tip')
    ax_c.plot(t, f_coil_bare, color='#1f77b4', linewidth=1.5, label=r'coilgun, steel/steel ($\sim 14\ \mu\mathrm{s}$)')
    ax_c.plot(t, f_coil_pad, color='#2ca02c', linewidth=1.5, label=r'coilgun + soft pad ($\sim 400\ \mu\mathrm{s}$)')

    ax_c.set_xlabel(r'Time ($\mu\mathrm{s}$)')
    ax_c.set_ylabel('Force (N)')
    ax_c.set_xlim(-100, 900)
    ax_c.set_ylim(-10, 315)
    ax_c.legend(loc='upper right', frameon=False, handlelength=1.5, fontsize=8)

    # ==========================================
    # (d) Frequency Spectra (|F(f)| dB)
    # ==========================================
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("(d) Force spectra", loc='left', fontweight='bold', pad=12)

    # Calculate FFT spectrum for each pulse
    # Time vector in seconds with high resolution
    t_sec = np.linspace(-0.005, 0.02, 500000) # 25 ms window, dt = 5e-8 s -> fs = 20 MHz
    dt_sec = t_sec[1] - t_sec[0]
    
    def compute_spectrum(tau_c_us, F_max_val):
        t_us = t_sec * 1e6
        f_time = hertz_pulse(t_us, t_start=0, tau_c=tau_c_us, F_max=F_max_val)
        F_fft = fft(f_time) * dt_sec
        freqs = fftfreq(len(t_sec), dt_sec)
        
        pos = (freqs >= 0) & (freqs <= 15e3)
        freq_khz = freqs[pos] / 1e3
        mag = np.abs(F_fft[pos])
        mag_db = 20 * np.log10(mag / mag[0] + 1e-12)
        return freq_khz, mag_db

    freq_khz, spec_bare = compute_spectrum(14, 298)
    _, spec_metal = compute_spectrum(250, 70)
    _, spec_pad = compute_spectrum(400, 50)
    _, spec_plastic = compute_spectrum(600, 40)

    # Shaded band of interest (0 to 3 kHz)
    ax_d.axvspan(0, 3.0, color='#e5f2eb', alpha=0.9, zorder=0)
    ax_d.text(1.5, -53, 'band of\ninterest', color='#2e7d32', ha='center', va='center', fontsize=9)

    # -10 dB threshold line
    ax_d.axhline(-10, color='gray', linestyle=':', linewidth=1.2, zorder=1)
    ax_d.text(9.9, -9, r'$-10\ \mathrm{dB}$', color='gray', ha='right', va='bottom', fontsize=9)

    # Module mode at 3.8 kHz
    ax_d.axvline(3.8, color='#d95f02', linestyle='--', linewidth=1.2, zorder=1)
    ax_d.text(3.9, -30, 'module\n3.8 kHz', color='#d95f02', ha='left', va='center', fontsize=8.5)

    # Mounting mode at 7.2 kHz
    ax_d.axvline(7.2, color='#d95f02', linestyle='--', linewidth=1.2, zorder=1)
    ax_d.text(7.3, -30, 'mounting\n7.2 kHz', color='#d95f02', ha='left', va='center', fontsize=8.5)

    # Plot spectral curves
    ax_d.plot(freq_khz, spec_bare, color='#1f77b4', linewidth=1.5, zorder=3)
    ax_d.plot(freq_khz, spec_metal, color='#c0392b', linewidth=1.5, zorder=3)
    ax_d.plot(freq_khz, spec_pad, color='#2ca02c', linewidth=1.5, zorder=3)
    ax_d.plot(freq_khz, spec_plastic, color='#1f1f1f', linewidth=1.5, zorder=3)

    ax_d.set_xlabel('Frequency (kHz)')
    ax_d.set_ylabel(r'$|F(f)|\ \mathrm{(dB)}$')
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(-60, 5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Figure saved to {save_path} and {save_path.replace('.png', '.pdf')}")

if __name__ == "__main__":
    create_figure_3()
