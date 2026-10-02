import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
})

def simulate_100_impacts():
    """
    Simulates 100 trials for:
    (a) Manual hammer impact (high scatter in peak force and contact duration)
        - Mean Fmax ~ 45 N, std Fmax = 7.334 N
        - Mean tau_c ~ 400 us, std tau_c = 19 us (0.019 ms)
    (b) Gauss exciter impact (electronically triggered, highly repeatable)
        - Mean Fmax ~ 50 N, std Fmax = 1.258 N
        - Mean tau_c ~ 400 us, std tau_c = 2 us (0.002 ms)
    """
    np.random.seed(42)
    N_trials = 100
    t = np.linspace(-50, 750, 4000) # microseconds

    # 1. Manual Hammer
    f_max_manual = np.random.normal(45.0, 7.334, N_trials)
    tau_manual = np.random.normal(400.0, 19.0, N_trials)
    t_start_manual = np.random.normal(20.0, 5.0, N_trials)
    
    signals_manual = np.zeros((N_trials, len(t)))
    for i in range(N_trials):
        idx = (t >= t_start_manual[i]) & (t <= t_start_manual[i] + tau_manual[i])
        t_rel = (t[idx] - t_start_manual[i]) / tau_manual[i]
        signals_manual[i, idx] = f_max_manual[i] * (np.sin(np.pi * t_rel))**1.5

    # 2. Gauss Exciter
    f_max_gauss = np.random.normal(50.0, 1.258, N_trials)
    tau_gauss = np.random.normal(400.0, 2.0, N_trials)
    t_start_gauss = np.random.normal(20.0, 0.5, N_trials)

    signals_gauss = np.zeros((N_trials, len(t)))
    for i in range(N_trials):
        idx = (t >= t_start_gauss[i]) & (t <= t_start_gauss[i] + tau_gauss[i])
        t_rel = (t[idx] - t_start_gauss[i]) / tau_gauss[i]
        signals_gauss[i, idx] = f_max_gauss[i] * (np.sin(np.pi * t_rel))**1.5

    # Calculate statistics
    mean_manual = np.mean(signals_manual, axis=0)
    std_manual = np.std(signals_manual, axis=0)
    min_manual = np.min(signals_manual, axis=0)
    max_manual = np.max(signals_manual, axis=0)

    mean_gauss = np.mean(signals_gauss, axis=0)
    std_gauss = np.std(signals_gauss, axis=0)
    min_gauss = np.min(signals_gauss, axis=0)
    max_gauss = np.max(signals_gauss, axis=0)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300, sharey=True)

    # (a) Manual Hammer
    ax1.set_title("(a) Manual impact hammer (100 trials)", loc='left', fontweight='bold')
    # Plot individual traces with low opacity
    for i in range(min(30, N_trials)):
        ax1.plot(t, signals_manual[i], color='#999999', alpha=0.3, linewidth=0.8)
    ax1.fill_between(t, min_manual, max_manual, color='#d32f2f', alpha=0.18, label=r'Trial envelope (min-max)')
    ax1.fill_between(t, np.maximum(0, mean_manual - std_manual), mean_manual + std_manual, color='#d32f2f', alpha=0.35, label=r'$\pm 1\sigma$ variation')
    ax1.plot(t, mean_manual, color='#b71c1c', linewidth=2.0, label=r'Mean trace')
    
    ax1.text(0.04, 0.92, r"$\sigma(F_{\max}) = 7.334\ \mathrm{N}$" + "\n" + r"$\sigma(\tau_c) = 0.019\ \mathrm{ms}$",
             transform=ax1.transAxes, va='top', fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="#fff3e0", ec="#ffb74d", lw=1))

    ax1.set_xlabel(r'Time ($\mu\mathrm{s}$)')
    ax1.set_ylabel('Force (N)')
    ax1.set_xlim(-20, 650)
    ax1.set_ylim(-5, 75)
    ax1.legend(loc='upper right', frameon=False, fontsize=8.5)

    # (b) Gauss Exciter
    ax2.set_title("(b) Gauss exciter (100 trials)", loc='left', fontweight='bold')
    for i in range(min(30, N_trials)):
        ax2.plot(t, signals_gauss[i], color='#999999', alpha=0.3, linewidth=0.8)
    ax2.fill_between(t, min_gauss, max_gauss, color='#2e7d32', alpha=0.18, label=r'Trial envelope (min-max)')
    ax2.fill_between(t, np.maximum(0, mean_gauss - std_gauss), mean_gauss + std_gauss, color='#2e7d32', alpha=0.35, label=r'$\pm 1\sigma$ variation')
    ax2.plot(t, mean_gauss, color='#1b5e20', linewidth=2.0, label=r'Mean trace')

    ax2.text(0.04, 0.92, r"$\sigma(F_{\max}) = 1.258\ \mathrm{N}$ ($-82.8\%$)" + "\n" + r"$\sigma(\tau_c) = 0.002\ \mathrm{ms}$ ($-89.5\%$)",
             transform=ax2.transAxes, va='top', fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#81c784", lw=1))

    ax2.set_xlabel(r'Time ($\mu\mathrm{s}$)')
    ax2.set_xlim(-20, 650)
    ax2.legend(loc='upper right', frameon=False, fontsize=8.5)

    plt.tight_layout()
    plt.savefig('repeatability_100shots_reproduced.png', dpi=300)
    plt.savefig('repeatability_100shots_reproduced.pdf')
    print("Repeatability 100-trial simulation saved to repeatability_100shots_reproduced.png")

if __name__ == "__main__":
    simulate_100_impacts()
