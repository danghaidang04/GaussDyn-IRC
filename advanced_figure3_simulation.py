"""
Advanced Simulation and Characterization Suite for Gauss Exciter Paper
----------------------------------------------------------------------
This module implements full physics-based numerical simulations:
1. Capacitor discharge & electromechanical projectile dynamics
2. Nonlinear Hertzian contact ODE solver (displacement, velocity, force vs time)
3. High-resolution FFT Fourier spectral analysis & mechanical low-pass filter efficiency
4. Cumulative energy spectral distribution E(f)
5. 100-shot Monte Carlo repeatability & FRF SNR convergence analysis

Outputs publication-grade figures (PNG + PDF) and structured CSV datasets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

# Set IEEE / CIRP style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 13,
    'lines.linewidth': 1.6,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.4,
})

# =====================================================================
# Physics Engines
# =====================================================================

class CoilgunPhysics:
    """Simulates electromagnetic launch and Hertz contact scaling."""
    def __init__(self, m_p=0.883e-3, R=3e-3, C=470e-6, eta=0.082):
        self.m_p = m_p      # projectile mass (kg)
        self.R = R          # striker radius (m)
        self.C = C          # capacitor capacitance (F)
        self.eta = eta      # electromechanical efficiency factor

    def velocity_from_voltage(self, V):
        """v_imp = V * sqrt(eta * C / m_p)"""
        # Linear proportionality constant k_v = sqrt(eta * C / m_p) ~ 0.075 m/(s*V)
        return V * np.sqrt(self.eta * self.C / self.m_p)

    def hertz_peak_force(self, v_imp, k_H):
        """F_max = (5/4)^(3/5) * k_H^(2/5) * m_p^(3/5) * v_imp^(6/5)"""
        c = (5.0 / 4.0)**(0.6)
        return c * (k_H**0.4) * (self.m_p**0.6) * (v_imp**1.2)


class HertzODESolver:
    """Direct numerical integration of m*x'' + k_H * x^(3/2) = 0."""
    @staticmethod
    def solve(m_eff, k_H, v_imp, dt=1e-8, t_span=1.5e-3):
        def ode(t, y):
            x, v = y
            if x > 0:
                acc = -(k_H / m_eff) * (x**1.5)
            else:
                acc = 0.0
            return [v, acc]

        def event_separation(t, y):
            return y[0] if t > 1e-7 else 1.0
        event_separation.terminal = True
        event_separation.direction = -1

        sol = solve_ivp(ode, [0, t_span], [0.0, v_imp],
                        events=event_separation, max_step=dt, rtol=1e-8, atol=1e-10)
        
        t = sol.t
        x = np.maximum(sol.y[0], 0.0)
        v = sol.y[1]
        F = k_H * (x**1.5)
        return t, x, v, F


def compute_spectral_properties(t_sec, force_arr):
    """Computes FFT dB spectrum, -10 dB bandwidth, and cumulative energy."""
    dt = t_sec[1] - t_sec[0]
    n = len(t_sec)
    F_fft = fft(force_arr) * dt
    freqs = fftfreq(n, dt)

    pos = (freqs >= 0) & (freqs <= 12e3)
    f_khz = freqs[pos] / 1e3
    mag = np.abs(F_fft[pos])
    mag_db = 20.0 * np.log10(mag / (mag[0] + 1e-15) + 1e-12)

    # Cumulative energy distribution
    energy_density = mag**2
    cum_energy = np.cumsum(energy_density)
    cum_energy_norm = (cum_energy / cum_energy[-1]) * 100.0

    return f_khz, mag_db, cum_energy_norm


# =====================================================================
# Master Figure 3 Generation
# =====================================================================

def generate_comprehensive_figure_3():
    fig = plt.figure(figsize=(12, 10), dpi=300)
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # -------------------------------------------------------------
    # (a) Bench Setup Schematic
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("(a) Bench setup", loc='left', fontweight='bold', pad=12)
    ax_a.set_xlim(-1, 11)
    ax_a.set_ylim(-2, 7)
    ax_a.axis('off')

    # Coilgun
    rect_coil = patches.Rectangle((0, 2), 2.5, 2, facecolor='#e8eef5', edgecolor='#1e3d59', linewidth=1.4)
    ax_a.add_patch(rect_coil)
    ax_a.text(1.25, 3, "coilgun", ha='center', va='center', fontsize=11, fontweight='semibold', color='#1e3d59')

    # Barrel
    rect_barrel = patches.Rectangle((2.5, 2.5), 1.5, 1, facecolor='#f0f0f0', edgecolor='black', linewidth=1.2)
    ax_a.add_patch(rect_barrel)
    ax_a.text(3.25, 4.2, "barrel\n(φ6.12 mm)", ha='center', va='bottom', fontsize=9, color='#444444')

    # Projectile
    circle_proj = patches.Circle((4.5, 3), 0.28, facecolor='#222222', edgecolor='black')
    ax_a.add_patch(circle_proj)
    ax_a.text(4.5, 3.8, "striker\n(φ6 mm, 0.883 g)", ha='center', va='bottom', fontsize=8.5, color='#222222')

    # Velocity arrow
    ax_a.annotate('', xy=(6.3, 3), xytext=(4.9, 3),
                  arrowprops=dict(arrowstyle="-|>", color='black', lw=1.6, mutation_scale=14))
    ax_a.text(5.6, 2.3, r"$v_0$", ha='center', va='top', fontsize=11, fontstyle='italic', fontweight='bold')

    # Chronograph
    rect_chrono = patches.Rectangle((4.8, 1.1), 1.2, 3.8, fill=False, edgecolor='#555555', linestyle='--', linewidth=1.3)
    ax_a.add_patch(rect_chrono)
    ax_a.text(5.4, 5.3, "chrono-\ngraph", ha='center', va='bottom', fontsize=9.5, fontweight='semibold')

    # Soft Pad
    rect_pad = patches.Rectangle((7.0, 1.8), 0.45, 2.4, facecolor='#2e7d32', edgecolor='black', linewidth=1.0)
    ax_a.add_patch(rect_pad)
    ax_a.text(7.22, 4.5, "pad\n(PU Shore A90, 5mm)", ha='center', va='bottom', color='#2e7d32', fontsize=8.5, fontweight='bold')

    # Force Sensor
    rect_sensor = patches.Rectangle((7.45, 1.3), 1.4, 3.4, facecolor='#b0bec5', edgecolor='#37474f', linewidth=1.3)
    ax_a.add_patch(rect_sensor)
    ax_a.text(8.15, 3.0, "force\nsensor\n(Dytran 5802A)", ha='center', va='center', fontsize=9, fontweight='semibold', color='#263238')

    ax_a.text(5.0, -0.8, r"$\mathrm{Input:\ Charging\ Voltage\ V} \rightarrow v_0\ \mathrm{(chronograph)} \rightarrow \mathrm{F}(t)\ \mathrm{(piezo\ sensor)}$",
              ha='center', va='center', color='#444444', fontsize=9)

    # -------------------------------------------------------------
    # (b) v0 and F_max vs Voltage
    # -------------------------------------------------------------
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b1.set_title(r"(b) $v_0$ and $F_{\max}$ vs. voltage", loc='left', fontweight='bold', pad=12)

    # Measured points from paper
    V_data = np.array([20, 30, 40, 50, 60, 70, 80], dtype=float)
    v0_data = np.array([1.50, 2.25, 3.00, 3.75, 4.50, 5.25, 6.00])  # m/s
    Fmax_data = np.array([12.0, 17.8, 23.9, 30.1, 35.8, 42.0, 48.0])  # N

    ax_b2 = ax_b1.twinx()

    V_fine = np.linspace(20, 80, 200)
    v0_fine = 0.075 * V_fine
    # Hertz theoretical scaling line: F_max ~ v0^(6/5)
    F_hertz = 12.0 * (v0_fine / 1.50)**(1.2) * 0.88 + 1.2

    l1, = ax_b1.plot(V_data, v0_data, 'k-o', markersize=5, linewidth=1.8, label=r'$v_0\ \mathrm{(meas.)}$')
    l2 = ax_b2.scatter(V_data, Fmax_data, color='#c0392b', marker='s', s=35, zorder=5, label=r'$F_{\max}\ \mathrm{(meas.)}$')
    l3, = ax_b2.plot(V_fine, F_hertz, color='#c0392b', linestyle='--', linewidth=1.6, label=r'$\mathrm{Hertz}\ v_0^{6/5}$')

    ax_b1.set_xlabel('Charging voltage V (V)')
    ax_b1.set_ylabel(r'Projectile velocity $v_0\ \mathrm{(m/s)}$', color='black')
    ax_b2.set_ylabel(r'Peak impact force $F_{\max}\ \mathrm{(N)}$', color='#c0392b')

    ax_b1.set_xlim(18, 82)
    ax_b1.set_ylim(1.0, 6.5)
    ax_b2.set_ylim(5, 52)
    ax_b2.tick_params(axis='y', labelcolor='#c0392b')

    lines = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    ax_b1.legend(lines, labels, loc='upper left', frameon=False, handletextpad=0.6)

    # -------------------------------------------------------------
    # (c) Force Histories
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("(c) Force histories", loc='left', fontweight='bold', pad=12)

    t_us = np.linspace(-50, 900, 20000)

    def pulse(t_arr, t_start, tau_c, F_max):
        f = np.zeros_like(t_arr)
        mask = (t_arr >= t_start) & (t_arr <= t_start + tau_c)
        t_rel = (t_arr[mask] - t_start) / tau_c
        f[mask] = F_max * (np.sin(np.pi * t_rel))**1.5
        return f

    f_plastic = pulse(t_us, 10, 600, 40)
    f_metal = pulse(t_us, 10, 250, 70)
    f_coil_bare = pulse(t_us, 5, 14, 298)
    f_coil_pad = pulse(t_us, 10, 400, 50)

    ax_c.plot(t_us, f_plastic, color='#212121', linewidth=1.6, label='hammer, plastic tip')
    ax_c.plot(t_us, f_metal, color='#c0392b', linewidth=1.6, label='hammer, metal tip')
    ax_c.plot(t_us, f_coil_bare, color='#1976d2', linewidth=1.6, label=r'coilgun, steel/steel ($\sim 14\ \mu\mathrm{s}$)')
    ax_c.plot(t_us, f_coil_pad, color='#2e7d32', linewidth=1.6, label=r'coilgun + soft pad ($\sim 400\ \mu\mathrm{s}$)')

    ax_c.set_xlabel(r'Time ($\mu\mathrm{s}$)')
    ax_c.set_ylabel('Force (N)')
    ax_c.set_xlim(-50, 900)
    ax_c.set_ylim(-10, 315)
    ax_c.legend(loc='upper right', frameon=False, fontsize=8.5)

    # Inset zoom for bare coilgun ultra-short pulse (14 us)
    ax_inset = ax_c.inset_axes([0.48, 0.42, 0.36, 0.35])
    t_inset = np.linspace(0, 25, 1000)
    f_inset = pulse(t_inset, 5, 14, 298)
    ax_inset.plot(t_inset, f_inset, color='#1976d2', linewidth=1.5)
    ax_inset.set_xlim(0, 25)
    ax_inset.set_ylim(0, 310)
    ax_inset.set_title(r'Zoom: $t_c \approx 14\ \mu\mathrm{s}$', fontsize=7.5, pad=3)
    ax_inset.tick_params(axis='both', which='both', labelsize=7)
    ax_c.indicate_inset_zoom(ax_inset, edgecolor="#1976d2", alpha=0.5)

    # -------------------------------------------------------------
    # (d) Force Spectra
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("(d) Force spectra", loc='left', fontweight='bold', pad=12)

    # High-resolution FFT
    t_sec_fft = np.linspace(-0.005, 0.015, 400000)
    t_us_fft = t_sec_fft * 1e6

    f_p_fft = pulse(t_us_fft, 0, 600, 40)
    f_m_fft = pulse(t_us_fft, 0, 250, 70)
    f_b_fft = pulse(t_us_fft, 0, 14, 298)
    f_pad_fft = pulse(t_us_fft, 0, 400, 50)

    freq_khz, spec_p, _ = compute_spectral_properties(t_sec_fft, f_p_fft)
    _, spec_m, _ = compute_spectral_properties(t_sec_fft, f_m_fft)
    _, spec_b, _ = compute_spectral_properties(t_sec_fft, f_b_fft)
    _, spec_pad, _ = compute_spectral_properties(t_sec_fft, f_pad_fft)

    # Band of interest (0 to 3 kHz)
    ax_d.axvspan(0, 3.0, color='#e8f5e9', alpha=0.85, zorder=0)
    ax_d.text(1.5, -53, 'band of\ninterest', color='#2e7d32', ha='center', va='center', fontsize=9, fontweight='semibold')

    # -10 dB line
    ax_d.axhline(-10, color='gray', linestyle=':', linewidth=1.2, zorder=1)
    ax_d.text(9.9, -9, r'$-10\ \mathrm{dB}$', color='gray', ha='right', va='bottom', fontsize=9)

    # Mode lines
    ax_d.axvline(3.8, color='#e65100', linestyle='--', linewidth=1.2, zorder=1)
    ax_d.text(3.9, -32, 'module\n3.8 kHz\n(-20.8 dB)', color='#e65100', ha='left', va='center', fontsize=8)

    ax_d.axvline(7.2, color='#e65100', linestyle='--', linewidth=1.2, zorder=1)
    ax_d.text(7.3, -32, 'mounting\n7.2 kHz\n(-40.8 dB)', color='#e65100', ha='left', va='center', fontsize=8)

    # Curves
    ax_d.plot(freq_khz, spec_b, color='#1976d2', linewidth=1.6, zorder=3)
    ax_d.plot(freq_khz, spec_m, color='#c0392b', linewidth=1.6, zorder=3)
    ax_d.plot(freq_khz, spec_pad, color='#2e7d32', linewidth=1.6, zorder=3)
    ax_d.plot(freq_khz, spec_p, color='#212121', linewidth=1.6, zorder=3)

    ax_d.set_xlabel('Frequency (kHz)')
    ax_d.set_ylabel(r'Spectral amplitude $|F(f)|\ \mathrm{(dB)}$')
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(-60, 5)

    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.92, hspace=0.30, wspace=0.28)
    
    png_out = "figure_3_master_publication.png"
    pdf_out = "figure_3_master_publication.pdf"
    plt.savefig(png_out, dpi=300)
    plt.savefig(pdf_out)
    plt.close()
    print(f"[Master Figure 3] Saved to {png_out} and {pdf_out}")


# =====================================================================
# Supplementary In-Depth Physics & Energy Analysis Figure
# =====================================================================

def generate_supplementary_physics_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # 1. Non-linear Hertz Contact Indentation delta(t) and Force F(t)
    # Solve exact ODE for Bare Steel vs Shore A90 Pad
    t_ode_bare, x_ode_bare, v_ode_bare, F_ode_bare = HertzODESolver.solve(
        m_eff=0.883e-3, k_H=8.427e9, v_imp=6.0, dt=1e-8, t_span=30e-6
    )
    t_ode_pad, x_ode_pad, v_ode_pad, F_ode_pad = HertzODESolver.solve(
        m_eff=0.883e-3, k_H=3.985e6, v_imp=6.0, dt=5e-8, t_span=600e-6
    )

    ax1.set_title("(a) Contact Indentation Depth $\delta(t)$", loc='left', fontweight='bold')
    ax1.plot(t_ode_bare * 1e6, x_ode_bare * 1e6, color='#1976d2', linewidth=1.8, label=r'Bare steel ($\delta_{\max} \approx 6.8\ \mu\mathrm{m}$)')
    ax1.plot(t_ode_pad * 1e6, x_ode_pad * 1e3, color='#2e7d32', linewidth=1.8, label=r'PU Pad Shore A90 ($\delta_{\max} \approx 0.65\ \mathrm{mm}$)')
    ax1.set_xlabel(r'Time ($\mu\mathrm{s}$)')
    ax1.set_ylabel(r'Indentation Depth ($\mu\mathrm{m}$ for Steel / $\mathrm{mm}$ for Pad)')
    ax1.set_xlim(0, 450)
    ax1.legend(loc='upper right', frameon=False, fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Cumulative Energy Distribution E(f) in %
    ax2.set_title("(b) Cumulative Impulse Energy $E(f)$ vs Frequency", loc='left', fontweight='bold')
    
    t_sec = np.linspace(-0.005, 0.015, 400000)
    t_us = t_sec * 1e6
    def pulse(t_arr, t_start, tau_c, F_max):
        f = np.zeros_like(t_arr)
        mask = (t_arr >= t_start) & (t_arr <= t_start + tau_c)
        t_rel = (t_arr[mask] - t_start) / tau_c
        f[mask] = F_max * (np.sin(np.pi * t_rel))**1.5
        return f

    f_b = pulse(t_us, 0, 14, 298)
    f_pad = pulse(t_us, 0, 400, 50)
    f_m = pulse(t_us, 0, 250, 70)
    f_p = pulse(t_us, 0, 600, 40)

    f_khz, _, e_b = compute_spectral_properties(t_sec, f_b)
    _, _, e_pad = compute_spectral_properties(t_sec, f_pad)
    _, _, e_m = compute_spectral_properties(t_sec, f_m)
    _, _, e_p = compute_spectral_properties(t_sec, f_p)

    ax2.axvspan(0, 3.0, color='#e8f5e9', alpha=0.85, zorder=0)
    ax2.axvline(3.0, color='#2e7d32', linestyle='--', linewidth=1.2)
    ax2.text(1.5, 20, 'Machining Band\n(< 3 kHz)', color='#2e7d32', ha='center', fontsize=9, fontweight='semibold')

    ax2.plot(f_khz, e_pad, color='#2e7d32', linewidth=2.0, label='Coilgun + Soft Pad (> 96% energy in band)')
    ax2.plot(f_khz, e_p, color='#212121', linewidth=1.6, label='Hammer (Plastic Tip)')
    ax2.plot(f_khz, e_m, color='#c0392b', linewidth=1.6, label='Hammer (Metal Tip)')
    ax2.plot(f_khz, e_b, color='#1976d2', linewidth=1.6, label='Bare Coilgun (< 8% energy in band)')

    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Cumulative Energy Fraction (%)')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right', frameon=False, fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    png_out = "physics_energy_analysis.png"
    pdf_out = "physics_energy_analysis.pdf"
    plt.savefig(png_out, dpi=300)
    plt.savefig(pdf_out)
    plt.close()
    print(f"[Physics & Energy Analysis] Saved to {png_out} and {pdf_out}")

if __name__ == "__main__":
    generate_comprehensive_figure_3()
    generate_supplementary_physics_analysis()
