"""
Generates a dedicated visual architectural diagram for the Simulation Pipeline
and Mathematical Formulation of the Gauss Exciter System.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig = plt.figure(figsize=(14, 11), dpi=300)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title Banner
rect_title = patches.FancyBboxPatch((5, 91), 90, 7.5, boxstyle="round,pad=0.5",
                                    facecolor='#1a237e', edgecolor='none')
ax.add_patch(rect_title)
ax.text(50, 94.7, "SIMULATION PIPELINE & MATHEMATICAL ARCHITECTURE",
        ha='center', va='center', color='white', fontsize=15, fontweight='bold')
ax.text(50, 92.2, "From Capacitor Discharge Inputs to Receptance Coupling & Chatter Stability Outputs",
        ha='center', va='center', color='#b0bec5', fontsize=10.5)

# Box Styles
def draw_box(x, y, w, h, title, content, bg_color, border_color, title_color='black'):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                 facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 2.8, title, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=title_color)
    ax.text(x + 2, y + h/2 - 1.2, content, ha='left', va='center',
            fontsize=8.5, color='#212121', linespacing=1.4)

def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color='#37474f', lw=2.0, mutation_scale=15))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 1.2, label, ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='#1565c0')

# =====================================================================
# ROW 1: INPUTS
# =====================================================================
ax.text(50, 87.5, "STAGE 1: PHYSICAL & EXPERIMENTAL INPUT PARAMETERS",
        ha='center', va='center', fontsize=12, fontweight='bold', color='#0d47a1')

draw_box(4, 73, 28, 12,
         "Coilgun Inputs",
         "• Charging Voltage: V in [20, 80] V\n• Capacitance: C = 470 uF\n• Efficiency: eta = 8.2%\n• Striker: m_p = 0.883 g, R = 3 mm",
         "#e3f2fd", "#1976d2", "#0d47a1")

draw_box(36, 73, 28, 12,
         "Target & Interface Inputs",
         "• Steel Target: E = 210 GPa, nu = 0.3\n• PU Pad (Shore A90): E = 42 MPa\n• Pad thickness = 5 mm\n• Sampling: fs = 20 MHz (dt = 50 ns)",
         "#e8f5e9", "#388e3c", "#1b5e20")

draw_box(68, 73, 28, 12,
         "Substructure Inputs",
         "• Spindle Bearing Dynamics G_s\n• Module Free-Free FRF Matrix: H_ij\n• Cutting Tool Geometry (CAD)\n• Milling Coeffs: Ks = 800 MPa, Nt = 2",
         "#fff3e0", "#f57c00", "#e65100")

# =====================================================================
# ROW 2: SIMULATION ENGINES (PHYSICS MODELS)
# =====================================================================
ax.text(50, 68, "STAGE 2: COMPUTATIONAL SIMULATION ENGINES (NUMERICAL SOLVERS)",
        ha='center', va='center', fontsize=12, fontweight='bold', color='#0d47a1')

draw_box(4, 49, 28, 16,
         "Sim 1: Electromechanical ODE",
         "1. Electrical Stored Energy:\n   E_0 = 0.5 * C * V^2\n2. Striker Impact Velocity:\n   v_0 = V * sqrt(eta * C / m_p) ~ 0.075 * V\n3. Hertz Peak Force Scaling:\n   F_max = (5/4)^(3/5) * kH^(2/5) * m^(3/5) * v_0^(6/5)",
         "#ede7f6", "#512da8", "#311b92")

draw_box(36, 49, 28, 16,
         "Sim 2: Hertz Contact & FFT",
         "1. Non-linear Impact ODE Solver:\n   m_eff * x''(t) + k_H * x(t)^1.5 = 0\n2. FFT Spectrum Analysis (20 MHz):\n   F(f) = integral F(t)*exp(-j2*pi*f*t) dt\n3. Mechanical Low-Pass Filtering:\n   Attenuates 3.8 kHz (-20.8 dB) & 7.2 kHz",
         "#e0f2f1", "#00796b", "#004d40")

draw_box(68, 49, 28, 16,
         "Sim 3: Inverse IRC & Stability",
         "1. Inverse Receptance Coupling (IRC):\n   G_s = [H_12^-1 * (H_11 - G_mm) * H_21^-1]^-1 - H_22\n2. Forward Tool Receptance Coupling:\n   G_11 = Substructure_Coupling(G_s, Tool)\n3. Altintas-Budak Chatter Stability Lobes",
         "#fce4ec", "#c2185b", "#880e4f")

# Connect Row 1 to Row 2
draw_arrow(18, 73, 18, 65, "Electrical parameters")
draw_arrow(50, 73, 50, 65, "Contact interface")
draw_arrow(82, 73, 82, 65, "Structural matrices")

# Cross connection between Sim1, Sim2, Sim3
draw_arrow(32, 57, 36, 57, "v_0, F_max")
draw_arrow(64, 57, 68, 57, "Force Spectrum F(f)")

# =====================================================================
# ROW 3: OUTPUT FIGURES & VALIDATION METRICS
# =====================================================================
ax.text(50, 43.5, "STAGE 3: VERIFIED EXPERIMENTAL OUTPUTS & REPRODUCED FIGURES",
        ha='center', va='center', fontsize=12, fontweight='bold', color='#0d47a1')

draw_box(4, 18, 28, 22,
         "Figure 3: Coilgun Characterization",
         "• Subplot (a): Bench Setup Flow\n• Subplot (b): v_0 (1.5 to 6 m/s) &\n  F_max (12 to 48 N) vs Voltage (20-80V)\n• Subplot (c): Force Profiles:\n  - Bare steel: tau_c = 14 us, F_max = 298 N\n  - Soft pad: tau_c = 400 us, F_max = 50 N\n  - Hammer metal: 250 us / Plastic: 600 us\n• Subplot (d): Spectrum & Bandwidth:\n  - Mode 3.8 kHz attenuated by -20.8 dB\n  - Mode 7.2 kHz attenuated by -40.8 dB",
         "#ffffff", "#1976d2", "#0d47a1")

draw_box(36, 18, 28, 22,
         "Figure 4: 100-Shot Repeatability",
         "• Subplot (a): Manual Hammer\n  - High operator scatter & hand bias\n  - std(F_max) = 7.334 N\n  - std(tau_c) = 0.019 ms (19 us)\n\n• Subplot (b): Gauss Exciter\n  - Precise electronic trigger\n  - std(F_max) = 1.258 N (82.8% reduction!)\n  - std(tau_c) = 0.002 ms (89.5% reduction!)\n  - Rapid H1 SNR convergence ~ sqrt(N)",
         "#ffffff", "#388e3c", "#1b5e20")

draw_box(68, 18, 28, 22,
         "Figure 5: Tool FRF & Chatter SLD",
         "• Subplot (a): Tool-Tip FRF Synthesis\n  - Reference direct hammer: G_11,ref\n  - Raw automated: exhibits parasitic mode\n  - IRC-corrected: eliminates module mode,\n    recovers true spindle dynamics (< 2% err)\n\n• Subplot (b): Chatter Stability Lobe\n  - Speed RPM vs Axial Depth a_lim (mm)\n  - Validated with experimental tests:\n    * Stable points (Circles O)\n    * Chatter points (Crosses X)",
         "#ffffff", "#d32f2f", "#b71c1c")

# Connect Row 2 to Row 3
draw_arrow(18, 49, 18, 40, "Outputs Fig 3")
draw_arrow(50, 49, 50, 40, "Outputs Fig 4")
draw_arrow(82, 49, 82, 40, "Outputs Fig 5")

# Bottom Footer Summary
rect_foot = patches.FancyBboxPatch((4, 2), 92, 12, boxstyle="round,pad=0.5",
                                   facecolor='#eceff1', edgecolor='#90a4ae', linewidth=1.2)
ax.add_patch(rect_foot)
ax.text(50, 11, "KEY SCIENTIFIC TAKEAWAY & ENGINEERING VALUE",
        ha='center', va='center', fontsize=11, fontweight='bold', color='#263238')
ax.text(50, 6.2,
        "1. Deterministic Bandwidth: Polyurethane pad acts as a mechanical filter, shaping pulse to < 3 kHz and suppressing parasitic modes.\n"
        "2. Superior Repeatability: 83-90% reduction in force/duration scatter enables automated unattended in-situ spindle tap testing.\n"
        "3. Decoupling Fidelity: Inverse Receptance Coupling (IRC) completely removes measurement module dynamics from identified spindle FRF.",
        ha='center', va='center', fontsize=8.8, color='#37474f', linespacing=1.4)

plt.tight_layout()
plt.savefig("pipeline_architecture_diagram.png", dpi=300)
plt.savefig("pipeline_architecture_diagram.pdf")
plt.close()
print("Pipeline Architecture Diagram saved to pipeline_architecture_diagram.png & pdf")
