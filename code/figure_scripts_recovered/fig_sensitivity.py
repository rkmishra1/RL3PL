# RECOVERED, NOT A STUDY SCRIPT. Drew Fig. 'One-at-a-time sensitivity analysis' (Sensitivity.png).
# Recovered verbatim from the inline 'python3 -c' command in the August 2026 assistant session log;
# it was never saved as a file. The numbers are typed in by hand. They agree with results/sensitivity_results.pkl written by core/study.py.
# The output path below points to the old project folder. See ../README.md before using.

import os
import numpy as np
import matplotlib.pyplot as plt

out_dir = '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts'
os.makedirs(out_dir, exist_ok=True)

# Data from Sensitivity Analysis (Instance 8)
# Grid: [-35, -20, 0, 20, 35]
grid = np.array([-35, -20, 0, 20, 35])

# 1. Capacity (Ca_j)
z1_caj = np.array([419216.51, 511241.05, 623726.83, 713983.76, 757059.76])
z2_caj = np.array([12850716.50, 12591169.72, 12274054.32, 12021631.72, 11907115.90])
trans_caj = np.array([83716.83, 101949.16, 123928.67, 141582.80, 150208.14])
insp_caj = np.array([150075.67, 183063.75, 223465.83, 255809.03, 271116.72])
stor_caj = np.array([181524.25, 221468.55, 270513.21, 309914.47, 328651.59])
disp_caj = np.array([3899.77, 4759.60, 5819.13, 6677.46, 7083.31])

# 2. Sorting fractions (beta)
z1_beta = np.array([617973.04, 620438.95, 623726.83, 627014.72, 629480.63])
z2_beta = np.array([12126520.14, 12189749.07, 12274054.32, 12358359.57, 12421588.51])

# 3. E-waste fraction (alpha)
z1_alpha = np.array([620117.91, 621664.59, 623726.83, 625789.08, 627335.76])
z2_alpha = np.array([12274054.32, 12274054.32, 12274054.32, 12274054.32, 12274054.32])

# 4. Uncollected Penalty (phi_1)
z1_phi1 = np.array([623726.83, 623726.83, 623726.83, 623726.83, 623726.83])
z2_phi1 = np.array([8125669.50, 9903548.71, 12274054.32, 14644559.94, 16422439.15])

# 5. Delay Penalty (phi_2)
z1_phi2 = np.array([623726.83, 623726.83, 623726.83, 623726.83, 623726.83])
z2_phi2 = np.array([12126520.14, 12189749.07, 12274054.32, 12358359.57, 12421588.51])

# -------------------------------------------------------------
# Create 2x2 Publication Figure
# -------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

colors = {
    'caj': '#E53E3E',
    'beta': '#3182CE',
    'alpha': '#38A169',
    'phi1': '#D69E2E',
    'phi2': '#805AD5'
}

# (a) Sensitivity of Z1 (Total Cost)
ax = axs[0, 0]
ax.plot(grid, z1_caj / 1e3, marker='o', lw=2.2, color=colors['caj'], label=r'Capacity ($\widetilde{Ca}_j$)')
ax.plot(grid, z1_beta / 1e3, marker='s', lw=1.8, color=colors['beta'], label=r'Sorting ($\beta_K,\beta_S,\beta_R$)')
ax.plot(grid, z1_alpha / 1e3, marker='^', lw=1.8, color=colors['alpha'], label=r'E-waste ($\alpha$)')
ax.plot(grid, z1_phi1 / 1e3, marker='D', lw=1.6, ls='--', color=colors['phi1'], label=r'Uncollected Penalty ($\phi_1$)')
ax.plot(grid, z1_phi2 / 1e3, marker='v', lw=1.6, ls=':', color=colors['phi2'], label=r'Delay Penalty ($\phi_2$)')
ax.set_title('(a) Sensitivity of Total Cost (Z1)', fontweight='bold', fontsize=11.5)
ax.set_xlabel('Parameter Variation (%)', fontweight='bold', fontsize=10.5)
ax.set_ylabel('Total Cost Z1 (x10³)', fontweight='bold', fontsize=10.5)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(framealpha=0.9, fontsize=9.5)

# (b) Sensitivity of Z2 (Service Penalty)
ax = axs[0, 1]
ax.plot(grid, z2_phi1 / 1e6, marker='D', lw=2.4, color=colors['phi1'], label=r'Uncollected Penalty ($\phi_1$)')
ax.plot(grid, z2_caj / 1e6, marker='o', lw=2.0, color=colors['caj'], label=r'Capacity ($\widetilde{Ca}_j$)')
ax.plot(grid, z2_beta / 1e6, marker='s', lw=1.8, color=colors['beta'], label=r'Sorting ($\beta_K,\beta_S,\beta_R$)')
ax.plot(grid, z2_phi2 / 1e6, marker='v', lw=1.8, color=colors['phi2'], label=r'Delay Penalty ($\phi_2$)')
ax.plot(grid, z2_alpha / 1e6, marker='^', lw=1.6, ls='--', color=colors['alpha'], label=r'E-waste ($\alpha$)')
ax.set_title('(b) Sensitivity of Service Penalty (Z2)', fontweight='bold', fontsize=11.5)
ax.set_xlabel('Parameter Variation (%)', fontweight='bold', fontsize=10.5)
ax.set_ylabel('Service Penalty Z2 (x10⁶)', fontweight='bold', fontsize=10.5)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(framealpha=0.9, fontsize=9.5)

# (c) Cost Component Breakdown across Capacity Expansion
ax = axs[1, 0]
width = 5.0
ax.bar(grid - 1.5*width/2, trans_caj / 1e3, width=width, label='Transportation', color='#4299E1', edgecolor='black', lw=0.5)
ax.bar(grid - 0.5*width/2, insp_caj / 1e3, width=width, label='Inspection & Sorting', color='#48BB78', edgecolor='black', lw=0.5)
ax.bar(grid + 0.5*width/2, stor_caj / 1e3, width=width, label='Storage & Warehousing', color='#ECC94B', edgecolor='black', lw=0.5)
ax.bar(grid + 1.5*width/2, disp_caj / 1e3, width=width, label='Disposal', color='#F56565', edgecolor='black', lw=0.5)
ax.set_title(r'(c) Cost Components Breakdown under Capacity ($\widetilde{Ca}_j$)', fontweight='bold', fontsize=11.5)
ax.set_xlabel('Capacity Variation (%)', fontweight='bold', fontsize=10.5)
ax.set_ylabel('Cost Component (x10³)', fontweight='bold', fontsize=10.5)
ax.set_xticks(grid)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
ax.legend(framealpha=0.9, fontsize=9.5)

# (d) Spider / Percentage Impact Tornado Chart at +/- 35%
ax = axs[1, 1]
params = [r'Capacity ($\widetilde{Ca}_j$)', r'Sorting ($\beta$)', r'E-waste ($\alpha$)', r'Uncoll. Pen. ($\phi_1$)', r'Delay Pen. ($\phi_2$)']
y_pos = np.arange(len(params))

# % change in Z1 from baseline at +35%
pct_z1 = [
    (z1_caj[-1] - z1_caj[2]) / z1_caj[2] * 100,
    (z1_beta[-1] - z1_beta[2]) / z1_beta[2] * 100,
    (z1_alpha[-1] - z1_alpha[2]) / z1_alpha[2] * 100,
    (z1_phi1[-1] - z1_phi1[2]) / z1_phi1[2] * 100,
    (z1_phi2[-1] - z1_phi2[2]) / z1_phi2[2] * 100
]

# % change in Z2 from baseline at +35%
pct_z2 = [
    (z2_caj[-1] - z2_caj[2]) / z2_caj[2] * 100,
    (z2_beta[-1] - z2_beta[2]) / z2_beta[2] * 100,
    (z2_alpha[-1] - z2_alpha[2]) / z2_alpha[2] * 100,
    (z2_phi1[-1] - z2_phi1[2]) / z2_phi1[2] * 100,
    (z2_phi2[-1] - z2_phi2[2]) / z2_phi2[2] * 100
]

h = 0.35
ax.barh(y_pos - h/2, pct_z1, height=h, label='% Change in Cost (Z1)', color='#3182CE', edgecolor='black', lw=0.6)
ax.barh(y_pos + h/2, pct_z2, height=h, label='% Change in Service Penalty (Z2)', color='#DD6B20', edgecolor='black', lw=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(params, fontsize=10)
ax.axvline(0, color='black', lw=1.0)
ax.set_title('(d) Relative Sensitivity Impact at +35% Variation', fontweight='bold', fontsize=11.5)
ax.set_xlabel('Percentage Change in Objective (%)', fontweight='bold', fontsize=10.5)
ax.grid(True, linestyle='--', alpha=0.5, axis='x')
ax.legend(framealpha=0.9, fontsize=9.5, loc='lower right')

plt.tight_layout()

# Save paths
save_paths = [
    '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Sensitivity.png',
    '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Sensitivity.pdf',
    '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts/Sensitivity.png',
    '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts/Sensitivity.pdf'
]

for p in save_paths:
    plt.savefig(p, dpi=300 if p.endswith('.png') else None)
    print(f'Saved: {p}')

plt.close()
