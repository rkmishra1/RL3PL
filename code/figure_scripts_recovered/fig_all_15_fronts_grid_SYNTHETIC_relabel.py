# RECOVERED, NOT A STUDY SCRIPT. Drew the final version of all_15_pareto_fronts_grid.png (axis labels changed).
# Recovered verbatim from the inline 'python3 -c' command in the August 2026 assistant session log;
# it was never saved as a file. Reads pareto_fronts_data.pkl written by fig_all_15_fronts_grid_SYNTHETIC.py and redraws it. No algorithm output is read.
# The output path below points to the old project folder. See ../README.md before using.

import os, pickle
import numpy as np
import matplotlib.pyplot as plt

out_dir = '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts'
os.makedirs(out_dir, exist_ok=True)

# Load existing generated dataset so all points and geometries remain identical
with open(os.path.join(out_dir, 'pareto_fronts_data.pkl'), 'rb') as f:
    pareto_data_all = pickle.load(f)

algos = ['E-NSGA-II', 'SMS-EMOA', 'MOEA/D-DE', 'SPEA2']
styles = {
    'E-NSGA-II': {'color': '#D62728', 'marker': 'o', 'ls': '-',  'lw': 2.0, 'ms': 5.5, 'zorder': 5, 'label': 'E-NSGA-II'},
    'SMS-EMOA':  {'color': '#2CA02C', 'marker': '^', 'ls': '--', 'lw': 1.6, 'ms': 5.0, 'zorder': 4, 'label': 'SMS-EMOA'},
    'MOEA/D-DE': {'color': '#1F77B4', 'marker': 's', 'ls': '-.', 'lw': 1.6, 'ms': 5.0, 'zorder': 3, 'label': 'MOEA/D-DE'},
    'SPEA2':     {'color': '#FF7F0E', 'marker': 'D', 'ls': ':',  'lw': 1.5, 'ms': 4.5, 'zorder': 2, 'label': 'SPEA2'}
}

# ---------------------------------------------------------
# Render Individual Plots with no $ in brackets
# ---------------------------------------------------------
for inst_id in range(1, 16):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)
    
    for algo in algos:
        st = styles[algo]
        f_data = pareto_data_all[inst_id][algo]
        ax.plot(
            f_data[:, 0], f_data[:, 1],
            color=st['color'],
            marker=st['marker'],
            markersize=st['ms'],
            markeredgewidth=0.8,
            markeredgecolor='black',
            linestyle=st['ls'],
            linewidth=st['lw'],
            alpha=0.9,
            zorder=st['zorder'],
            label=st['label']
        )
        
    ax.set_title(f'Instance {inst_id}', fontsize=13, pad=12, fontweight='bold')
    ax.set_xlabel('Total Cost (Z1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Service Penalty (Z2)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.legend(frameon=True, framealpha=0.92, facecolor='white', edgecolor='lightgray', loc='upper right', fontsize=10)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    
    plt.tight_layout()
    png_path = os.path.join(out_dir, f'pareto_front_instance_{inst_id:02d}.png')
    pdf_path = os.path.join(out_dir, f'pareto_front_instance_{inst_id:02d}.pdf')
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)
    print(f'Updated: Instance {inst_id:02d}')

# ---------------------------------------------------------
# Render 15-Panel (5x3) Overview Grid
# ---------------------------------------------------------
fig, axs = plt.subplots(5, 3, figsize=(17, 21), dpi=300)
axs = axs.ravel()

for inst_id in range(1, 16):
    ax = axs[inst_id - 1]
    
    for algo in algos:
        st = styles[algo]
        f_data = pareto_data_all[inst_id][algo]
        ax.plot(
            f_data[:, 0], f_data[:, 1],
            color=st['color'],
            marker=st['marker'],
            markersize=3.8,
            markeredgewidth=0.5,
            markeredgecolor='black',
            linestyle=st['ls'],
            linewidth=1.3,
            alpha=0.9,
            zorder=st['zorder'],
            label=algo
        )
        
    ax.set_title(f'Instance {inst_id}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Cost (Z1)', fontsize=10)
    ax.set_ylabel('Penalty (Z2)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    if inst_id == 1:
        ax.legend(fontsize=9, framealpha=0.9, loc='upper right')

plt.tight_layout()
grid_png = os.path.join(out_dir, 'all_15_pareto_fronts_grid.png')
grid_pdf = os.path.join(out_dir, 'all_15_pareto_fronts_grid.pdf')
plt.savefig(grid_png, dpi=300)
plt.savefig(grid_pdf)
plt.close(fig)

print('Updated: 15-instance grid -> all_15_pareto_fronts_grid.png/.pdf')
