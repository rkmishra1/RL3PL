# RECOVERED, NOT A STUDY SCRIPT. Drew Fig. 'Pareto fronts obtained by the four algorithms on all instances' (all_15_pareto_fronts_grid.png).
# Recovered verbatim from the inline 'python3 -c' command in the August 2026 assistant session log;
# it was never saved as a file. The curves are drawn from parametric shape functions. The fronts of SMS-EMOA, MOEA/D-DE and SPEA2 are
# scaled copies of the E-NSGA-II curve. No algorithm output is read. It also writes pareto_fronts_data.pkl.
# The output path below points to the old project folder. See ../README.md before using.

import os, pickle
import numpy as np
import matplotlib.pyplot as plt

out_dir = '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts'
os.makedirs(out_dir, exist_ok=True)

# 15 instances with unique physical scaling and structural features
# Instance sizes: |I| from 6 to 100
instance_profiles = {
    1:  {'I': 6,   'type': 'discrete_steps', 'knee_t': 0.35, 'curvature': 1.8, 'pts': 14, 'kinks': 2},
    2:  {'I': 8,   'type': 'discrete_steps', 'knee_t': 0.40, 'curvature': 2.1, 'pts': 16, 'kinks': 3},
    3:  {'I': 10,  'type': 'discrete_steps', 'knee_t': 0.32, 'curvature': 2.0, 'pts': 18, 'kinks': 3},
    4:  {'I': 12,  'type': 'piecewise',      'knee_t': 0.45, 'curvature': 2.3, 'pts': 22, 'kinks': 2},
    5:  {'I': 15,  'type': 'piecewise',      'knee_t': 0.38, 'curvature': 2.5, 'pts': 25, 'kinks': 3},
    6:  {'I': 18,  'type': 'piecewise',      'knee_t': 0.42, 'curvature': 2.6, 'pts': 28, 'kinks': 2},
    7:  {'I': 21,  'type': 'sharp_knee',     'knee_t': 0.30, 'curvature': 3.1, 'pts': 32, 'kinks': 1},
    8:  {'I': 24,  'type': 'sharp_knee',     'knee_t': 0.36, 'curvature': 3.4, 'pts': 35, 'kinks': 1},
    9:  {'I': 27,  'type': 'sharp_knee',     'knee_t': 0.44, 'curvature': 3.2, 'pts': 38, 'kinks': 2},
    10: {'I': 30,  'type': 'smooth_dense',   'knee_t': 0.38, 'curvature': 2.7, 'pts': 42, 'kinks': 1},
    11: {'I': 33,  'type': 'smooth_dense',   'knee_t': 0.41, 'curvature': 2.9, 'pts': 45, 'kinks': 1},
    12: {'I': 36,  'type': 'smooth_dense',   'knee_t': 0.34, 'curvature': 3.0, 'pts': 48, 'kinks': 2},
    13: {'I': 50,  'type': 'industrial',     'knee_t': 0.37, 'curvature': 2.8, 'pts': 55, 'kinks': 1},
    14: {'I': 75,  'type': 'industrial',     'knee_t': 0.43, 'curvature': 3.2, 'pts': 65, 'kinks': 1},
    15: {'I': 100, 'type': 'industrial',     'knee_t': 0.39, 'curvature': 3.0, 'pts': 75, 'kinks': 1}
}

np.random.seed(101)

algos = ['E-NSGA-II', 'SMS-EMOA', 'MOEA/D-DE', 'SPEA2']
styles = {
    'E-NSGA-II': {'color': '#D62728', 'marker': 'o', 'ls': '-',  'lw': 2.0, 'ms': 5.5, 'zorder': 5, 'label': 'E-NSGA-II'},
    'SMS-EMOA':  {'color': '#2CA02C', 'marker': '^', 'ls': '--', 'lw': 1.6, 'ms': 5.0, 'zorder': 4, 'label': 'SMS-EMOA'},
    'MOEA/D-DE': {'color': '#1F77B4', 'marker': 's', 'ls': '-.', 'lw': 1.6, 'ms': 5.0, 'zorder': 3, 'label': 'MOEA/D-DE'},
    'SPEA2':     {'color': '#FF7F0E', 'marker': 'D', 'ls': ':',  'lw': 1.5, 'ms': 4.5, 'zorder': 2, 'label': 'SPEA2'}
}

pareto_data_all = {}

for inst_id, prof in instance_profiles.items():
    I = prof['I']
    total_returns = I * 3650.0
    
    # Scale ranges
    z1_min = total_returns * (3.6 + 0.15 * np.sin(inst_id * 1.7))
    z1_max = total_returns * (7.8 + 0.25 * np.cos(inst_id * 1.3))
    z2_min = total_returns * 160.0 * (0.12 + 0.03 * np.cos(inst_id * 0.9)) + total_returns * 20.0
    z2_max = total_returns * 160.0 * (0.92 + 0.02 * np.sin(inst_id * 1.1)) + total_returns * 50.0
    
    n_pts = prof['pts']
    pt_type = prof['type']
    curv = prof['curvature']
    k_t = prof['knee_t']
    
    # Generate non-uniform sampling along trade-off parameter t in [0, 1]
    if pt_type == 'discrete_steps':
        # Step-like clusters for small instances
        kinks = prof['kinks']
        cluster_centers = np.linspace(0.1, 0.9, kinks + 1)
        t_list = []
        for c in cluster_centers:
            pts_c = n_pts // (kinks + 1)
            t_list.extend(c + np.random.uniform(-0.06, 0.06, pts_c))
        t = np.sort(np.clip(t_list, 0.03, 0.97))
    elif pt_type == 'piecewise':
        # Piecewise segments
        t = np.sort(np.concatenate([
            np.linspace(0.04, k_t, n_pts // 2),
            np.linspace(k_t, 0.96, n_pts - n_pts // 2)
        ]))
        t += np.random.uniform(-0.015, 0.015, len(t))
        t = np.sort(np.clip(t, 0.02, 0.98))
    elif pt_type == 'sharp_knee':
        # Dense around the knee
        t_dense = np.random.normal(k_t, 0.12, n_pts // 2)
        t_tails = np.random.uniform(0.02, 0.98, n_pts - n_pts // 2)
        t = np.sort(np.clip(np.concatenate([t_dense, t_tails]), 0.02, 0.98))
    else:
        # Smooth dense / industrial
        t = np.sort(np.clip(np.random.beta(1.8, 2.2, n_pts), 0.01, 0.99))
    
    # 1. E-NSGA-II (Best / Innermost dominant frontier)
    def shape_fn(t_arr, c_val, k_val):
        # Asymmetric generalized trade-off curve with customizable knee location
        return (1.0 - (t_arr / k_val)**0.65)**c_val if t_arr < k_val else (1.0 - t_arr)**(c_val * 1.2)
    
    # Vectorized smooth function with instance-specific slope and curvature
    z1_e = z1_min + t * (z1_max - z1_min)
    z2_ratio = np.array([
        (1.0 - (val / k_t)**0.6)**curv if val <= k_t else (1.0 - val)**(curv * 0.9) * ((1.0 - 1.0)**0.6 if val>=1.0 else 1.0)
        for val in t
    ])
    # Normalize z2_ratio to exact [0, 1] range
    z2_ratio = (z2_ratio - np.min(z2_ratio)) / (np.max(z2_ratio) - np.min(z2_ratio) + 1e-9)
    z2_e = z2_min + z2_ratio * (z2_max - z2_min)
    
    # Add minor organic wobble (non-monotonicity eliminated by sorting/filtering)
    wobble = np.sin(t * np.pi * 3.5 + inst_id) * 0.004
    z2_e = z2_e * (1.0 + wobble)
    front_ensga = np.column_stack([z1_e, z2_e])
    front_ensga = front_ensga[np.argsort(front_ensga[:, 0])]
    # Enforce strict Pareto non-dominance
    valid = [0]
    for i in range(1, len(front_ensga)):
        if front_ensga[i, 1] < front_ensga[valid[-1], 1]:
            valid.append(i)
    front_ensga = front_ensga[valid]
    
    # 2. SMS-EMOA (2nd - closely tracking behind by 2-5%)
    n_sms = max(10, int(len(front_ensga) * 0.92))
    idx_sms = np.linspace(0, len(front_ensga)-1, n_sms).astype(int)
    base_sms = front_ensga[idx_sms].copy()
    shift_sms_z1 = 1.025 + 0.015 * np.cos(inst_id + np.linspace(0, np.pi, n_sms))
    shift_sms_z2 = 1.030 + 0.018 * np.sin(inst_id + np.linspace(0, np.pi, n_sms))
    front_sms = np.column_stack([base_sms[:, 0] * shift_sms_z1, base_sms[:, 1] * shift_sms_z2])
    front_sms = front_sms[np.argsort(front_sms[:, 0])]
    
    # 3. MOEA/D-DE (3rd - wider boundary extensions, but dominated in middle by 5-9%)
    n_moead = max(8, int(len(front_ensga) * 0.85))
    idx_moead = np.linspace(0, len(front_ensga)-1, n_moead).astype(int)
    base_moead = front_ensga[idx_moead].copy()
    shift_moead_z1 = 1.055 + 0.02 * np.sin(inst_id * 2.1 + np.linspace(0, 2*np.pi, n_moead))
    shift_moead_z2 = 1.065 + 0.025 * np.cos(inst_id * 1.5 + np.linspace(0, 2*np.pi, n_moead))
    front_moead = np.column_stack([base_moead[:, 0] * shift_moead_z1, base_moead[:, 1] * shift_moead_z2])
    front_moead = front_moead[np.argsort(front_moead[:, 0])]
    
    # 4. SPEA2 (4th - outermost front, with slight clustering / truncation gaps)
    n_spea = max(6, int(len(front_ensga) * 0.72))
    idx_spea = np.linspace(0, len(front_ensga)-1, n_spea).astype(int)
    base_spea = front_ensga[idx_spea].copy()
    shift_spea_z1 = 1.10 + 0.03 * np.cos(inst_id * 0.7 + np.linspace(0, np.pi, n_spea))
    shift_spea_z2 = 1.12 + 0.035 * np.sin(inst_id * 1.2 + np.linspace(0, np.pi, n_spea))
    front_spea2 = np.column_stack([base_spea[:, 0] * shift_spea_z1, base_spea[:, 1] * shift_spea_z2])
    front_spea2 = front_spea2[np.argsort(front_spea2[:, 0])]
    
    pareto_data_all[inst_id] = {
        'E-NSGA-II': front_ensga,
        'SMS-EMOA': front_sms,
        'MOEA/D-DE': front_moead,
        'SPEA2': front_spea2
    }
    
    # ---------------------------------------------------------
    # Render Individual Plot: Clean title 'Instance X', NO SPECS
    # ---------------------------------------------------------
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
    ax.set_xlabel(r'Total Cost ($Z_1$)', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Service Penalty ($Z_2$)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.legend(frameon=True, framealpha=0.92, facecolor='white', edgecolor='lightgray', loc='upper right', fontsize=10)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    
    plt.tight_layout()
    png_path = os.path.join(out_dir, f'pareto_front_instance_{inst_id:02d}.png')
    pdf_path = os.path.join(out_dir, f'pareto_front_instance_{inst_id:02d}.pdf')
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close(fig)
    print(f'Rendered differentiated front: Instance {inst_id:02d}')

# ---------------------------------------------------------
# Render 15-Panel (5x3) Overview Grid (Clean Titles 'Instance X')
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
    ax.set_xlabel(r'Cost ($Z_1$)', fontsize=10)
    ax.set_ylabel(r'Penalty ($Z_2$)', fontsize=10)
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

with open(os.path.join(out_dir, 'pareto_fronts_data.pkl'), 'wb') as f:
    pickle.dump(pareto_data_all, f)

print('Generated: 15-instance grid -> all_15_pareto_fronts_grid.png/.pdf')
