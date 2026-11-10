import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from study import Decoder, run_nsga2_base, select_knee_point

out_dir = '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts'
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# 1. CASE STUDY 1: DELHI NATIONAL CAPITAL REGION (NCR), INDIA
# ==============================================================================
print("Running Delhi NCR Case Study...")
I_delhi, J_delhi, K_delhi, S_delhi, R_cnt_delhi, X_delhi = 11, 4, 3, 2, 2, 1

districts_delhi = [
    'South Delhi', 'North-West Delhi', 'West Delhi', 'South-East Delhi',
    'New Delhi', 'Central Delhi', 'East Delhi', 'Shahdara',
    'North-East Delhi', 'North Delhi', 'South-West Delhi'
]

facilities_delhi = [
    'Okhla Ind. Area (South)', 'Narela Ind. Area (North)',
    'Mandoli Complex (East)', 'Patparganj Ind. Area (East)'
]

mean_R_delhi = np.array([16.0, 14.0, 12.0, 11.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]) # MT/scen
base_Ca_delhi = np.array([35.0, 30.0, 20.0, 15.0]) # MT/scen

# Road distances T_ij (in 10s of km)
T_ij_delhi = np.array([
    [1.2, 3.8, 2.5, 1.8], # South Delhi
    [3.5, 1.0, 2.8, 3.2], # North-West Delhi
    [2.4, 2.8, 3.0, 2.5], # West Delhi
    [0.8, 4.2, 2.2, 1.4], # South-East Delhi
    [1.5, 3.0, 2.0, 1.2], # New Delhi
    [2.0, 2.5, 1.8, 1.0], # Central Delhi
    [2.2, 3.2, 1.2, 0.6], # East Delhi
    [2.6, 3.0, 0.8, 1.0], # Shahdara
    [2.8, 2.6, 0.6, 1.2], # North-East Delhi
    [3.0, 1.2, 2.2, 2.4], # North Delhi
    [1.8, 3.5, 3.2, 2.4]  # South-West Delhi
])

T_jk_delhi = np.array([[1.5, 2.5, 3.0], [3.0, 1.5, 2.5], [2.0, 2.0, 1.5], [1.8, 1.8, 1.2]])
T_js_delhi = np.array([[2.0, 3.0], [2.5, 2.0], [1.8, 2.2], [1.5, 1.8]])
T_si_delhi = np.array([
    [1.2, 3.2, 2.0, 1.0, 1.2, 1.5, 2.0, 2.4, 2.6, 2.8, 1.5],
    [3.0, 1.2, 2.5, 3.8, 2.6, 2.0, 1.5, 1.2, 1.0, 1.0, 3.0]
])
T_jr_delhi = np.array([[2.0, 3.5], [3.5, 2.0], [2.2, 2.5], [1.8, 2.0]])
T_rx_delhi = np.array([[3.0], [2.5]])
d_x_delhi = np.array([5.0])
tau_i_delhi = np.array([5.0]*11)

beta_K, beta_S, beta_R = 0.50, 0.30, 0.20
alpha_delhi = 0.40
phi_1, phi_2 = 160.0, 200.0

rng = np.random.default_rng(101)
n_scen = 50
R_d = np.zeros((n_scen, I_delhi))
Ca_d = np.zeros((n_scen, J_delhi))
C_d = np.zeros((n_scen, J_delhi))
W_d = np.zeros((n_scen, J_delhi))
t_d = np.zeros((n_scen, S_delhi, I_delhi))

for s in range(n_scen):
    R_d[s] = mean_R_delhi * rng.uniform(0.85, 1.15, I_delhi)
    Ca_d[s] = base_Ca_delhi * rng.uniform(0.90, 1.10, J_delhi)
    C_d[s] = rng.uniform(15, 18, J_delhi)
    W_d[s] = rng.uniform(15, 25, J_delhi)
    t_d[s] = rng.uniform(2, 8, (S_delhi, I_delhi))

inst_delhi = {
    'I': I_delhi, 'J': J_delhi, 'K': K_delhi, 'S': S_delhi, 'R_cnt': R_cnt_delhi, 'X': X_delhi,
    'T_ij': T_ij_delhi, 'T_jk': T_jk_delhi, 'T_js': T_js_delhi, 'T_si': T_si_delhi, 'T_jr': T_jr_delhi, 'T_rx': T_rx_delhi,
    'd_x': d_x_delhi, 'tau_i': tau_i_delhi,
    'beta_K': beta_K, 'beta_S': beta_S, 'beta_R': beta_R, 'alpha': alpha_delhi,
    'phi_1': phi_1, 'phi_2': phi_2,
    'R': R_d, 'Ca': Ca_d, 'C': C_d, 'W': W_d, 't': t_d
}

front_delhi, a_delhi, rho_delhi = run_nsga2_base(inst_delhi, is_enhanced=True, seed=101)
front_delhi = front_delhi[np.argsort(front_delhi[:, 0])]

knee_delhi, knee_idx_d = select_knee_point(front_delhi)
knee_a_d = a_delhi[knee_idx_d]
knee_rho_d = rho_delhi[knee_idx_d]

# Baseline nearest assignment for Delhi
base_a_d = np.argmin(T_ij_delhi, axis=1).reshape(1, -1)
base_rho_d = np.ones((1, I_delhi))
dec_d = Decoder(inst_delhi)
C_unit_d, delta_d = dec_d.precompute_scenario_costs(R_d, Ca_d, C_d, W_d, t_d)
base_Z1_d, base_Z2_d = dec_d.decode(base_a_d, base_rho_d, R_d, Ca_d, C_unit_d, delta_d)
base_delhi = np.array([base_Z1_d[0], base_Z2_d[0]])

cost_save_delhi = (base_delhi[0] - knee_delhi[0]) / base_delhi[0] * 100.0
print(f"Delhi Baseline: Cost={base_delhi[0]:.2f}, Penalty={base_delhi[1]:.2f}")
print(f"Delhi Knee: Cost={knee_delhi[0]:.2f}, Penalty={knee_delhi[1]:.2f}, Cost Savings={cost_save_delhi:.2f}%")

# ==============================================================================
# 2. CASE STUDY 2: GREATER LONDON, UNITED KINGDOM
# ==============================================================================
print("\nRunning Greater London Case Study...")
I_london, J_london, K_london, S_london, R_cnt_london, X_london = 8, 3, 2, 2, 2, 1

districts_london = [
    'Westminster & City', 'Barnet & Enfield', 'Southwark & Lambeth',
    'Croydon & Bromley', 'Camden & Islington', 'Tower Hamlets & Hackney',
    'Kensington & Chelsea', 'Greenwich & Lewisham'
]

facilities_london = [
    'Edmonton EcoPark (North)',
    'Rainham & Dagenham Centre (East)',
    'Beddington Recovery Hub (South)'
]

mean_R_london = np.array([18.0, 14.0, 12.0, 11.0, 9.0, 6.0, 6.0, 4.0]) # MT/scen
base_Ca_london = np.array([35.0, 30.0, 25.0]) # MT/scen

T_ij_london = np.array([
    [1.8, 2.2, 1.9], # Westminster & City
    [1.0, 2.8, 3.5], # Barnet & Enfield
    [2.2, 2.0, 1.4], # Southwark & Lambeth
    [3.2, 2.4, 0.9], # Croydon & Bromley
    [1.2, 2.2, 2.6], # Camden & Islington
    [1.5, 1.4, 2.4], # Tower Hamlets & Hackney
    [2.2, 2.8, 1.8], # Kensington & Chelsea
    [2.5, 1.5, 1.6]  # Greenwich & Lewisham
])

T_jk_london = np.array([[1.8, 2.5], [1.5, 1.8], [2.2, 1.6]])
T_js_london = np.array([[1.5, 2.2], [1.8, 1.5], [2.0, 1.4]])
T_si_london = np.array([
    [1.5, 1.0, 2.0, 3.0, 1.2, 1.4, 2.0, 2.4],
    [1.8, 3.2, 1.2, 1.0, 2.4, 2.2, 1.6, 1.5]
])
T_jr_london = np.array([[2.0, 2.5], [1.5, 1.8], [2.2, 1.5]])
T_rx_london = np.array([[2.5], [2.0]])
d_x_london = np.array([6.0])
tau_i_london = np.array([5.5]*8)

beta_K_lon, beta_S_lon, beta_R_lon = 0.55, 0.25, 0.20
alpha_london = 0.35
phi_1_lon, phi_2_lon = 160.0, 200.0

rng_lon = np.random.default_rng(202)
R_l = np.zeros((n_scen, I_london))
Ca_l = np.zeros((n_scen, J_london))
C_l = np.zeros((n_scen, J_london))
W_l = np.zeros((n_scen, J_london))
t_l = np.zeros((n_scen, S_london, I_london))

for s in range(n_scen):
    R_l[s] = mean_R_london * rng_lon.uniform(0.88, 1.12, I_london)
    Ca_l[s] = base_Ca_london * rng_lon.uniform(0.92, 1.08, J_london)
    C_l[s] = rng_lon.uniform(16, 20, J_london)
    W_l[s] = rng_lon.uniform(18, 26, J_london)
    t_l[s] = rng_lon.uniform(2, 7, (S_london, I_london))

inst_london = {
    'I': I_london, 'J': J_london, 'K': K_london, 'S': S_london, 'R_cnt': R_cnt_london, 'X': X_london,
    'T_ij': T_ij_london, 'T_jk': T_jk_london, 'T_js': T_js_london, 'T_si': T_si_london, 'T_jr': T_jr_london, 'T_rx': T_rx_london,
    'd_x': d_x_london, 'tau_i': tau_i_london,
    'beta_K': beta_K_lon, 'beta_S': beta_S_lon, 'beta_R': beta_R_lon, 'alpha': alpha_london,
    'phi_1': phi_1_lon, 'phi_2': phi_2_lon,
    'R': R_l, 'Ca': Ca_l, 'C': C_l, 'W': W_l, 't': t_l
}

front_london, a_london, rho_london = run_nsga2_base(inst_london, is_enhanced=True, seed=202)
front_london = front_london[np.argsort(front_london[:, 0])]

knee_london, knee_idx_l = select_knee_point(front_london)
knee_a_l = a_london[knee_idx_l]
knee_rho_l = rho_london[knee_idx_l]

base_a_l = np.argmin(T_ij_london, axis=1).reshape(1, -1)
base_rho_l = np.ones((1, I_london))
dec_l = Decoder(inst_london)
C_unit_l, delta_l = dec_l.precompute_scenario_costs(R_l, Ca_l, C_l, W_l, t_l)
base_Z1_l, base_Z2_l = dec_l.decode(base_a_l, base_rho_l, R_l, Ca_l, C_unit_l, delta_l)
base_london = np.array([base_Z1_l[0], base_Z2_l[0]])

cost_save_london = (base_london[0] - knee_london[0]) / base_london[0] * 100.0
print(f"London Baseline: Cost={base_london[0]:.2f}, Penalty={base_london[1]:.2f}")
print(f"London Knee: Cost={knee_london[0]:.2f}, Penalty={knee_london[1]:.2f}, Cost Savings={cost_save_london:.2f}%")

# Save results dictionary
case_studies_res = {
    'delhi': {'front': front_delhi, 'knee': knee_delhi, 'baseline': base_delhi, 'a': knee_a_d, 'rho': knee_rho_d, 'districts': districts_delhi, 'facilities': facilities_delhi, 'R': mean_R_delhi},
    'london': {'front': front_london, 'knee': knee_london, 'baseline': base_london, 'a': knee_a_l, 'rho': knee_rho_l, 'districts': districts_london, 'facilities': facilities_london, 'R': mean_R_london}
}
with open(os.path.join(out_dir, 'case_studies_results.pkl'), 'wb') as f:
    pickle.dump(case_studies_res, f)

# ==============================================================================
# 3. PLOT FIGURES FOR DELHI NCR
# ==============================================================================
# Fig 1: Delhi Pareto Front
plt.figure(figsize=(8.5, 6.0), dpi=300)
ax = plt.gca()
ax.plot(front_delhi[:, 0], front_delhi[:, 1], color='#D62728', lw=2.4, label='E-NSGA-II Pareto Front', zorder=3)
ax.scatter(front_delhi[:, 0], front_delhi[:, 1], color='#E53E3E', s=35, edgecolors='black', lw=0.5, zorder=4)
ax.scatter(base_delhi[0], base_delhi[1], color='#2B6CB0', marker='X', s=180, edgecolors='black', lw=1.5, label='Status-Quo Heuristic Baseline', zorder=6)
ax.annotate('Status-Quo Baseline\n(Nearest District Assignment)', (base_delhi[0], base_delhi[1]), textcoords='offset points',
            xytext=(15, 20), fontsize=9.5, fontweight='bold', color='#1A365D',
            arrowprops=dict(arrowstyle='->', color='#2B6CB0', lw=1.4),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF8FF', edgecolor='#90CDF4', alpha=0.95))
ax.scatter(knee_delhi[0], knee_delhi[1], color='#C53030', marker='*', s=320, edgecolors='black', lw=1.5, label=r'Knee Point ($d_{\max}$)', zorder=7)
ax.annotate(r'Knee Point ($d_{\max}$)' + f'\nZ1 = {knee_delhi[0]:.2f}, Z2 = {knee_delhi[1]:.2f}', (knee_delhi[0], knee_delhi[1]),
            textcoords='offset points', xytext=(-130, 25), fontsize=10, fontweight='bold', color='#742A2A',
            arrowprops=dict(arrowstyle='->', color='#C53030', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF5F5', edgecolor='#FEB2B2', alpha=0.95))
ax.set_title('Delhi NCR E-Waste Network: Empirical Pareto Front', fontsize=13, pad=12, fontweight='bold')
ax.set_xlabel('Total Cost (Z1)', fontsize=11, fontweight='bold')
ax.set_ylabel('Service Penalty (Z2)', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, framealpha=0.95, loc='upper right', fontsize=9.5)
plt.tight_layout()
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Delhi_Pareto_Front.png', dpi=300)
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Delhi_Pareto_Front.pdf')
plt.close()

# Fig 2: Delhi Network Flow Diagram
fig, ax = plt.subplots(figsize=(12, 8.0), dpi=300)
x_dist, x_fac, x_down, x_disp = 1.0, 4.8, 8.5, 11.8
y_dist = np.linspace(8.5, 0.8, I_delhi)
y_fac = np.linspace(7.2, 1.8, J_delhi)

for i in range(I_delhi):
    ax.scatter(x_dist, y_dist[i], color='#3182CE', s=420, edgecolors='black', zorder=5)
    ax.text(x_dist - 0.25, y_dist[i], f"{districts_delhi[i]}\n({mean_R_delhi[i]} MT)", ha='right', va='center', fontsize=8.5, fontweight='bold', color='#1A365D')

for j in range(J_delhi):
    ax.scatter(x_fac, y_fac[j], color='#38A169', s=550, marker='s', edgecolors='black', zorder=5)
    ax.text(x_fac + 0.25, y_fac[j], f"{facilities_delhi[j]}\n[Cap: {base_Ca_delhi[j]} MT]", ha='left', va='center', fontsize=8.8, fontweight='bold', color='#22543D')

down_labels_d = ['Commercial Resale (50%)', 'Repair Centers (30%)', 'Material Recycling (20%)']
y_down_d = [6.0, 4.5, 3.0]
for k in range(3):
    ax.scatter(x_down, y_down_d[k], color='#DD6B20', s=500, marker='^', edgecolors='black', zorder=5)
    ax.text(x_down + 0.28, y_down_d[k], down_labels_d[k], ha='left', va='center', fontsize=9.0, fontweight='bold', color='#7B341E')

ax.scatter(x_disp, 3.0, color='#E53E3E', s=500, marker='D', edgecolors='black', zorder=5)
ax.text(x_disp + 0.28, 3.0, 'Hazardous Disposal (Landfill)\n(Secondary Waste α=40%)', ha='left', va='center', fontsize=9.0, fontweight='bold', color='#742A2A')

for i in range(I_delhi):
    assigned_j = int(knee_a_d[i])
    intensity = float(knee_rho_d[i])
    if intensity > 0.05:
        lw = max(1.2, intensity * 3.5)
        ax.annotate('', xy=(x_fac, y_fac[assigned_j]), xytext=(x_dist, y_dist[i]),
                    arrowprops=dict(arrowstyle='->', color='#2B6CB0', lw=lw, alpha=0.85, shrinkA=8, shrinkB=8))
        mid_x = (x_dist + x_fac) / 2.0 + 0.08 * (i % 3 - 1)
        mid_y = (y_dist[i] + y_fac[assigned_j]) / 2.0
        flow_vol = float(mean_R_delhi[i] * intensity)
        ax.text(mid_x, mid_y, f"{flow_vol:.1f} MT", fontsize=7.8, color='#2C5282', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#CBD5E0', alpha=0.85))

for j in range(J_delhi):
    for k in range(3):
        ax.annotate('', xy=(x_down, y_down_d[k]), xytext=(x_fac, y_fac[j]),
                    arrowprops=dict(arrowstyle='->', color='#C05621', lw=1.2, alpha=0.5, shrinkA=8, shrinkB=8, ls='--'))

ax.annotate('', xy=(x_disp, 3.0), xytext=(x_down, y_down_d[2]),
            arrowprops=dict(arrowstyle='->', color='#C53030', lw=1.8, alpha=0.8, shrinkA=8, shrinkB=8))

ax.set_xlim(-2.5, 15.5)
ax.set_ylim(0.0, 9.5)
ax.axis('off')
ax.set_title('Delhi NCR E-Waste Network: Optimal Allocation Topology (Knee Point Plan)', fontsize=13, pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Delhi_Network_Flow.png', dpi=300)
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Delhi_Network_Flow.pdf')
plt.close()

# ==============================================================================
# 4. PLOT FIGURES FOR GREATER LONDON
# ==============================================================================
# Fig 3: London Pareto Front
plt.figure(figsize=(8.5, 6.0), dpi=300)
ax = plt.gca()
ax.plot(front_london[:, 0], front_london[:, 1], color='#D62728', lw=2.4, label='E-NSGA-II Pareto Front', zorder=3)
ax.scatter(front_london[:, 0], front_london[:, 1], color='#E53E3E', s=35, edgecolors='black', lw=0.5, zorder=4)
ax.scatter(base_london[0], base_london[1], color='#2B6CB0', marker='X', s=180, edgecolors='black', lw=1.5, label='Status-Quo Heuristic Baseline', zorder=6)
ax.annotate('Status-Quo Baseline\n(Nearest Borough Assignment)', (base_london[0], base_london[1]), textcoords='offset points',
            xytext=(15, 20), fontsize=9.5, fontweight='bold', color='#1A365D',
            arrowprops=dict(arrowstyle='->', color='#2B6CB0', lw=1.4),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF8FF', edgecolor='#90CDF4', alpha=0.95))
ax.scatter(knee_london[0], knee_london[1], color='#C53030', marker='*', s=320, edgecolors='black', lw=1.5, label=r'Knee Point ($d_{\max}$)', zorder=7)
ax.annotate(r'Knee Point ($d_{\max}$)' + f'\nZ1 = {knee_london[0]:.2f}, Z2 = {knee_london[1]:.2f}', (knee_london[0], knee_london[1]),
            textcoords='offset points', xytext=(-130, 25), fontsize=10, fontweight='bold', color='#742A2A',
            arrowprops=dict(arrowstyle='->', color='#C53030', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF5F5', edgecolor='#FEB2B2', alpha=0.95))
ax.set_title('Greater London WEEE Network: Empirical Pareto Front', fontsize=13, pad=12, fontweight='bold')
ax.set_xlabel('Total Cost (Z1)', fontsize=11, fontweight='bold')
ax.set_ylabel('Service Penalty (Z2)', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, framealpha=0.95, loc='upper right', fontsize=9.5)
plt.tight_layout()
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/London_Pareto_Front.png', dpi=300)
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/London_Pareto_Front.pdf')
plt.close()

# Fig 4: London Network Flow Diagram
fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
x_dist, x_fac, x_down, x_disp = 1.0, 4.8, 8.5, 11.8
y_dist = np.linspace(8.0, 1.0, I_london)
y_fac = np.linspace(6.8, 2.2, J_london)

for i in range(I_london):
    ax.scatter(x_dist, y_dist[i], color='#3182CE', s=450, edgecolors='black', zorder=5)
    ax.text(x_dist - 0.25, y_dist[i], f"{districts_london[i]}\n({mean_R_london[i]} MT)", ha='right', va='center', fontsize=9.0, fontweight='bold', color='#1A365D')

for j in range(J_london):
    ax.scatter(x_fac, y_fac[j], color='#38A169', s=550, marker='s', edgecolors='black', zorder=5)
    ax.text(x_fac + 0.25, y_fac[j], f"{facilities_london[j]}\n[Cap: {base_Ca_london[j]} MT]", ha='left', va='center', fontsize=9.0, fontweight='bold', color='#22543D')

down_labels_l = ['Remarketing & Resale (55%)', 'Remanufacturing (25%)', 'Material Refining (20%)']
y_down_l = [6.0, 4.5, 3.0]
for k in range(3):
    ax.scatter(x_down, y_down_l[k], color='#DD6B20', s=500, marker='^', edgecolors='black', zorder=5)
    ax.text(x_down + 0.28, y_down_l[k], down_labels_l[k], ha='left', va='center', fontsize=9.0, fontweight='bold', color='#7B341E')

ax.scatter(x_disp, 3.0, color='#E53E3E', s=500, marker='D', edgecolors='black', zorder=5)
ax.text(x_disp + 0.28, 3.0, 'Controlled Disposal / Residue\n(Secondary Waste α=35%)', ha='left', va='center', fontsize=9.0, fontweight='bold', color='#742A2A')

for i in range(I_london):
    assigned_j = int(knee_a_l[i])
    intensity = float(knee_rho_l[i])
    if intensity > 0.05:
        lw = max(1.2, intensity * 3.5)
        ax.annotate('', xy=(x_fac, y_fac[assigned_j]), xytext=(x_dist, y_dist[i]),
                    arrowprops=dict(arrowstyle='->', color='#2B6CB0', lw=lw, alpha=0.85, shrinkA=8, shrinkB=8))
        mid_x = (x_dist + x_fac) / 2.0 + 0.08 * (i % 3 - 1)
        mid_y = (y_dist[i] + y_fac[assigned_j]) / 2.0
        flow_vol = float(mean_R_london[i] * intensity)
        ax.text(mid_x, mid_y, f"{flow_vol:.1f} MT", fontsize=8.2, color='#2C5282', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#CBD5E0', alpha=0.85))

for j in range(J_london):
    for k in range(3):
        ax.annotate('', xy=(x_down, y_down_l[k]), xytext=(x_fac, y_fac[j]),
                    arrowprops=dict(arrowstyle='->', color='#C05621', lw=1.2, alpha=0.5, shrinkA=8, shrinkB=8, ls='--'))

ax.annotate('', xy=(x_disp, 3.0), xytext=(x_down, y_down_l[2]),
            arrowprops=dict(arrowstyle='->', color='#C53030', lw=1.8, alpha=0.8, shrinkA=8, shrinkB=8))

ax.set_xlim(-2.5, 15.5)
ax.set_ylim(0.0, 9.0)
ax.axis('off')
ax.set_title('Greater London WEEE Network: Optimal Allocation Topology (Knee Point Plan)', fontsize=13, pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/London_Network_Flow.png', dpi=300)
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/London_Network_Flow.pdf')
plt.close()

print('ALL GENERATED CLEANLY!')
