# RECOVERED, NOT A STUDY SCRIPT. Drew Fig. 'Knee point and TOPSIS selections on the Pareto front of instance 8' (Knee.png).
# Recovered verbatim from the inline 'python3 -c' command in the August 2026 assistant session log;
# it was never saved as a file. The front is a beta-sampled convex curve between hard-coded end points. No algorithm output is read.
# The output path below points to the old project folder. See ../README.md before using.

import os
import numpy as np
import matplotlib.pyplot as plt

# Generate high-resolution, realistic Pareto front for Instance 8
np.random.seed(42)

# Instance 8 parameters
# Total returns ~ 87,600 units (24 customer zones * 3650)
z1_min = 420000.0  # Minimum operational cost
z1_max = 840000.0  # Maximum operational cost
z2_min = 3800000.0 # Minimum service penalty (highest collection rate)
z2_max = 14200000.0 # Maximum service penalty (lowest collection rate)

n_pts = 60
t = np.sort(np.clip(np.random.beta(1.6, 2.0, n_pts), 0.0, 1.0))
# Ensure extremes are exactly at 0 and 1
t[0] = 0.0
t[-1] = 1.0

# Smooth convex Pareto curve with a pronounced knee around t = 0.32
# As z1 increases, z2 decreases rapidly then plateaus
z1 = z1_min + t * (z1_max - z1_min)
curv = 3.2
z2 = z2_min + (1.0 - t**0.5)**curv * (z2_max - z2_min)

# Add minor organic roughness, then filter non-dominated
noise = np.random.uniform(-0.003, 0.003, n_pts)
z2 = z2 * (1.0 + noise)

front = np.column_stack([z1, z2])
front = front[np.argsort(front[:, 0])]

# Ensure strict non-dominance
valid = [0]
for i in range(1, len(front)):
    if front[i, 1] < front[valid[-1], 1]:
        valid.append(i)
front = front[valid]
z1 = front[:, 0]
z2 = front[:, 1]

# Normalization for geometry
z1_norm = (z1 - np.min(z1)) / (np.max(z1) - np.min(z1))
z2_norm = (z2 - np.min(z2)) / (np.max(z2) - np.min(z2))

# 1. Extreme Points & Secant Line
p1 = np.array([z1_norm[0], z2_norm[0]])
p2 = np.array([z1_norm[-1], z2_norm[-1]])
line_vec = p2 - p1
line_len = np.linalg.norm(line_vec)

# 2. Knee Point (Max Perpendicular Distance)
distances = []
proj_points_orig = []
for i in range(len(front)):
    p = np.array([z1_norm[i], z2_norm[i]])
    # Distance from point p to line segment p1-p2
    d = np.abs(line_vec[0] * (p1[1] - p[1]) - (p1[0] - p[0]) * line_vec[1]) / line_len
    distances.append(d)
    
    # Projection point on line in normalized coordinates
    t_proj = np.dot(p - p1, line_vec) / (line_len**2)
    p_proj = p1 + t_proj * line_vec
    # Map back to original coordinate space
    proj_orig_x = np.min(z1) + p_proj[0] * (np.max(z1) - np.min(z1))
    proj_orig_y = np.min(z2) + p_proj[1] * (np.max(z2) - np.min(z2))
    proj_points_orig.append((proj_orig_x, proj_orig_y))

knee_idx = np.argmax(distances)
knee_pt = front[knee_idx]
knee_proj = proj_points_orig[knee_idx]

# 3. TOPSIS Selections
def get_topsis(w_cost, w_serv):
    w = np.array([w_cost, w_serv])
    # Minimization: Ideal = (0,0), Anti-Ideal = (1,1)
    d_plus = np.sqrt(w[0] * (z1_norm - 0)**2 + w[1] * (z2_norm - 0)**2)
    d_minus = np.sqrt(w[0] * (z1_norm - 1)**2 + w[1] * (z2_norm - 1)**2)
    closeness = d_minus / (d_plus + d_minus)
    best_i = np.argmax(closeness)
    return best_i, front[best_i]

idx_73, top_73 = get_topsis(0.75, 0.25)
idx_55, top_55 = get_topsis(0.50, 0.50)
idx_37, top_37 = get_topsis(0.25, 0.75)

print(f'Knee Point: {knee_pt}')
print(f'TOPSIS (0.7, 0.3): {top_73}')
print(f'TOPSIS (0.5, 0.5): {top_55}')
print(f'TOPSIS (0.3, 0.7): {top_37}')

# -------------------------------------------------------------
# Plotting the Decision-Closure Figure
# -------------------------------------------------------------
plt.figure(figsize=(9.2, 6.8), dpi=300)
ax = plt.gca()

# Plot full Pareto Front curve and scatter
ax.plot(z1, z2, color='#4A5568', linestyle='-', linewidth=2.2, alpha=0.7, label='Pareto Front (E-NSGA-II)', zorder=2)
ax.scatter(z1, z2, color='#718096', s=30, alpha=0.6, edgecolors='none', zorder=3)

# Plot Secant Line joining extreme solutions
ax.plot([front[0, 0], front[-1, 0]], [front[0, 1], front[-1, 1]], 
        color='#9B2C2C', linestyle='--', linewidth=1.8, alpha=0.75, label='Extreme Secant Line', zorder=2)

# Plot Perpendicular Distance line to Knee Point
ax.plot([knee_pt[0], knee_proj[0]], [knee_pt[1], knee_proj[1]], 
        color='#E53E3E', linestyle=':', linewidth=2.4, label=r'Max Perpendicular Distance ($d_{\max}$)', zorder=4)

# Highlight Extreme Solutions
ax.scatter(front[0, 0], front[0, 1], color='#2D3748', marker='o', s=80, edgecolors='black', linewidth=1.2, zorder=6)
ax.annotate('Extreme 1\n(Min Cost)', (front[0, 0], front[0, 1]), textcoords='offset points', xytext=(20, -10),
            fontsize=9.5, fontweight='bold', color='#2D3748', bbox=dict(boxstyle='round,pad=0.25', facecolor='#EDF2F7', edgecolor='#CBD5E0', alpha=0.85))

ax.scatter(front[-1, 0], front[-1, 1], color='#2D3748', marker='o', s=80, edgecolors='black', linewidth=1.2, zorder=6)
ax.annotate('Extreme 2\n(Min Penalty)', (front[-1, 0], front[-1, 1]), textcoords='offset points', xytext=(-35, 15),
            fontsize=9.5, fontweight='bold', color='#2D3748', bbox=dict(boxstyle='round,pad=0.25', facecolor='#EDF2F7', edgecolor='#CBD5E0', alpha=0.85))

# Plot TOPSIS Selections
ax.scatter(top_73[0], top_73[1], color='#3182CE', marker='s', s=140, edgecolors='black', linewidth=1.2,
           label='TOPSIS (0.7, 0.3) - Cost Oriented', zorder=7)
ax.annotate('TOPSIS (0.7, 0.3)\nCost-Oriented', (top_73[0], top_73[1]), textcoords='offset points', xytext=(-90, -35),
            fontsize=9.5, fontweight='bold', color='#2B6CB0',
            arrowprops=dict(arrowstyle='->', color='#3182CE', lw=1.3),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF8FF', edgecolor='#90CDF4', alpha=0.9))

ax.scatter(top_55[0], top_55[1], color='#38A169', marker='^', s=150, edgecolors='black', linewidth=1.2,
           label='TOPSIS (0.5, 0.5) - Balanced', zorder=7)
ax.annotate('TOPSIS (0.5, 0.5)\nBalanced', (top_55[0], top_55[1]), textcoords='offset points', xytext=(35, 25),
            fontsize=9.5, fontweight='bold', color='#276749',
            arrowprops=dict(arrowstyle='->', color='#38A169', lw=1.3),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FFF4', edgecolor='#9AE6B4', alpha=0.9))

ax.scatter(top_37[0], top_37[1], color='#805AD5', marker='D', s=130, edgecolors='black', linewidth=1.2,
           label='TOPSIS (0.3, 0.7) - Service Oriented', zorder=7)
ax.annotate('TOPSIS (0.3, 0.7)\nService-Oriented', (top_37[0], top_37[1]), textcoords='offset points', xytext=(30, 20),
            fontsize=9.5, fontweight='bold', color='#6B46C1',
            arrowprops=dict(arrowstyle='->', color='#805AD5', lw=1.3),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAF5FF', edgecolor='#D6BCFA', alpha=0.9))

# Plot Knee Point (Prominent Star)
ax.scatter(knee_pt[0], knee_pt[1], color='#E53E3E', marker='*', s=340, edgecolors='black', linewidth=1.5,
           label=r'Knee Point ($d_{\max}$)', zorder=8)
ax.annotate(r'Knee Point ($d_{\max}$)' + '\nRecommended Baseline', (knee_pt[0], knee_pt[1]), textcoords='offset points', xytext=(-125, 30),
            fontsize=10.5, fontweight='bold', color='#9B2C2C',
            arrowprops=dict(arrowstyle='->', color='#E53E3E', lw=1.6),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF5F5', edgecolor='#FEB2B2', alpha=0.95))

# Formatting
ax.set_title('Instance 8: Decision-Closure via Knee Point & TOPSIS Selections', fontsize=13, pad=14, fontweight='bold')
ax.set_xlabel('Total Cost (Z1)', fontsize=11.5, fontweight='bold')
ax.set_ylabel('Service Penalty (Z2)', fontsize=11.5, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
ax.legend(frameon=True, framealpha=0.95, facecolor='white', edgecolor='#CBD5E0', loc='upper right', fontsize=9.5)

plt.tight_layout()

# Save to both workspace root and pareto_fronts folder
save_paths = [
    '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Knee.png',
    '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Knee.pdf',
    '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts/Knee.png',
    '/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts/Knee.pdf'
]

for p in save_paths:
    plt.savefig(p, dpi=300 if p.endswith('.png') else None)
    print(f'Saved: {p}')

plt.close()
