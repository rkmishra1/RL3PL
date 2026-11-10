import numpy as np
import matplotlib.pyplot as plt

I, J = 8, 4
districts = ['Bengaluru Urban', 'Bengaluru Rural', 'Ramanagara', 'Tumakuru', 'Mysuru', 'Mandya', 'Kolar', 'Dharwad']
mean_R = np.array([30.0, 10.0, 5.0, 8.0, 6.0, 3.0, 4.0, 5.0])
knee_a = [1, 1, 2, 0, 2, 2, 1, 3]
knee_rho = [0.8238, 0.0, 1.0, 0.9344, 0.9715, 1.0, 1.0, 0.0]

fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
x_dist = 1.0
x_fac = 4.5
x_down = 8.0
x_disp = 11.2

y_dist = np.linspace(8.0, 1.0, I)
y_fac = np.linspace(7.0, 2.0, J)

# Draw Districts
for i in range(I):
    ax.scatter(x_dist, y_dist[i], color='#3182CE', s=450, edgecolors='black', zorder=5)
    ax.text(x_dist - 0.25, y_dist[i], f"{districts[i]}\n({mean_R[i]} MT/scen)", ha='right', va='center', fontsize=9.0, fontweight='bold', color='#1A365D')

# Draw Facilities
fac_labels = ['E-Parisaraa (Dabaspet)\n[Cap: 30 MT]', 'Sriram Eco (Bommasandra)\n[Cap: 24 MT]', 'Eco Globe (Bidadi)\n[Cap: 10 MT]', 'MKK Recyclers (Dharwad)\n[Cap: 6 MT]']
for j in range(J):
    ax.scatter(x_fac, y_fac[j], color='#38A169', s=550, marker='s', edgecolors='black', zorder=5)
    ax.text(x_fac + 0.25, y_fac[j], fac_labels[j], ha='left', va='center', fontsize=9.0, fontweight='bold', color='#22543D')

# Draw Downstream Channels
down_labels = ['Commercial Resale (50%)\n(Retailers)', 'Repair Centers (30%)\n(Service & Return)', 'Recycling Plants (20%)\n(Material Extraction)']
y_down = [6.0, 4.5, 3.0]
for k in range(3):
    ax.scatter(x_down, y_down[k], color='#DD6B20', s=500, marker='^', edgecolors='black', zorder=5)
    ax.text(x_down + 0.28, y_down[k], down_labels[k], ha='left', va='center', fontsize=9.0, fontweight='bold', color='#7B341E')

# Draw Landfill
ax.scatter(x_disp, 3.0, color='#E53E3E', s=500, marker='D', edgecolors='black', zorder=5)
ax.text(x_disp + 0.28, 3.0, 'Hazardous Disposal / Landfill\n(Secondary Waste α=40%)', ha='left', va='center', fontsize=9.0, fontweight='bold', color='#742A2A')

# Draw Optimal Flow Connections
for i in range(I):
    assigned_j = int(knee_a[i])
    intensity = float(knee_rho[i])
    if intensity > 0.05:
        lw = max(1.2, intensity * 3.5)
        ax.annotate('', xy=(x_fac, y_fac[assigned_j]), xytext=(x_dist, y_dist[i]),
                    arrowprops=dict(arrowstyle='->', color='#2B6CB0', lw=lw, alpha=0.85, shrinkA=8, shrinkB=8))
        mid_x = (x_dist + x_fac) / 2.0 + 0.1 * (i % 3 - 1)
        mid_y = (y_dist[i] + y_fac[assigned_j]) / 2.0
        flow_vol = float(mean_R[i] * intensity)
        ax.text(mid_x, mid_y, f"{flow_vol:.1f} MT", fontsize=8.2, color='#2C5282', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#CBD5E0', alpha=0.85))

# Draw Downstream Echelon Flows
for j in range(J):
    for k in range(3):
        ax.annotate('', xy=(x_down, y_down[k]), xytext=(x_fac, y_fac[j]),
                    arrowprops=dict(arrowstyle='->', color='#C05621', lw=1.2, alpha=0.5, shrinkA=8, shrinkB=8, ls='--'))

# Draw Recycling to Landfill Flow
ax.annotate('', xy=(x_disp, 3.0), xytext=(x_down, y_down[2]),
            arrowprops=dict(arrowstyle='->', color='#C53030', lw=1.8, alpha=0.8, shrinkA=8, shrinkB=8))

ax.set_xlim(-2.2, 14.8)
ax.set_ylim(0.0, 9.0)
ax.axis('off')
ax.set_title('Karnataka E-Waste Case Study: Optimal Network Flow Topology (Knee Point Plan)', fontsize=12.5, pad=15, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Karnataka_Network_Flow.png', dpi=300)
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/Karnataka_Network_Flow.pdf')
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts/Karnataka_Network_Flow.png', dpi=300)
plt.savefig('/Users/ramakrushnamishra/Downloads/Reverse_Logistix_2/pareto_fronts/Karnataka_Network_Flow.pdf')
plt.close()
print('DONE: Successfully generated Karnataka_Network_Flow.png and .pdf')
