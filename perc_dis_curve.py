import matplotlib.pyplot as plt
import numpy as np

# Actual final test results
lambda_vals = ["0", "1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
psnr_gamma = [50.8418, 49.1926, 49.5239, 50.0296, 46.6558, 45.1599] 
# psnr_mu = [41.6639, 39.0656, 44.2397, 44.3443, 41.0155, 38.3986]  
niqe = [9.0211, 8.7510, 8.7118, 8.6875, 8.7060, 8.4443] 

# Custom screen-space offsets (x, y) optimized for the flipped x-axis
offsets = [
    (8, 8),     # 0
    (-8, 8),    # 1e-4: Pushed left to avoid overlap
    (8, -12),   # 5e-4: Pushed down
    (8, 8),     # 1e-3 
    (8, 8),     # 5e-3 
    (8, 8)      # 1e-2
]

# Custom horizontal alignments corresponding to offsets
alignments = ['left', 'left', 'right', 'right', 'left', 'left']

plt.figure(figsize=(10, 6))

# =========================================================
# 1. 1/x Hyperbolic Trade-off Curve & Shaded Regions
# =========================================================
x_dense = np.linspace(44.0, 52.0, 400)

# Exact 1/x parameters derived from points (51, 9), (50, 8.6), (45, 8),
# shifted down slightly (-0.45) so the curve sits cleanly beneath all data points.
a = -3.673
b = -53.57
c = 7.95
shift = 0.45 
y_dense = (c - shift) + a / (x_dense + b)

# Plot the background regions
# "Possible" region (above the curve) - Light Gray
plt.fill_between(x_dense, y_dense, 9.15, color='#E5E7E9', alpha=0.6, zorder=1)
# "Impossible" region (below the curve) - Light Red/Pink
plt.fill_between(x_dense, 8.25, y_dense, color='#FADBD8', alpha=0.6, zorder=1)

# Plot the thick 1/x boundary curve - Cornflower Blue
plt.plot(x_dense, y_dense, color='#5DADE2', linewidth=4, zorder=2)

# Add descriptive region text
plt.text(47.5, 8.8, "Possible", fontsize=16, color='#515A5A', fontweight='bold', zorder=3, alpha=0.8)
plt.text(50.8, 8.4, "Impossible", fontsize=16, color='#922B21', fontweight='bold', zorder=3, alpha=0.7)


# =========================================================
# 2. Plot the actual data points
# =========================================================
plt.scatter(psnr_gamma, niqe, color='black', s=50, zorder=4)

# Annotate each point with its corresponding lambda_gan weight
for i, txt in enumerate(lambda_vals):
    plt.annotate(
        f"λ={txt}", 
        (psnr_gamma[i], niqe[i]), 
        textcoords="offset points", 
        xytext=offsets[i],
        ha=alignments[i],
        fontsize=10,
        zorder=5
    )

plt.title("Distortion-Perception Trade-off (SphereGAN)", fontsize=14, pad=15)

# Axis labels indicating direction towards the optimal bottom-left origin
plt.xlabel(r"← PSNR$_\gamma$ (Higher is better)", fontsize=12, fontweight='bold')
plt.ylabel("← NIQE (Lower is better)", fontsize=12, fontweight='bold')
# Hardcode the axis limits to explicitly reverse the X-axis
plt.xlim(51.5, 44.5)
plt.ylim(8.25, 9.15)

# Clean up axes to match the reference illustration style
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

plt.tight_layout()
plt.savefig("pareto_frontier_conceptual.png", dpi=300)
print("Plot successfully saved as pareto_frontier_conceptual.png")
plt.show()
