"""
Manual Plotting: Phenotypically Predictive Genes Overlap
Presentation-quality bar chart for phenotype classification gene overlap.

Modify the variables below to customize for your dataset.
"""
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# =============================================================================
DATASET_NAME = "OSDR-612,613,352"
GROUP1 = "Ground Control"
GROUP2 = "SpaceFlight"
N_GENES = 50

# Counts in order: CVAE, Diffusion, GAN
CVAE_COUNT = 0
DIFFUSION_COUNT = 5
GAN_COUNT = 23

# =============================================================================
# PLOTTING PARAMETERS FOR PRESENTATION QUALITY
# =============================================================================
plt.rcParams.update({
    'font.size': 22,
    'font.family': 'Arial',
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 30,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.5,
})

# Colors (unified across all analysis plots)
colors = {
    'CVAE': '#8E44AD',       # Purple
    'Diffusion': '#E57373',  # Light coral/red
    'GAN': '#E69F00'         # Orange
}

# Model order: CVAE, Diffusion, GAN
model_names = ['CVAE', 'Diffusion', 'GAN']
counts = [CVAE_COUNT, DIFFUSION_COUNT, GAN_COUNT]
bar_colors = [colors[name] for name in model_names]

# Create figure
fig, ax = plt.subplots(figsize=(14, 9))

# Create bars
bars = ax.bar(model_names, counts, color=bar_colors, alpha=0.9,
              edgecolor='black', linewidth=2, width=0.5)

# Add value labels on top of bars - large and readable
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + N_GENES * 0.02,
            f'{count}', ha='center', va='bottom',
            fontweight='bold', fontsize=26)

# Title and labels
plot_title = f"Overlap of Phenotypically Predictive Genes\n{GROUP2} vs {GROUP1} Classification ({DATASET_NAME})"
ax.set_title(plot_title, fontsize=28, fontweight='bold', pad=25)
ax.set_xlabel('Synthetic Data Model', fontsize=26, fontweight='bold')
ax.set_ylabel(f'Number of Overlapping Genes (out of {N_GENES})', fontsize=26, fontweight='bold')
ax.set_ylim(0, N_GENES)

# Styling
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#FAFAFA')
ax.tick_params(axis='both', which='major', labelsize=22)
ax.tick_params(axis='x', which='major', pad=12, labelsize=24)

plt.tight_layout()

# Save to outputs directory
output_dir = Path(__file__).parent.parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
fig_path = output_dir / f'phenotype_overlap_{DATASET_NAME.lower().replace(",", "_").replace(" ", "_")}.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Chart saved to: {fig_path}")
plt.show()
