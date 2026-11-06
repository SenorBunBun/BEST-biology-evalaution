import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# PLOTTING PARAMETERS FOR PRESENTATION QUALITY
# =============================================================================
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'Arial',
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 26,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.5,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True
})

# Data - organized by model (CVAE, Diffusion, GAN order)
models = ['CVAE', 'Diffusion', 'GAN']
# Values per model: [OSDR-612,613,352, CANCER-IMMU 10x]
osdr_values = [0, 5, 23]      # CVAE, Diffusion, GAN for OSDR
cancer_values = [6, 17, 37]   # CVAE, Diffusion, GAN for CANCER

# Convert to percentages (divide by 50)
osdr_pct = [v / 50 * 100 for v in osdr_values]
cancer_pct = [v / 50 * 100 for v in cancer_values]

# Colors (unified with shared_deg_comparison.py)
colors = {
    'CVAE': '#8E44AD',       # Purple
    'Diffusion': '#E57373',  # Light coral/red
    'GAN': '#E69F00'         # Orange
}
bar_colors = [colors['CVAE'], colors['Diffusion'], colors['GAN']]

# Set up the bar positions
x = np.arange(len(models))
width = 0.35

# Create the figure
fig, ax = plt.subplots(figsize=(14, 9))

# Create bars (grouped by model, with datasets as sub-bars)
bars1 = ax.bar(x - width/2, osdr_pct, width, label='OSDR-612,613,352',
               color=bar_colors, edgecolor='black', linewidth=2, alpha=0.9)
bars2 = ax.bar(x + width/2, cancer_pct, width, label='CANCER-IMMU 10x',
               color=bar_colors, edgecolor='black', linewidth=2, alpha=0.6, hatch='//')

# Customize the plot
ax.set_ylabel('Percentage (%)', fontsize=22, fontweight='bold')
ax.set_xlabel('Synthetic Data Model', fontsize=22, fontweight='bold')
ax.set_title('Phenotypically Predictive Genes by Model', fontsize=24, fontweight='bold', pad=25)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=20, fontweight='bold')
ax.legend(fontsize=18, loc='upper left')
ax.set_ylim(0, 85)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Style the axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Background color
ax.set_facecolor('#FAFAFA')

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='x', which='major', pad=12)

# Add value labels on bars - large and readable
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{height:.0f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=16)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{height:.0f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=16)

plt.tight_layout()
plt.savefig('phenotype_predictive_genes_bargraph.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Bar graph saved as 'phenotype_predictive_genes_bargraph.png'")
