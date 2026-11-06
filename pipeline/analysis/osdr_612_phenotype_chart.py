import matplotlib.pyplot as plt
n_genes = 50

group2 = "SpaceFlight"

group1 = "Ground Control"



colors = {
    'Diffusion': 'lightcoral',
    'CVAE': '#8E44AD',
    'GAN': '#E69F00'
}

fig, ax = plt.subplots(figsize=(10, 6))

model_names = ['Diffusion', 'CVAE', 'GAN']
counts = [5, 0, 23]
bar_colors = [colors[name] for name in model_names]

bars = ax.bar(model_names, counts, color=bar_colors, alpha=0.8,
              edgecolor='black', linewidth=1.5, width=0.5)

for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + n_genes * 0.02,
            f'{count}', ha='center', va='bottom',
            fontweight='bold', fontsize=16)


plot_title = f"Overlap of Phenotypically Predictive Genes:\n{group2} vs {group1} Classification"
ax.set_title(plot_title, fontsize=18, fontweight='bold', pad=20)

ax.set_xlabel('Synthetic Data Model', fontsize=14, fontweight='bold')
ax.set_ylabel(f'Number of Overlapping Genes (out of {n_genes})', fontsize=14, fontweight='bold')
ax.set_ylim(0, n_genes)

ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#FAFAFA')

fig_path = 'osdr_612_all_models_overlap_comparison.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Chart saved to: {fig_path}")
plt.close()