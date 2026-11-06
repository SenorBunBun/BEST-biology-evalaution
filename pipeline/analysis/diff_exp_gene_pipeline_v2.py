#!/usr/bin/env python3
"""
Differential Gene Expression Cross-AnnData Comparison Script

This script performs differential gene expression analysis between two groups in a specified column
(e.g., Flight: 0.0 vs 1.0 for spaceflight data, or tissue: metastasis normal vs metastasis tumor for cancer data)
in two separate AnnData objects, identifies the top DEGs from each, and finds overlaps.

Default behavior preserves the original spaceflight analysis (Flight column, groups 0.0 vs 1.0).
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import sparse

warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 1  # verbosity level
sc.settings.set_figure_params(dpi=80, facecolor='white')

class DEGCrossComparison:
    """Class to handle differential gene expression comparison between two AnnData objects."""

    def __init__(self, adata1, adata2, name1="Dataset_1", name2="Dataset_2", output_dir="deg_results",
                 column_name="Flight", group1="0.0", group2="1.0",
                 group1_display=None, group2_display=None, show_plots=False):
        """
        Initialize the comparison with two AnnData objects.

        Parameters:
        -----------
        adata1, adata2 : AnnData
            AnnData objects to compare
        name1, name2 : str
            Custom names for the datasets (used in plots and outputs)
        output_dir : str
            Directory to save results
        column_name : str
            Name of the column in obs to use for grouping (default: "Flight" for spaceflight data)
        group1 : str
            Reference group value (default: "0.0" for ground control in spaceflight data)
        group2 : str
            Target group value to compare against reference (default: "1.0" for flight in spaceflight data)
        group1_display : str
            Display name for group1 in plots (default: same as group1)
        group2_display : str
            Display name for group2 in plots (default: same as group2)
        show_plots : bool
            Whether to display plots with plt.show() (default: False)
        """
        self.adata1 = adata1.copy()
        self.adata2 = adata2.copy()
        self.name1 = name1
        self.name2 = name2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Store column and group information
        self.column_name = column_name
        self.group1 = group1  # reference group
        self.group2 = group2  # target group
        self.group1_display = group1_display if group1_display else group1
        self.group2_display = group2_display if group2_display else group2
        self.show_plots = show_plots

        # Validate inputs
        self._validate_data()

        # Results storage
        self.results = {}
        
    def _validate_data(self):
        """Validate that both AnnData objects have the required column with specified groups."""
        for i, adata in enumerate([self.adata1, self.adata2], 1):
            if self.column_name not in adata.obs.columns:
                raise ValueError(f"AnnData {i} does not have '{self.column_name}' column in obs")

            # Get unique values and handle different data types
            column_values = set(adata.obs[self.column_name].unique())

            # Convert values to strings for comparison, handling NaN values
            column_values_str = set()
            for val in column_values:
                if pd.isna(val):
                    continue  # Skip NaN values
                column_values_str.add(str(val))

            # Check if the specified groups exist in the data
            if str(self.group1) not in column_values_str:
                raise ValueError(f"AnnData {i} '{self.column_name}' column does not contain group '{self.group1}'. Found: {column_values_str}")
            if str(self.group2) not in column_values_str:
                raise ValueError(f"AnnData {i} '{self.column_name}' column does not contain group '{self.group2}'. Found: {column_values_str}")

            # Convert to string for consistency
            adata.obs[self.column_name] = adata.obs[self.column_name].astype(str)

        print("✓ Data validation passed")
        print(f"{self.name1}: {self.adata1.n_obs} cells, {self.adata1.n_vars} genes")
        print(f"{self.name2}: {self.adata2.n_obs} cells, {self.adata2.n_vars} genes")
        print(f"Comparing '{self.group2}' vs '{self.group1}' in column '{self.column_name}'")
        
    def _prepare_adata_for_deg(self, adata, name):
        """Prepare AnnData for DEG analysis."""
        print(f"\nPreparing {name} for DEG analysis...")

        # Check if raw data exists, if not use X
        if adata.raw is None:
            print(f"No raw data found in {name}, using current X matrix")
        else:
            print(f"Using raw data from {name}")

        # Print group distribution
        group_counts = adata.obs[self.column_name].value_counts()
        print(f"{name} {self.column_name} group distribution:")
        for group, count in group_counts.items():
            print(f"  Group {group}: {count} cells")

        return adata
    
    def perform_deg_analysis(self, adata, name, method='wilcoxon', n_genes=100):
        """
        Perform differential gene expression analysis.
        
        Parameters:
        -----------
        adata : AnnData
            AnnData object to analyze
        name : str
            Name identifier for the dataset
        method : str
            Method for DEG analysis ('wilcoxon', 't-test', 'logreg')
        n_genes : int
            Number of top genes to return
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with DEG results
        """
        print(f"\nPerforming DEG analysis for {name}...")
        
        # Prepare data
        adata_prep = self._prepare_adata_for_deg(adata, name)

        # Use the specified groups
        reference_group = str(self.group1)
        target_group = str(self.group2)

        print(f"Comparing group '{target_group}' vs reference group '{reference_group}'")

        adata.X = sparse.csr_matrix(adata.X)
        print(adata.X)

        # Perform differential expression analysis
        sc.tl.rank_genes_groups(
            adata_prep,
            self.column_name,
            method=method,
            reference=reference_group,  # Compare target vs reference group
            use_raw=False
        )

        print(adata_prep.uns['rank_genes_groups']['pvals_adj'])

        # Extract results for the target group (compared to reference group)
        result_df = sc.get.rank_genes_groups_df(adata_prep, group=target_group)
        
        # Add additional statistics
        result_df['abs_logfoldchanges'] = np.abs(result_df['logfoldchanges'])
        result_df['dataset'] = name
        
        # Sort by adjusted p-value and log fold change
        result_df = result_df.sort_values(['pvals_adj', 'abs_logfoldchanges'], 
                                        ascending=[True, False])
        
        print(f"✓ DEG analysis completed for {name}")
        print(f"  Found {len(result_df)} differentially expressed genes")
        print(f"  Top gene: {result_df.iloc[0]['names']} (adj p-val: {result_df.iloc[0]['pvals_adj']:.2e})")
        
        return result_df.head(n_genes)
    
    def find_overlaps(self, deg_results1, deg_results2, top_n=100):
        """
        Find overlapping genes between two DEG result sets.
        
        Parameters:
        -----------
        deg_results1, deg_results2 : pd.DataFrame
            DEG results from the two datasets
        top_n : int
            Number of top genes to consider for overlap
            
        Returns:
        --------
        dict
            Dictionary containing overlap analysis results
        """
        print(f"\nAnalyzing overlaps between top {top_n} DEGs...")
        
        # Get top N genes from each dataset
        genes1 = set(deg_results1.head(top_n)['names'])
        genes2 = set(deg_results2.head(top_n)['names'])
        
        # Calculate overlaps
        overlap_genes = genes1.intersection(genes2)
        unique_to_1 = genes1 - genes2
        unique_to_2 = genes2 - genes1
        
        # Calculate statistics
        overlap_count = len(overlap_genes)
        jaccard_index = overlap_count / len(genes1.union(genes2))
        overlap_percentage_1 = (overlap_count / len(genes1)) * 100
        overlap_percentage_2 = (overlap_count / len(genes2)) * 100
        
        overlap_results = {
            'top_n': top_n,
            'dataset1_genes': len(genes1),
            'dataset2_genes': len(genes2),
            'overlap_count': overlap_count,
            'overlap_genes': sorted(list(overlap_genes)),
            'unique_to_dataset1': sorted(list(unique_to_1)),
            'unique_to_dataset2': sorted(list(unique_to_2)),
            'jaccard_index': jaccard_index,
            'overlap_percentage_dataset1': overlap_percentage_1,
            'overlap_percentage_dataset2': overlap_percentage_2
        }
        
        return overlap_results
    
    def create_overlap_comparison_df(self, deg_results1, deg_results2, overlap_genes):
        """Create a comparison DataFrame for overlapping genes."""
        # Get overlapping gene data from both datasets
        overlap_data1 = deg_results1[deg_results1['names'].isin(overlap_genes)].copy()
        overlap_data2 = deg_results2[deg_results2['names'].isin(overlap_genes)].copy()
        
        # Merge the data with custom dataset names
        comparison_df = overlap_data1[['names', 'logfoldchanges', 'pvals_adj']].merge(
            overlap_data2[['names', 'logfoldchanges', 'pvals_adj']], 
            on='names', 
            suffixes=(f'_{self.name1}', f'_{self.name2}')
        )
        
        # Calculate correlation of log fold changes
        lfc_correlation = comparison_df[f'logfoldchanges_{self.name1}'].corr(
            comparison_df[f'logfoldchanges_{self.name2}']
        )
        
        return comparison_df, lfc_correlation
    
    def plot_results(self, deg_results1, deg_results2, overlap_results):
        """Create visualization plots for the DEG comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Volcano plots for both datasets
        dataset_info = [(deg_results1, self.name1), (deg_results2, self.name2)]
        for i, (deg_results, name) in enumerate(dataset_info):
            ax = axes[0, i]
            
            # Create volcano plot
            x = deg_results['logfoldchanges']
            y = -np.log10(deg_results['pvals_adj'])
            
            scatter = ax.scatter(x, y, c=y, cmap='viridis', alpha=0.6, s=20)
            
            # Highlight significant genes
            significant = deg_results['pvals_adj'] < 0.05
            ax.scatter(x[significant], y[significant], color='red', alpha=0.8, s=30)
            
            ax.set_xlabel('Log Fold Change')
            ax.set_ylabel('-log10(Adjusted P-value)')
            ax.set_title(f'Volcano Plot - {name}')
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line for significance threshold
            ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 3: Overlap visualization
        ax = axes[1, 0]
        
        # Create Venn diagram-like bar plot
        categories = [f'{self.name1}\nOnly', 'Overlap', f'{self.name2}\nOnly']
        values = [
            len(overlap_results['unique_to_dataset1']),
            overlap_results['overlap_count'],
            len(overlap_results['unique_to_dataset2'])
        ]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black')
        ax.set_ylabel('Number of Genes')
        ax.set_title('Gene Overlap Analysis')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value}', ha='center', va='bottom')
        
        # Plot 4: Correlation of log fold changes for overlapping genes
        ax = axes[1, 1]
        
        if len(overlap_results['overlap_genes']) > 1:
            comparison_df, correlation = self.create_overlap_comparison_df(
                deg_results1, deg_results2, overlap_results['overlap_genes']
            )
            
            ax.scatter(comparison_df[f'logfoldchanges_{self.name1}'], 
                      comparison_df[f'logfoldchanges_{self.name2}'],
                      alpha=0.7, s=50)
            
            # Add correlation line
            x_range = ax.get_xlim()
            ax.plot(x_range, x_range, 'r--', alpha=0.5, label='y=x')
            
            ax.set_xlabel(f'Log Fold Change - {self.name1}')
            ax.set_ylabel(f'Log Fold Change - {self.name2}')
            ax.set_title(f'LFC Correlation (r={correlation:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient overlapping\ngenes for correlation', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('LFC Correlation - Not Available')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'deg_comparison_plots.png', dpi=300, bbox_inches='tight')
        if self.show_plots:
            plt.show()
        plt.close()
    
    def save_results(self, deg_results1, deg_results2, overlap_results):
        """Save all results to files."""
        print(f"\nSaving results to {self.output_dir}...")
        
        # Save DEG results with custom names
        deg_results1.to_csv(self.output_dir / f'deg_results_{self.name1}.csv', index=False)
        deg_results2.to_csv(self.output_dir / f'deg_results_{self.name2}.csv', index=False)
        
        # Save overlap results
        overlap_df = pd.DataFrame([
            {'Metric': 'Top N genes analyzed', 'Value': overlap_results['top_n']},
            {'Metric': f'{self.name1} DEGs', 'Value': overlap_results['dataset1_genes']},
            {'Metric': f'{self.name2} DEGs', 'Value': overlap_results['dataset2_genes']},
            {'Metric': 'Overlapping genes', 'Value': overlap_results['overlap_count']},
            {'Metric': 'Jaccard Index', 'Value': f"{overlap_results['jaccard_index']:.4f}"},
            {'Metric': f'Overlap % ({self.name1})', 'Value': f"{overlap_results['overlap_percentage_dataset1']:.2f}%"},
            {'Metric': f'Overlap % ({self.name2})', 'Value': f"{overlap_results['overlap_percentage_dataset2']:.2f}%"}
        ])
        overlap_df.to_csv(self.output_dir / 'overlap_summary.csv', index=False)
        
        # Save gene lists
        pd.DataFrame({'Gene': overlap_results['overlap_genes']}).to_csv(
            self.output_dir / 'overlapping_genes.csv', index=False)
        pd.DataFrame({'Gene': overlap_results['unique_to_dataset1']}).to_csv(
            self.output_dir / f'unique_to_{self.name1}.csv', index=False)
        pd.DataFrame({'Gene': overlap_results['unique_to_dataset2']}).to_csv(
            self.output_dir / f'unique_to_{self.name2}.csv', index=False)
        
        # Save detailed comparison for overlapping genes
        if len(overlap_results['overlap_genes']) > 0:
            comparison_df, correlation = self.create_overlap_comparison_df(
                deg_results1, deg_results2, overlap_results['overlap_genes']
            )
            comparison_df.to_csv(self.output_dir / 'overlapping_genes_comparison.csv', index=False)
        
        print("✓ All results saved successfully")
    
    def print_summary(self, overlap_results):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print("DIFFERENTIAL GENE EXPRESSION CROSS-ADATA COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n📊 ANALYSIS OVERVIEW:")
        print(f"   • Top {overlap_results['top_n']} DEGs analyzed from each dataset")
        print(f"   • {self.name1}: {overlap_results['dataset1_genes']} DEGs")
        print(f"   • {self.name2}: {overlap_results['dataset2_genes']} DEGs")
        
        print(f"\n🔄 OVERLAP RESULTS:")
        print(f"   • Overlapping genes: {overlap_results['overlap_count']}")
        print(f"   • Jaccard Index: {overlap_results['jaccard_index']:.4f}")
        print(f"   • Overlap percentage ({self.name1}): {overlap_results['overlap_percentage_dataset1']:.2f}%")
        print(f"   • Overlap percentage ({self.name2}): {overlap_results['overlap_percentage_dataset2']:.2f}%")
        
        if overlap_results['overlap_count'] > 0:
            print(f"\n🧬 TOP 10 OVERLAPPING GENES:")
            for i, gene in enumerate(overlap_results['overlap_genes'][:10], 1):
                print(f"   {i:2d}. {gene}")
            if len(overlap_results['overlap_genes']) > 10:
                print(f"   ... and {len(overlap_results['overlap_genes']) - 10} more")
        
        print(f"\n📈 INTERPRETATION:")
        jaccard = overlap_results['jaccard_index']
        if jaccard > 0.5:
            interpretation = "High similarity"
        elif jaccard > 0.3:
            interpretation = "Moderate similarity"
        elif jaccard > 0.1:
            interpretation = "Low similarity"
        else:
            interpretation = "Very low similarity"
        
        print(f"   • {interpretation} between datasets (Jaccard = {jaccard:.4f})")
        print(f"   • Unique to {self.name1}: {len(overlap_results['unique_to_dataset1'])} genes")
        print(f"   • Unique to {self.name2}: {len(overlap_results['unique_to_dataset2'])} genes")
        
        print("\n" + "="*80)
    
    def run_analysis(self, method='wilcoxon', n_genes=100):
        """
        Run the complete DEG comparison analysis.
        
        Parameters:
        -----------
        method : str
            Method for DEG analysis
        n_genes : int
            Number of top genes to analyze
        """
        print("Starting DEG Cross-AnnData Comparison Analysis...")
        print(f"Method: {method}, Top N genes: {n_genes}")
        
        # Perform DEG analysis on both datasets
        deg_results1 = self.perform_deg_analysis(self.adata1, self.name1, method, n_genes)
        deg_results2 = self.perform_deg_analysis(self.adata2, self.name2, method, n_genes)
        
        # Find overlaps
        overlap_results = self.find_overlaps(deg_results1, deg_results2, n_genes)
        
        # Create visualizations
        self.plot_results(deg_results1, deg_results2, overlap_results)
        
        # Save results
        self.save_results(deg_results1, deg_results2, overlap_results)
        
        # Print summary
        self.print_summary(overlap_results)
        
        # Store results in the object
        self.results = {
            'deg_results1': deg_results1,
            'deg_results2': deg_results2,
            'overlap_results': overlap_results
        }
        
        return self.results

def load_example_data():
    """Load example data for testing (if available)."""
    try:
        # Try to load example datasets
        adata1 = sc.datasets.pbmc68k_reduced()
        adata2 = sc.datasets.pbmc68k_reduced()
        
        # Add fake Flight column for demonstration using 0.0/1.0 format
        np.random.seed(42)
        adata1.obs['Flight'] = np.random.choice([0.0, 1.0], size=adata1.n_obs, p=[0.6, 0.4])
        adata2.obs['Flight'] = np.random.choice([0.0, 1.0], size=adata2.n_obs, p=[0.5, 0.5])
        
        return adata1, adata2
    except:
        return None, None

# Main execution function
def main(adata1=None, adata2=None, name1="Dataset_1", name2="Dataset_2", output_dir="deg_results",
         method='wilcoxon', n_genes=100, column_name="Flight", group1="0.0", group2="1.0",
         group1_display=None, group2_display=None, show_plots=False):
    """
    Main function to run the DEG comparison analysis.

    Parameters:
    -----------
    adata1, adata2 : AnnData objects
        The two datasets to compare
    name1, name2 : str
        Custom names for the datasets (used in plots and outputs)
    output_dir : str
        Directory to save results
    method : str
        DEG analysis method
    n_genes : int
        Number of top genes to analyze
    column_name : str
        Name of the column in obs to use for grouping (default: "Flight" for spaceflight data)
    group1 : str
        Reference group value (default: "0.0" for ground control in spaceflight data)
    group2 : str
        Target group value to compare against reference (default: "1.0" for flight in spaceflight data)
    group1_display : str
        Display name for group1 in plots (default: same as group1)
    group2_display : str
        Display name for group2 in plots (default: same as group2)
    show_plots : bool
        Whether to display plots with plt.show() (default: False)
    """
    
    # Load example data if not provided
    if adata1 is None or adata2 is None:
        print("No AnnData objects provided. Loading example data...")
        adata1, adata2 = load_example_data()
        if adata1 is None:
            print("❌ Could not load example data. Please provide AnnData objects.")
            return None
        # Use default names for example data if not specified
        if name1 == "Dataset_1" and name2 == "Dataset_2":
            name1, name2 = "Example_PBMC_1", "Example_PBMC_2"
    
    # Initialize and run analysis
    try:
        analyzer = DEGCrossComparison(adata1, adata2, name1, name2, output_dir,
                                      column_name=column_name, group1=group1, group2=group2,
                                      group1_display=group1_display, group2_display=group2_display,
                                      show_plots=show_plots)
        results = analyzer.run_analysis(method=method, n_genes=n_genes)
        return analyzer
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    print("DEG Cross-AnnData Comparison Script")
    print("===================================")
    
    # You can replace these with your actual AnnData objects
    # adata1 = sc.read_h5ad('path_to_your_first_dataset.h5ad')
    # adata2 = sc.read_h5ad('path_to_your_second_dataset.h5ad')
    
    # Run with example data
    analyzer = main(
        adata1=None,  # Replace with your first AnnData object
        adata2=None,  # Replace with your second AnnData object
        name1="Control_Condition",    # Replace with your first dataset name
        name2="Treatment_Condition",  # Replace with your second dataset name
        output_dir="deg_comparison_results",
        method='wilcoxon',
        n_genes=100
    )
    
    if analyzer:
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {analyzer.output_dir}")