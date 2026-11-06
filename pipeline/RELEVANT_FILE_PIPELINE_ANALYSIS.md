Relevant Files for Output (Once 3 Datasets Obtains)

### Config File Thoughts

ToDO: Config file, original file, which condition cols should we even keep/analyze, diffexp/rf which columns

Preset Name after preprocessing based on the original file, model generaated the same thing

Sparsity can be an argument, HVG selection, min_cells I suppose

N_top genes (for both phenotypically and diff exp)

Colors for the 3 models (set to default if not provided)

CVAE: Model Params

Diffusion: Model Params

GAN: Model Params

dataset_name

Folder path for all results


Split into training phases (each with seperate calls if needed)

--> maybe 3 scripts, preprocessing, training, outputs (or two)

training definitely seperated by the 3

### Preprocessing

1.) src/raw_data_preproecssing.ipynb to filter output to regular & normalized (if needed)
    
- min_cells=3
- Removed sparsly expressed genes (>0.997% genes empty)
- Highly variable 2000 genes

2.) Augmentation also present, but seems unlikely we do that right now

### Training

-Redo naming for results as it is kinda poor

### Generation


### OUTPUTS (ONCE ALL 3 ARE GENERATED in *.h5ad files)

1.) full_umap.py (current in cancer_data) (all 3 generated)

- Features to just display real and caching
- Minaturize if needed (if over 400,000+) [minaturize_dataset.py]

2.) scientific_metrics_comparison_v2.py (all 3 generated) --> quantative comparison plot

- Efficient caching already implemented

3.) diff_exp_analysis_cancer_all.py (serve as a base file of what modular running of 3 would look like)

-runs differential gene expression on real and then the other 3 datasets, and generates a shared report

4.) rf_classifier....., the cancer, just calls it and it generates the directories and graphs automatically --> phenotypically predictve plot

5.) Ontology Search
- Then, for now, manually take those genes, ontology_search and save results
- In tsv files, feed to gene_ontology_overlap_table.py in arguments 
- Generates xslx to put whereever

Plots:
-UMAP (provided by UMAP)
-scientific_metrics_comparison, provided by scientific metrics comparison
-shared_deg_comparison (extension of diff_exp, has to run after)
-rf_classifier generates that

Reodering required: CVAE, Diffusion, GAN (left to right)

Across Datasets:
-phenotype_predictive_genes_bargraph (ordering of CVAE, Diffusion, GAN)
