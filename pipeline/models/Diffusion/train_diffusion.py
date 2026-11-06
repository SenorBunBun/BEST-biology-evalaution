import scanpy as sc
import torch
import os

# Import the required classes (assuming the paste files are saved as modules)
from diffusion_model_cfg import ClassifierFreeDiffusionModel
from diffusion_training_utils import DiffusionTrainer, create_data_loaders
from vae_diffusion import DiffusionIntegrationVAE
import utils as diff_utils

# Load adata
print("Loading data...")
adata = sc.read_h5ad('data/raw_0.997_filtered_hvg_2000_augmented_combined.h5ad')
X_processed = diff_utils.preprocess_for_diffusion(adata.X)
adata.X = X_processed
print(adata.X)

# Setup condition config - adjust column names based on your data
condition_config = {
    'Flight' : 'categorical',
    'Strain' : 'categorical',
    'Age at Launch' : 'continuous',
    'Duration' : 'continuous'
}

# Setup VAE for latent space conversion
print("Setting up VAE...")
vae_model = DiffusionIntegrationVAE(
    n_genes=adata.n_vars,
    latent_size=128,
    hidden_dims=[1024, 1024],
    model='diffusion_model_results/diffusion_vae_norm.pt',
    verbose=True,
    auto_load=True
)

# Create data loaders
print("Creating data loaders...")
train_loader, val_loader, test_loader, processor, full_condition_config = create_data_loaders(
    adata,
    condition_config,
    vae_model=vae_model,
    batch_size=128,
    train_split=0.8,
    val_split=0.1
)

# Create diffusion model with optimized settings
print("Creating diffusion model...")
model = ClassifierFreeDiffusionModel(
    latent_dim=128,
    condition_config=full_condition_config,
    hidden_dims=[512, 512, 256, 128],
    classifier_free_prob=0.1,
    dropout=0.1
)

# Setup trainer
print("Setting up trainer...")
trainer = DiffusionTrainer(
    model=model,
    condition_processor=processor,
    vae_model=vae_model,
    num_timesteps=1000,
    beta_schedule="cosine"
)

# Train with optimized hyperparameters
print("Starting training...")
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,  # Reduced for faster training
    learning_rate=1e-4,
    weight_decay=1e-5,
    patience=15,
    save_every=5,
    save_dir="diffusion_checkpoints_retrain"
)

print("Training completed! Check 'diffusion_checkpoints' directory for saved models.")