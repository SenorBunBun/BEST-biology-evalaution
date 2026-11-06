import torch
import numpy as np
import src.utils as utils
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from src.unbatch_vanilla_cvae_relu_qol_categorical import (
    Enhanced_Vanilla_CVAE_Categorical,
    create_dataloader_with_encoders
)
import os
import time


def main():
    """
    Enhanced training script for CVAE with comprehensive monitoring
    and proper categorical condition handling

    Key features:
    - Fits label encoders on FULL dataset before train/val split
    - Ensures consistent encoding across all data splits
    - Saves encoders alongside model for inference
    """
    print("Enhanced CVAE Training Pipeline (with Categorical Handling)", flush=True)
    print("=" * 60, flush=True)

    # =============================================================================
    # CONFIGURATION
    # =============================================================================

    # Model hyperparameters
    #CONFIG = {
    #    'n_genes': 2000,
    #    'latent_size': 32,
    #    'beta': 0.01, #0.05 originally, 0.001 if not doing any annealing
    #    'dropout': 0.2, #originally 0.03
    #    'lr': 0.001, #originally 0.01
    #    'weight_decay': 0.0001,  #originally 0.001
    #    'batch_size': 128,
    #    'n_epochs': 130, #50 origianlly
    #    'patience': 25, #15 originally
    #    'overfitting_patience': 8,
    #    'min_delta': 1e-4,
    #    'val_split': 0.3,
    #    'random_state': 42,
    #    'verbose': True
    #}

    # V6 Config
    # V6 Config
    # CONFIG = {
    #    'n_genes': 2000,
    #    'latent_size': 128,
    #    'beta': 0.05, #0.05 originally, 0.001 if not doing any annealing
    #    'dropout': 0.2, #originally 0.03
    #    'lr': 0.001, #originally 0.01
    #    'weight_decay': 0.0001,  #originally 0.001
    #    'batch_size': 128,
    #    'n_epochs': 120, #50 origianlly
    #    'patience': 15, #15 originally
    #    'overfitting_patience': 8,
    #    'min_delta': 1e-4,
    #    'val_split': 0.3,
    #    'random_state': 42,
    #    'verbose': True
    #}

    CONFIG = {
        'n_genes': 2000,
        'latent_size': 32,
        'beta': 0.05, #0.05 originally, 0.001 if not doing any annealing
        'dropout': 0.03, #originally 0.03
        'lr': 0.01, #originally 0.01
        'weight_decay': 0.001,  #originally 0.001
        'batch_size': 128,
        'n_epochs': 50, #50 origianlly
        'patience': 15, #15 originally
        'overfitting_patience': 10,
        'min_delta': 1e-4,
        'val_split': 0.3,
        'random_state': 42,
        'verbose': True
    }

    # File paths

    PATHS = {
        'data': 'cancer_data/cancer_0.997_filtered_hvg_2000_norm.h5ad',
        'model': 'trained_models_cancer/vanilla_cvae_relu_raw_filtered_0.997_800_400_32.pt',
        'metrics': 'trained_models_cancer/vanilla_cvae_relu_raw_filtered_0.997_800_400_32.json',
        'plot': 'trained_models_cancer/vanilla_cvae_relu_raw_filtered_0.997_800_400_32.png'
    }

    # Condition columns for CVAE
    condition_cols = ["annotation_major", "tissue"]

    print(f"Configuration:", flush=True)
    print(f"   - Genes: {CONFIG['n_genes']}", flush=True)
    print(f"   - Latent size: {CONFIG['latent_size']}", flush=True)
    print(f"   - Conditions: {len(condition_cols)} ({', '.join(condition_cols)})", flush=True)
    print(f"   - Batch size: {CONFIG['batch_size']}", flush=True)
    print(f"   - Max epochs: {CONFIG['n_epochs']}", flush=True)
    print(f"   - Learning rate: {CONFIG['lr']}", flush=True)
    print(f"   - Beta (KL weight): {CONFIG['beta']}", flush=True)
    print(flush=True)

    # =============================================================================
    # DATA LOADING AND PREPROCESSING
    # =============================================================================

    print("Loading and preprocessing data...", flush=True)
    try:
        adata = sc.read_h5ad(PATHS['data'])
        print(f"[OK] Loaded data: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
        print(adata.X, flush=True)

        # CRITICAL FIX: Convert CSC to CSR for fast row access during training
        if sparse.issparse(adata.X):
            print(f"[INFO] Data is sparse, format: {adata.X.format}", flush=True)
            if adata.X.format == 'csc':
                print(f"[FIXING] Converting CSC to CSR for efficient row access...", flush=True)
                adata.X = adata.X.tocsr()
                print(f"[OK] Converted to CSR format", flush=True)
        else:
            print(f"[INFO] Data is dense (numpy array)", flush=True)

        # Check if we have the expected number of genes
        if adata.n_vars != CONFIG['n_genes']:
            print(f"[WARN] Expected {CONFIG['n_genes']} genes, got {adata.n_vars}", flush=True)
            CONFIG['n_genes'] = adata.n_vars
            print(f"   Updated n_genes to {CONFIG['n_genes']}", flush=True)

        # Check condition columns
        missing_cols = [col for col in condition_cols if col not in adata.obs.columns]
        if missing_cols:
            print(f"[ERROR] Missing condition columns: {missing_cols}", flush=True)
            print(f"   Available columns: {list(adata.obs.columns)}", flush=True)
            return

        print(f"[OK] All condition columns found: {condition_cols}", flush=True)

    except Exception as e:
        print(f"[ERROR] Error loading data: {e}", flush=True)
        return

    # =============================================================================
    # MODEL INITIALIZATION (BEFORE SPLITTING - need to fit encoders first)
    # =============================================================================

    print(f"\nInitializing Enhanced CVAE with categorical handling...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:

        # FIT LABEL ENCODERS ON FULL DATASET BEFORE SPLITTING
        # This is critical for consistent encoding!
        print(f"\n[IMPORTANT] Fitting label encoders on full dataset...", flush=True)
        cvae.fit_label_encoders(adata, condition_cols)

        # Count parameters
        total_params = sum(p.numel() for p in cvae.parameters())
        trainable_params = sum(p.numel() for p in cvae.parameters() if p.requires_grad)

        print(f"\n[OK] Model initialized successfully:", flush=True)
        print(f"   - Total parameters: {total_params:,}", flush=True)
        print(f"   - Trainable parameters: {trainable_params:,}", flush=True)
        print(f"   - Model size: ~{total_params * 4 / 1e6:.2f} MB", flush=True)

    except Exception as e:
        print(f"[ERROR] Error initializing model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # =============================================================================
    # DATA SPLITTING AND DATALOADER CREATION
    # =============================================================================

    print(f"\nCreating data splits and loaders...", flush=True)
    try:
        # Split data AFTER fitting encoders
        train_adata, val_adata = utils.split_adata(
            adata,
            condition_cols,
            val_split=CONFIG['val_split'],
            random_state=CONFIG['random_state']
        )

        print(f"[OK] Data split completed:", flush=True)
        print(f"   - Training: {train_adata.n_obs} cells", flush=True)
        print(f"   - Validation: {val_adata.n_obs} cells", flush=True)

        # Create data loaders using pre-fitted encoders
        print(f"\nCreating dataloaders with consistent encoders...", flush=True)
        train_dataloader = create_dataloader_with_encoders(
            train_adata,
            condition_cols,
            cvae.label_encoders,
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )
        val_dataloader = create_dataloader_with_encoders(
            val_adata,
            condition_cols,
            cvae.label_encoders,
            batch_size=CONFIG['batch_size'],
            shuffle=False
        )

        print(f"[OK] DataLoaders created:", flush=True)
        print(f"   - Training batches: {len(train_dataloader)}", flush=True)
        print(f"   - Validation batches: {len(val_dataloader)}", flush=True)

        # Quick data validation check
        print(f"\nValidating data loading...", flush=True)
        for x_batch, y_batch in train_dataloader:
            print(f"   - Sample batch shape: x={x_batch.shape}, y={y_batch.shape}", flush=True)
            print(f"   - Data types: x={x_batch.dtype}, y={y_batch.dtype}", flush=True)
            print(f"   - x range: [{x_batch.min():.2f}, {x_batch.max():.2f}]", flush=True)
            print(f"   - y range: [{y_batch.min():.2f}, {y_batch.max():.2f}]", flush=True)
            print(f"   - NaN check: x has NaN={torch.isnan(x_batch).any()}, y has NaN={torch.isnan(y_batch).any()}", flush=True)
            break
        print(f"[OK] Data validation passed!", flush=True)

    except Exception as e:
        print(f"[ERROR] Error creating data splits: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # =============================================================================
    # DEVICE SETUP INFO AND TEST
    # =============================================================================

    print(f"\nDevice setup:", flush=True)
    print(f"   - CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"   - Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name()}", flush=True)
        print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

        # Test CUDA operations
        print(f"\nTesting CUDA operations...", flush=True)
        try:
            test_tensor = torch.randn(100, 100).to(device)
            test_result = test_tensor @ test_tensor.T
            print(f"[OK] CUDA operations working", flush=True)
            del test_tensor, test_result
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ERROR] CUDA test failed: {e}", flush=True)
            return

    # =============================================================================
    # TRAINING
    # =============================================================================

    print(f"\nStarting training...", flush=True)
    training_start_time = time.time()

    try:
        training_history = cvae.train_net(
        #training_history = cvae.train_net_with_kl_control(
            train_loader=train_dataloader,
            test_loader=val_dataloader,
            n_epochs=CONFIG['n_epochs'],
            patience=CONFIG['patience'],
            overfitting_patience=CONFIG['overfitting_patience'],
            min_delta=CONFIG['min_delta'],
            save_model=True
        )

        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        print(f"\n[COMPLETE] Training completed successfully!", flush=True)
        print(f"   - Total time: {total_training_time / 60:.2f} minutes", flush=True)
        print(f"   - Epochs completed: {len(training_history['train_loss'])}", flush=True)
        print(f"   - Early stopping: {'Yes' if training_history.get('stopped_early', False) else 'No'}", flush=True)
        if training_history.get('stopped_early', False):
            print(f"   - Stop reason: {training_history.get('stop_reason', 'Unknown')}", flush=True)

    except Exception as e:
        print(f"[ERROR] Error during training: {e}", flush=True)
        return

    # =============================================================================
    # FINAL EVALUATION
    # =============================================================================

    print(f"\nFinal evaluation...", flush=True)
    try:
        # Test on validation set
        final_metrics = cvae.test_model(val_dataloader)

        print(f"[OK] Final validation metrics:", flush=True)
        print(f"   - Total Loss: {final_metrics['total_loss']:.6f}", flush=True)
        print(f"   - Reconstruction Loss: {final_metrics['recon_loss']:.6f}", flush=True)
        print(f"   - KL Divergence: {final_metrics['kl_loss']:.6f}", flush=True)

        # Print training summary
        print(f"\nTraining summary:", flush=True)
        print(f"   - Final train loss: {training_history['train_loss'][-1]:.6f}", flush=True)
        if len(training_history['val_loss']) > 0:
            print(f"   - Final val loss: {training_history['val_loss'][-1]:.6f}", flush=True)
            print(f"   - Best val loss: {training_history.get('best_val_loss', 'N/A'):.6f}", flush=True)
            print(f"   - Best epoch: {training_history.get('best_epoch', 'N/A')}", flush=True)

        # Average epoch time
        if len(training_history['epoch_times']) > 0:
            avg_epoch_time = np.mean(training_history['epoch_times'])
            print(f"   - Average epoch time: {avg_epoch_time:.2f}s", flush=True)

    except Exception as e:
        print(f"[WARN] Error during final evaluation: {e}", flush=True)

    # =============================================================================
    # VISUALIZATION AND ANALYSIS
    # =============================================================================

    print(f"\nCreating training analysis plots...", flush=True)
    try:
        # Create comprehensive training plot
        #cvae.plot_comprehensive_training_history(
        #    history=training_history,
        #   save_path=PATHS['plot']
        #)

        cvae.plot_enhanced_training_history(
            history=training_history,
            save_path=PATHS['plot']
        )

        print(f"[OK] Training analysis saved to: {PATHS['plot']}", flush=True)

    except Exception as e:
        print(f"[WARN] Error creating plots: {e}", flush=True)

    # =============================================================================
    # RESULTS SUMMARY
    # =============================================================================

    print(f"\n" + "=" * 60, flush=True)
    print("TRAINING COMPLETE - SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Files created:", flush=True)
    print(f"   - Model: {PATHS['model']}", flush=True)
    print(f"   - Metrics: {PATHS['metrics']}", flush=True)
    print(f"   - Plot: {PATHS['plot']}", flush=True)
    print(flush=True)
    print(f"Key Results:", flush=True)
    print(f"   - Architecture: {CONFIG['n_genes']}+{len(condition_cols)} -> {cvae.hidden_dims} -> {CONFIG['latent_size']}", flush=True)
    print(f"   - Training time: {total_training_time / 60:.2f} minutes", flush=True)
    print(f"   - Epochs: {len(training_history['train_loss'])}/{CONFIG['n_epochs']}", flush=True)
    print(f"   - Final train loss: {training_history['train_loss'][-1]:.6f}", flush=True)
    if len(training_history['val_loss']) > 0:
        print(f"   - Final val loss: {training_history['val_loss'][-1]:.6f}", flush=True)
        print(f"   - Best val loss: {training_history.get('best_val_loss', 'N/A'):.6f}", flush=True)

    print(f"\nModel ready for:", flush=True)
    print(f"   - Generation of synthetic single-cell data", flush=True)
    print(f"   - Conditional generation based on: {', '.join(condition_cols)}", flush=True)
    print(f"   - Further analysis and downstream tasks", flush=True)

    print(f"\nEnhanced CVAE training pipeline completed successfully!", flush=True)

    return cvae, training_history


def load_and_analyze_previous_training():
    """
    Utility function to load and analyze a previously trained model
    """
    print("Loading previous training results...", flush=True)

    # Initialize model (architecture must match)
    condition_cols = ['annotation_major', 'tissue']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cvae = Enhanced_Vanilla_CVAE_Categorical(
        n_genes=2000,
        n_labels=len(condition_cols),
        latent_size=32,
        device=device,
        model='trained_models_cancer/vanilla_cvae_relu_raw_filtered_0.997_800_400_32.pt',
        metrics_path='cancer_data/trained_models_cancer/vanilla_cvae_relu_raw_filtered_0.997_800_400_32.json',
        condition_cols=condition_cols
    )

    # Load trained weights
    try:
        cvae.load_state_dict(torch.load(cvae.model, map_location=device))
        print("[OK] Model weights loaded successfully", flush=True)
    except Exception as e:
        print(f"[ERROR] Error loading model weights: {e}", flush=True)
        return None

    # Load label encoders
    try:
        cvae.load_label_encoders()
        print("[OK] Label encoders loaded successfully", flush=True)
    except Exception as e:
        print(f"[WARN] Could not load label encoders: {e}", flush=True)

    # Load and plot training history
    try:
        cvae.plot_comprehensive_training_history(
            save_path='cancer_data/trained_models_cancer/reloaded_training_analysis.png'
        )
        print("[OK] Training analysis recreated", flush=True)
    except Exception as e:
        print(f"[WARN] Could not recreate training analysis: {e}", flush=True)

    return cvae


if __name__ == "__main__":
    """
    Main execution

    Usage:
    python enhanced_cvae_training.py          # Train new model

    Or import and use functions:
    from enhanced_cvae_training import main, load_and_analyze_previous_training

    # Train new model
    model, history = main()

    # Load and analyze previous training
    model = load_and_analyze_previous_training()
    """

    try:
        model, history = main()

        # Optional: Show some quick stats
        if model and history:
            print(f"\nQuick model test:", flush=True)
            print(f"   - Model device: {model.device}", flush=True)
            print(f"   - Training completed: {len(history['train_loss'])} epochs", flush=True)
            print(f"   - Model ready for inference!", flush=True)

    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Training interrupted by user", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}", flush=True)
        import traceback

        traceback.print_exc()