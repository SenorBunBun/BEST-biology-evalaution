import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class DiffusionIntegrationVAE(nn.Module):
    """
    PyTorch Variational Autoencoder specifically designed for integration with diffusion models.
    
    Based on cfDiffusion architecture:
    - Compresses scRNA-seq data to 128D latent space
    - Optimized for dimensionality reduction and denoising
    - Follows research-proven architectural choices for scRNA-seq
    - Flexible hidden dimensions architecture
    
    Architecture:
    - Encoder: Flexible MLP layers with sigmoid activation (n_genes -> hidden_dims -> latent_size*2)
    - Decoder: Flexible MLP layers with sigmoid activation (latent_size -> reversed_hidden_dims -> n_genes)
    - Loss: MSE reconstruction + KL divergence
    
    Args:
        n_genes (int): Number of input genes/features
        latent_size (int): Size of latent space (default: 128)
        hidden_dims (list): List of hidden layer dimensions (default: [1024, 1024])
        auto_load (bool): Whether to automatically load pre-trained weights from model_path (default: False)
        model (str): Path to saved model weights (default: 'diffusion_model_results/diffusion_vae.pth')
        verbose (bool): Whether to print status messages (default: True)
        Other kwargs: beta, dropout, lr, wd, metrics_path
    """
    
    def __init__(self, n_genes=2000, latent_size=128, hidden_dims=None, **kwargs):
        super(DiffusionIntegrationVAE, self).__init__()
        
        # Core parameters - following cfDiffusion specification
        self.n_genes = n_genes
        self.latent_size = latent_size
        self.beta = kwargs.get('beta', 0.001)  # Lower beta for dimensionality reduction focus
        self.dropout = kwargs.get('dropout', 0.1)
        self.learn_rate = kwargs.get('lr', 0.001)
        self.weight_decay = kwargs.get('wd', 0.001)
        self.model_path = kwargs.get('model', 'diffusion_model_results/diffusion_vae.pth')
        self.metrics_path = kwargs.get('metrics_path', 'diffusion_model_results/vae_training_metrics.json')
        self.auto_load = kwargs.get('auto_load', False)  # Whether to automatically load pre-trained weights
        self.verbose = kwargs.get('verbose', True)
        
        # Flexible hidden dimensions - default follows cfDiffusion: [1024, 1024]
        if hidden_dims is None:
            self.hidden_dims = [1024, 1024]
        else:
            self.hidden_dims = hidden_dims
        
        # Validate hidden dimensions
        if not isinstance(self.hidden_dims, list) or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must be a non-empty list")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build the network
        self._build_encoder()
        self._build_decoder()
        
        # Move to device
        self.to(self.device)
        
        # Initialize weights
        self.init_weights()
        
        # Conditionally load pre-trained weights
        if self.auto_load:
            if self.verbose:
                print(f"🔄 Attempting to load pre-trained weights from: {self.model_path}")
            success = self.load_trained_model()
            if not success and self.verbose:
                print(f"⚠️ Could not load pre-trained weights, using randomly initialized weights")
        elif self.verbose:
            print("🎲 Using randomly initialized weights")
    
    def _build_encoder(self):
        """
        Build flexible encoder network
        Uses sigmoid activation (proven better for scRNA-seq)
        
        Architecture: n_genes -> hidden_dims[0] -> hidden_dims[1] -> ... -> latent_size*2
        """
        encoder_layers = []
        
        # Input dimension
        input_dim = self.n_genes
        
        # Build hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            # Batch normalization
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            # Sigmoid activation
            encoder_layers.append(nn.Sigmoid())
            # Dropout
            encoder_layers.append(nn.Dropout(self.dropout))
            
            input_dim = hidden_dim
        
        # Create encoder backbone
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections: last_hidden_dim -> latent_size
        self.z_mean = nn.Linear(self.hidden_dims[-1], self.latent_size)
        self.z_log_var = nn.Linear(self.hidden_dims[-1], self.latent_size)
    
    def _build_decoder(self):
        """
        Build flexible decoder network
        Uses sigmoid activation for hidden layers
        
        Architecture: latent_size -> reversed_hidden_dims -> n_genes
        """
        decoder_layers = []
        
        # Start from latent size
        input_dim = self.latent_size
        
        # Reverse hidden dimensions for decoder
        reversed_hidden_dims = self.hidden_dims[::-1]
        
        # Build hidden layers (except the last one)
        for i, hidden_dim in enumerate(reversed_hidden_dims):
            # Linear layer
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            # Batch normalization
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            # Sigmoid activation
            decoder_layers.append(nn.Sigmoid())
            # Dropout
            decoder_layers.append(nn.Dropout(self.dropout))
            
            input_dim = hidden_dim
        
        # Output layer: last_hidden_dim -> n_genes
        decoder_layers.append(nn.Linear(input_dim, self.n_genes))
        # No activation for output (linear output for normalized data)
        
        # Create decoder
        self.decoder = nn.Sequential(*decoder_layers)
    
    def init_weights(self):
        """Initialize weights using He uniform initialization"""
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='sigmoid')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        self.apply(init_layer)
        
        if self.verbose:
            print("✅ VAE weights initialized successfully")
    
    def reparameterize(self, mean, log_var):
        """Sample latent space using reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def encode(self, x):
        """Encode input to latent space"""
        # Pass through encoder backbone
        h = self.encoder(x)
        
        # Get latent parameters
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        
        # Sample from latent space
        z = self.reparameterize(z_mean, z_log_var)
        
        return z, z_mean, z_log_var
    
    def decode(self, z):
        """Decode latent representation back to gene expression space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass through the VAE"""
        z, z_mean, z_log_var = self.encode(x)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'z_mean': z_mean,
            'z_log_var': z_log_var,
            'z': z
        }
    
    def kl_divergence(self, z_mean, z_log_var):
        """KL divergence between posterior and prior"""
        kl = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
        return kl
    
    def reconstruction_loss(self, x_true, x_reconstructed):
        """MSE reconstruction loss (appropriate for normalized data)"""
        mse = torch.sum((x_true - x_reconstructed).pow(2), dim=-1)
        return mse
    
    def criterion(self, x_true, outputs, beta=None):
        """VAE loss function optimized for diffusion integration"""
        if beta is None:
            beta = self.beta
        
        # Reconstruction loss (MSE for normalized data)
        recon_loss = self.reconstruction_loss(x_true, outputs['reconstruction'])
        
        # KL divergence loss
        kl_loss = self.kl_divergence(outputs['z_mean'], outputs['z_log_var'])
        
        # Total loss
        total_loss = torch.mean(recon_loss + beta * kl_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': torch.mean(recon_loss),
            'kl_loss': torch.mean(kl_loss)
        }
    
    def train_net(self, train_loader, n_epochs=100, patience=15, 
                  validation_loader=None, save_model=True, 
                  overfitting_patience=8, min_delta=1e-4):
        """
        Train the VAE for diffusion integration with comprehensive monitoring
        
        Args:
            train_loader: Training data loader
            n_epochs: Maximum number of epochs
            patience: Early stopping patience for validation loss
            validation_loader: Validation data loader
            save_model: Whether to save the model
            overfitting_patience: Patience for overfitting detection
            min_delta: Minimum change to qualify as improvement
        """
        import time
        import json
        import os
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay
        )
        
        # Ensure directories exist
        model_dir = os.path.dirname(self.model_path)
        metrics_dir = os.path.dirname(self.metrics_path)
        
        for directory in [model_dir, metrics_dir]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
        
        # Training history with more detailed metrics
        history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_loss': [], 'val_recon': [], 'val_kl': [],
            'learning_rate': [], 'epoch_times': [],
            'best_epoch': 0, 'best_val_loss': float('inf'),
            'total_training_time': 0, 'stopped_early': False,
            'stop_reason': 'completed'
        }
        
        # Early stopping and overfitting detection variables
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        overfitting_counter = 0
        best_epoch = 0
        no_improvement_epochs = 0
        
        # Training monitoring
        start_time = time.time()
        epoch_times = []
        
        print("🧬 TRAINING Diffusion Integration VAE...")
        print(f"📊 Model will be saved to: {self.model_path}")
        print(f"📈 Metrics will be saved to: {self.metrics_path}")
        print(f"🏗️  Architecture: {self.n_genes} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.latent_size}")
        print(f"💻 Device: {self.device}")
        print(f"⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, optimizer)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_recon'].append(train_metrics['recon_loss'])
            history['train_kl'].append(train_metrics['kl_loss'])
            history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            history['epoch_times'].append(epoch_time)
            
            # Validation phase
            if validation_loader is not None:
                val_metrics = self._validate_epoch(validation_loader)
                history['val_loss'].append(val_metrics['total_loss'])
                history['val_recon'].append(val_metrics['recon_loss'])
                history['val_kl'].append(val_metrics['kl_loss'])
                
                current_val_loss = val_metrics['total_loss']
                current_train_loss = train_metrics['total_loss']
                
                # Check for improvement in validation loss
                if current_val_loss < best_val_loss - min_delta:
                    best_val_loss = current_val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    no_improvement_epochs = 0
                    
                    history['best_epoch'] = best_epoch
                    history['best_val_loss'] = best_val_loss
                    
                    if save_model:
                        try:
                            torch.save(self.state_dict(), self.model_path)
                            if self.verbose and epoch % 10 == 0:
                                print(f"💾 Model saved (best val loss: {best_val_loss:.6f})")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not save model: {e}")
                else:
                    patience_counter += 1
                    no_improvement_epochs += 1
                
                # Check for improvement in training loss (for overfitting detection)
                if current_train_loss < best_train_loss - min_delta:
                    best_train_loss = current_train_loss
                    overfitting_counter = 0
                else:
                    overfitting_counter += 1
                
                # Overfitting detection
                overfitting_detected = (
                    overfitting_counter >= overfitting_patience and 
                    no_improvement_epochs >= overfitting_patience and
                    len(history['val_loss']) > overfitting_patience
                )
                
                # Calculate estimated time remaining
                avg_epoch_time = np.mean(epoch_times[-10:])  # Last 10 epochs
                remaining_epochs = min(n_epochs - epoch - 1, patience - patience_counter)
                eta_minutes = (remaining_epochs * avg_epoch_time) / 60
                
                # Progress display
                if self.verbose:
                    progress_bar = self._create_progress_bar(epoch + 1, n_epochs, 40)
                    print(f"Epoch {epoch+1:3d}/{n_epochs} {progress_bar} | "
                          f"Train: {current_train_loss:.4f} | Val: {current_val_loss:.4f} | "
                          f"Recon: {train_metrics['recon_loss']:.4f} | KL: {train_metrics['kl_loss']:.4f} | "
                          f"Time: {epoch_time:.1f}s | ETA: {eta_minutes:.1f}m")
                
                # Early stopping conditions
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping: No improvement for {patience} epochs")
                    print(f"🏆 Best model from epoch {best_epoch} (val loss: {best_val_loss:.6f})")
                    history['stopped_early'] = True
                    history['stop_reason'] = 'patience_exceeded'
                    break
                
                if overfitting_detected:
                    print(f"\n🔴 Overfitting detected: Training loss improving but validation loss stagnant")
                    print(f"🏆 Best model from epoch {best_epoch} (val loss: {best_val_loss:.6f})")
                    history['stopped_early'] = True
                    history['stop_reason'] = 'overfitting_detected'
                    break
                
                # Check for loss explosion
                if (np.isnan(current_val_loss) or np.isnan(current_train_loss) or 
                    current_val_loss > 1e6 or current_train_loss > 1e6):
                    print(f"\n💥 Training unstable: Loss values became extreme or NaN")
                    history['stopped_early'] = True
                    history['stop_reason'] = 'loss_explosion'
                    break
                
            else:
                # No validation - just track training loss
                current_train_loss = train_metrics['total_loss']
                
                if current_train_loss < best_train_loss - min_delta:
                    best_train_loss = current_train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.verbose:
                    progress_bar = self._create_progress_bar(epoch + 1, n_epochs, 40)
                    print(f"Epoch {epoch+1:3d}/{n_epochs} {progress_bar} | "
                          f"Train: {current_train_loss:.4f} | "
                          f"Recon: {train_metrics['recon_loss']:.4f} | "
                          f"KL: {train_metrics['kl_loss']:.4f} | "
                          f"Time: {epoch_time:.1f}s")
                
                # Early stopping for training-only mode
                if patience_counter >= patience * 2:
                    print(f"\n🛑 Early stopping: No improvement in training loss for {patience * 2} epochs")
                    history['stopped_early'] = True
                    history['stop_reason'] = 'training_plateau'
                    break
        
        # Training completion
        total_time = time.time() - start_time
        history['total_training_time'] = total_time
        
        print("\n" + "="*60)
        print("✅ VAE Training completed!")
        print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
        print(f"📊 Total epochs completed: {len(history['train_loss'])}")
        
        if validation_loader is not None:
            print(f"🏆 Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
            print(f"📈 Final validation loss: {history['val_loss'][-1]:.6f}")
        
        print(f"📉 Final training loss: {history['train_loss'][-1]:.6f}")
        
        # Save final model if no validation was used
        if save_model and validation_loader is None:
            try:
                torch.save(self.state_dict(), self.model_path)
                print(f"💾 Final model saved to: {self.model_path}")
            except Exception as e:
                print(f"⚠️ Error saving final model: {e}")
        elif validation_loader is not None:
            print(f"💾 Best model saved to: {self.model_path}")
        
        # Save training metrics
        try:
            # Convert numpy values to regular Python types for JSON serialization
            history_serializable = {}
            for key, value in history.items():
                if isinstance(value, list):
                    history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
                elif isinstance(value, (np.floating, np.integer)):
                    history_serializable[key] = float(value)
                else:
                    history_serializable[key] = value
            
            with open(self.metrics_path, 'w') as f:
                json.dump(history_serializable, f, indent=2)
            print(f"📈 Training metrics saved to: {self.metrics_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save training metrics: {e}")
        
        return history
    
    def _create_progress_bar(self, current, total, width=40):
        """Create a simple progress bar"""
        progress = current / total
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {progress*100:.1f}%"
    
    def _train_epoch(self, data_loader, optimizer):
        """Train for one epoch"""
        self.train()
        epoch_metrics = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
        
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                x_batch = batch[0]  # Handle DataLoader returning tuples
            else:
                x_batch = batch
            
            x_batch = x_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = self(x_batch)
            loss_dict = self.criterion(x_batch, outputs)
            
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            # Store metrics
            for key in epoch_metrics:
                epoch_metrics[key].append(loss_dict[key].item())
        
        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def _validate_epoch(self, data_loader):
        """Validate for one epoch"""
        self.eval()
        epoch_metrics = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    x_batch = batch[0]
                else:
                    x_batch = batch
                
                x_batch = x_batch.to(self.device)
                outputs = self(x_batch)
                loss_dict = self.criterion(x_batch, outputs)
                
                for key in epoch_metrics:
                    epoch_metrics[key].append(loss_dict[key].item())
        
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def test_model(self, test_loader):
        """Comprehensive model evaluation"""
        self.eval()
        test_metrics = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
        all_embeddings = []
        all_reconstructions = []

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    x_batch = batch[0]
                else:
                    x_batch = batch

                x_batch = x_batch.to(self.device)
                outputs = self(x_batch)
                loss_dict = self.criterion(x_batch, outputs)

                for key in test_metrics:
                    test_metrics[key].append(loss_dict[key].item())

                # Collect embeddings and reconstructions
                all_embeddings.append(outputs['z_mean'].cpu().numpy())
                all_reconstructions.append(outputs['reconstruction'].cpu().numpy())

        # Handle empty test loader
        if not all_embeddings:
            return {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'kl_loss': 0.0,
                'embeddings': np.array([]),
                'reconstructions': np.array([])
            }

        # Aggregate results
        final_metrics = {key: np.mean(values) for key, values in test_metrics.items()}
        final_metrics.update({
            'embeddings': np.concatenate(all_embeddings, axis=0),
            'reconstructions': np.concatenate(all_reconstructions, axis=0)
        })

        return final_metrics
    
    def get_latent_representation(self, x):
        """Get latent representation for diffusion model input"""
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif hasattr(x, 'toarray'):
            x = torch.tensor(x.toarray(), dtype=torch.float32)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            _, z_mean, z_log_var = self.encode(x)
        
        return z_mean.cpu().numpy(), z_log_var.cpu().numpy()
    
    def reconstruct_from_latent(self, z):
        """Reconstruct gene expression from latent representation"""
        self.eval()
        
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)
        
        z = z.to(self.device)
        
        with torch.no_grad():
            reconstruction = self.decode(z)
        
        return reconstruction.cpu().numpy()
    
    def generate_random_samples(self, num_samples):
        """Generate random samples from prior distribution"""
        self.eval()
        
        z = torch.randn(num_samples, self.latent_size).to(self.device)
        
        with torch.no_grad():
            samples = self.decode(z)
        
        return samples.cpu().numpy()
    
    def load_trained_model(self, model_path=None):
        """
        Load a trained model from saved state dict
        
        Args:
            model_path: Path to saved state dict. If     model=None, uses self.model_path
        """
        if model_path is None:
            model_path = self.model_path
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict)
            self.to(self.device)
            print(f"✅ Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model from {model_path}: {e}")
            return False
    
    def save_model_for_deployment(self, save_path):
        """
        Save complete model for deployment
        
        Args:
            save_path: Path to save the complete model
        """
        try:
            torch.save({
                'state_dict': self.state_dict(),
                'config': {
                    'n_genes': self.n_genes,
                    'latent_size': self.latent_size,
                    'hidden_dims': self.hidden_dims,
                    'beta': self.beta,
                    'dropout': self.dropout
                }
            }, save_path)
            print(f"💾 Complete model saved to: {save_path}")
            
        except Exception as e:
            print(f"❌ Error saving complete model: {e}")
    
    def load_training_metrics(self, metrics_path=None):
        """
        Load saved training metrics
        
        Args:
            metrics_path: Path to metrics file. If None, uses self.metrics_path
            
        Returns:
            dict: Training history and metrics
        """
        import json
        
        if metrics_path is None:
            metrics_path = self.metrics_path
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"📈 Training metrics loaded from: {metrics_path}")
            return metrics
        except Exception as e:
            print(f"❌ Error loading metrics from {metrics_path}: {e}")
            return None
    
    def plot_comprehensive_training_history(self, history=None, save_path=None):
        """
        Plot comprehensive training curves with additional insights
        
        Args:
            history: Training history dict. If None, tries to load from saved metrics
            save_path: Path to save the plot. If None, just displays
        """
        if history is None:
            history = self.load_training_metrics()
            if history is None:
                print("❌ No training history available")
                return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Total loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history['train_loss'], label='Train', color='blue', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax1.plot(history['val_loss'], label='Validation', color='red', linewidth=2)
            # Mark best epoch
            if 'best_epoch' in history and history['best_epoch'] > 0:
                best_idx = history['best_epoch'] - 1
                if best_idx < len(history['val_loss']):
                    ax1.axvline(x=best_idx, color='green', linestyle='--', alpha=0.7, label='Best Model')
        ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reconstruction loss
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['train_recon'], label='Train', color='blue', linewidth=2)
        if 'val_recon' in history and len(history['val_recon']) > 0:
            ax2.plot(history['val_recon'], label='Validation', color='red', linewidth=2)
        ax2.set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. KL divergence
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(history['train_kl'], label='Train', color='blue', linewidth=2)
        if 'val_kl' in history and len(history['val_kl']) > 0:
            ax3.plot(history['val_kl'], label='Validation', color='red', linewidth=2)
        ax3.set_title('KL Divergence', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('KL Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Learning rate schedule
        ax4 = fig.add_subplot(gs[1, 0])
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            ax4.plot(history['learning_rate'], color='purple', linewidth=2)
            ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.grid(True, alpha=0.3)
        
        # 5. Epoch timing
        ax5 = fig.add_subplot(gs[1, 1])
        if 'epoch_times' in history and len(history['epoch_times']) > 0:
            ax5.plot(history['epoch_times'], color='orange', linewidth=2)
            ax5.axhline(y=np.mean(history['epoch_times']), color='red', linestyle='--', 
                       alpha=0.7, label=f'Mean: {np.mean(history["epoch_times"]):.1f}s')
            ax5.set_title('Epoch Training Time', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Time (seconds)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Loss correlation (if validation exists)
        ax6 = fig.add_subplot(gs[1, 2])
        if 'val_loss' in history and len(history['val_loss']) > 0:
            # Plot train vs val loss to detect overfitting
            min_len = min(len(history['train_loss']), len(history['val_loss']))
            train_subset = history['train_loss'][:min_len]
            val_subset = history['val_loss'][:min_len]
            
            ax6.scatter(train_subset, val_subset, alpha=0.6, c=range(min_len), 
                       cmap='viridis', s=30)
            
            # Add diagonal line (perfect correlation)
            min_loss = min(min(train_subset), min(val_subset))
            max_loss = max(max(train_subset), max(val_subset))
            ax6.plot([min_loss, max_loss], [min_loss, max_loss], 'r--', alpha=0.7, 
                    label='Perfect Correlation')
            
            ax6.set_title('Train vs Validation Loss', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Training Loss')
            ax6.set_ylabel('Validation Loss')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(ax6.collections[0], ax=ax6)
            cbar.set_label('Epoch')
        
        # 7. Training summary (text)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create summary text
        summary_text = f"""
        TRAINING SUMMARY
        ================
        Architecture: {self.n_genes} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.latent_size}
        Total Epochs: {len(history['train_loss'])}
        Training Time: {history.get('total_training_time', 0)/60:.1f} minutes
        Early Stopping: {'Yes' if history.get('stopped_early', False) else 'No'}
        Stop Reason: {history.get('stop_reason', 'completed')}
        Device: {self.device}
        
        FINAL METRICS
        =============
        Final Train Loss: {history['train_loss'][-1]:.6f}
        Final Recon Loss: {history['train_recon'][-1]:.6f}
        Final KL Loss: {history['train_kl'][-1]:.6f}
        """
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            summary_text += f"""
        Final Val Loss: {history['val_loss'][-1]:.6f}
        Best Val Loss: {history.get('best_val_loss', 'N/A')} (Epoch {history.get('best_epoch', 'N/A')})
        """
        
        if 'epoch_times' in history and len(history['epoch_times']) > 0:
            summary_text += f"""
        Avg Epoch Time: {np.mean(history['epoch_times']):.1f}s
        Total Training Time: {sum(history['epoch_times'])/60:.1f} minutes
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('PyTorch VAE Training Analysis Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training analysis plot saved to: {save_path}")
        
        plt.tight_layout()
        plt.show()


# Utility function for creating PyTorch datasets
def create_pytorch_datasets(X_data, batch_size=256, validation_split=0.2, test_split=0.1,
                            num_workers=0, split_indices=None):
    """
    Create PyTorch data loaders from preprocessed data (already normalized)

    Args:
        X_data: Preprocessed gene expression data (already normalized/log-transformed)
        batch_size: Batch size for training
        validation_split: Fraction for validation
        test_split: Fraction for test
        num_workers: Number of workers for data loading
        split_indices: Path to pre-computed split indices (optional)

    Returns:
        train_loader, val_loader, test_loader
    """
    # Convert to numpy if needed
    if hasattr(X_data, 'toarray'):
        X_data = X_data.toarray()
    X_data = X_data.astype(np.float32)

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X_data, dtype=torch.float32)

    # Create dataset
    dataset = TensorDataset(X_tensor)
    n_samples = len(dataset)

    if split_indices:
        # Load pre-computed stratified split
        import sys
        from pathlib import Path
        pipeline_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(pipeline_dir))
        from utils.data_splitter import load_split_indices

        train_indices, val_indices = load_split_indices(split_indices)
        # Test indices are whatever is left
        all_indices = set(range(n_samples))
        test_indices = list(all_indices - set(train_indices) - set(val_indices))

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices) if test_indices else torch.utils.data.Subset(dataset, [])

        print(f"[OK] Loaded stratified split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    else:
        # Fallback to random split
        test_size = int(n_samples * test_split)
        val_size = int(n_samples * validation_split)
        train_size = n_samples - test_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader



# Example usage with your processed h5ad data
if __name__ == "__main__":
    """
    Example usage with your preprocessed h5ad file:
    
    import scanpy as sc
    
    # Load your preprocessed data
    adata = sc.read_h5ad('your_processed_data.h5ad')
    X_processed = adata.X  # Should already be normalized/log-transformed
    
    # Create datasets
    train_loader, val_loader, test_loader = create_pytorch_datasets(X_processed, batch_size=128)
    
    # Initialize VAE with flexible architecture
    vae = DiffusionIntegrationVAE(
        n_genes=X_processed.shape[1],  # Should be 2000
        latent_size=128,
        hidden_dims=[800, 400],  # Can now be [800, 400, 200] or [1024, 512, 256] etc.
        beta=0.001,
        dropout=0.1,
        lr=0.001,
        model='diffusion_model_results/diffusion_vae.pth',
        metrics_path='diffusion_model_results/training_metrics.json',
        auto_load=False,  # Set to True to automatically load pre-trained weights
        verbose=True
    )
    
    # Train with comprehensive monitoring
    history = vae.train_net(
        train_loader=train_loader,
        validation_loader=val_loader,
        n_epochs=200,
        patience=20,
        overfitting_patience=10,
        min_delta=1e-4,
        save_model=True
    )
    
    # Evaluate and analyze
    test_results = vae.test_model(test_loader)
    print(f"📊 Final Test Results:")
    print(f"   Total Loss: {test_results['total_loss']:.6f}")
    print(f"   Reconstruction Loss: {test_results['recon_loss']:.6f}")
    print(f"   KL Loss: {test_results['kl_loss']:.6f}")
    
    # Create comprehensive analysis
    vae.plot_comprehensive_training_history(
        history, 
        save_path='diffusion_model_results/vae_training_analysis.png'
    )
    
    # Get latent representations for diffusion training
    z_mean, z_log_var = vae.get_latent_representation(X_processed)
    
    # Save latent representations for diffusion model
    np.save('diffusion_model_results/latent_representations_mean.npy', z_mean)
    np.save('diffusion_model_results/latent_representations_logvar.npy', z_log_var)
    
    print("✅ PyTorch VAE training complete with full monitoring!")
    
    # Example of loading everything back later:
    # =============================================
    
    # Option 1: Automatically load pre-trained weights during initialization
    # new_vae = DiffusionIntegrationVAE(
    #     n_genes=2000, 
    #     latent_size=128, 
    #     hidden_dims=[800, 400],
    #     model='diffusion_model_results/diffusion_vae.pth',
    #     auto_load=True  # Automatically loads pre-trained weights
    # )
    
    # Option 2: Manual loading (original method)
    # new_vae = DiffusionIntegrationVAE(n_genes=2000, latent_size=128, hidden_dims=[800, 400])
    # new_vae.load_trained_model('diffusion_model_results/diffusion_vae.pth')
    
    # Load training metrics and plot
    # metrics = new_vae.load_training_metrics('diffusion_model_results/vae_training_metrics.json')
    # new_vae.plot_comprehensive_training_history(metrics)
    
    # Or load complete model:
    # loaded_vae = load_complete_model('diffusion_model_results/complete_model.pth')
    """
    
    print("🧬 Enhanced PyTorch DiffusionIntegrationVAE ready for use!")
    print("\n🎯 Key Features:")
    print("   ✅ Full PyTorch implementation with CUDA support")
    print("   ✅ Flexible hidden dimensions architecture")
    print("   ✅ Automatic model saving in 'diffusion_model_results/'")
    print("   ✅ Auto-load pre-trained weights (auto_load=True)")
    print("   ✅ Comprehensive training metrics saved as JSON")
    print("   ✅ Overfitting detection and early stopping")
    print("   ✅ Real-time progress tracking with ETA")
    print("   ✅ Loss explosion detection")
    print("   ✅ Training analysis dashboard")
    print("   ✅ cfDiffusion-compatible 128D latent space")
    print("\n🏗️  Flexible Architecture Examples:")
    print("   • Default: [800, 400] (original cfDiffusion)")
    print("   • Deeper: [1024, 512, 256] (3 hidden layers)")
    print("   • Wider: [1600, 800] (wider layers)")
    print("   • Custom: [512, 256, 128, 64] (4 hidden layers)")
    print("\n📁 Output Structure:")
    print("   diffusion_model_results/")
    print("   ├── diffusion_vae.pth")
    print("   ├── complete_model.pth")
    print("   ├── training_metrics.json")
    print("   ├── training_analysis.png")
    print("   ├── latent_representations_mean.npy")
    print("   └── latent_representations_logvar.npy")