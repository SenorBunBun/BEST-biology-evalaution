import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import time
import json
import os


class Enhanced_Vanilla_CVAE(torch.nn.Module):
    def __init__(self, n_genes, n_labels, latent_size, **kwargs):
        super(Enhanced_Vanilla_CVAE, self).__init__()

        self.n_genes = n_genes
        self.n_labels = n_labels
        self.latent_size = latent_size #32 for original
        self.beta = kwargs.get('beta', 0.05)
        self.dropout = kwargs.get('dropout', 0.03)
        self.learn_rate = kwargs.get('lr', 0.001)
        self.weight_decay = kwargs.get('wd', 0.001)
        self.init_w = kwargs.get('init_w', False)
        self.model = kwargs.get('model', 'trained_models/enhanced_vanilla_cvae_cellrangerhvg_relu.pt')
        self.metrics_path = kwargs.get('metrics_path', 'trained_models/cvae_training_metrics.json')
        self.verbose = kwargs.get('verbose', True)
        self.device = kwargs.get('device', torch.device('cpu'))
        #self.hidden_dims = [1024, 1024]
        self.hidden_dims = [800, 400] #relu original
        layer_1 = self.hidden_dims[0]
        layer_2 = self.hidden_dims[1]

        # Build the same architecture as original
        self.encoder = nn.Sequential(
            nn.Linear(self.n_genes + self.n_labels, layer_1),
            nn.BatchNorm1d(layer_1),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(layer_1, layer_2),
            nn.BatchNorm1d(layer_2),
            nn.Sigmoid(),
            nn.Dropout(self.dropout))

        self.mean = nn.Sequential(
            nn.Linear(layer_2, self.latent_size),
            nn.Dropout(self.dropout))

        self.logvar = nn.Sequential(
            nn.Linear(layer_2, self.latent_size),
            nn.Dropout(self.dropout))

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size + self.n_labels, layer_2),
            nn.BatchNorm1d(layer_2),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(layer_2, layer_1),
            nn.BatchNorm1d(layer_1),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
            nn.Linear(layer_1, self.n_genes)
         #   ,nn.ReLU()
        )

        if self.init_w:
            self.encoder.apply(self.init_weight)
            self.decoder.apply(self.init_weight)
            self.mean.apply(self.init_weight)
            self.logvar.apply(self.init_weight)

    def init_weight(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    def get_sample(self, mean, logvar):
        """ Sample latent space with reparameterization method """
        std = logvar.mul(0.5).exp()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        z = eps.mul(std).add(mean)
        return z

    def encode(self, x, y):
        if y.dim() == 1:
            y = y.unsqueeze(1)
        out = self.encoder(torch.cat((x, y), dim=1))
        mean, logvar = self.mean(out), self.logvar(out)
        z = self.get_sample(mean, logvar)
        return z, mean, logvar

    def decode(self, z, y):
        if y.dim() == 1:
            y = y.unsqueeze(1)
        x_reconstructed = self.decoder(torch.cat((z, y), dim=1))
        return x_reconstructed

    def forward(self, x, y):
        """ Full forward pass through network """
        z, mean, logvar = self.encode(x, y)
        X_reconstructed = self.decode(z, y)
        return X_reconstructed, mean, logvar, z

    def criterion(self, X_reconstructed, X_true, mean, logvar):
        """ Enhanced VAE Loss with detailed breakdown """
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        kl_loss = torch.mean(kld)

        # Reconstruction loss (MSE)
        mse = torch.sum((X_true - X_reconstructed).pow(2), dim=-1)
        recon_loss = torch.mean(mse)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def train_net(self, train_loader, n_epochs=100, patience=15,
                  test_loader=None, save_model=True,
                  overfitting_patience=8, min_delta=1e-4,
                  epoch_callback=None, checkpoint_freq=None, checkpoint_dir=None):
        """
        Enhanced training with comprehensive monitoring

        Args:
            epoch_callback: Optional callback function called after each epoch.
                           Signature: callback(epoch, model, history) -> None
            checkpoint_freq: Save checkpoint every N epochs (optional)
            checkpoint_dir: Directory for checkpoints (required if checkpoint_freq is set)
        """
        # Setup optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay
        )

        # Ensure directories exist
        model_dir = os.path.dirname(self.model)
        metrics_dir = os.path.dirname(self.metrics_path)

        for directory in [model_dir, metrics_dir]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}", flush=True)

        # Training history with detailed metrics
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

        print("TRAINING Enhanced Vanilla CVAE...", flush=True)
        print(f"Model will be saved to: {self.model}", flush=True)
        print(f"Metrics will be saved to: {self.metrics_path}", flush=True)
        print(f"Architecture: {self.n_genes}+{self.n_labels} -> [800,400] -> {self.latent_size}", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("-" * 60, flush=True)

        self.to(self.device)

        for epoch in range(n_epochs):
            print(f"\nStarting epoch {epoch + 1}/{n_epochs}...", flush=True)
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self._train_epoch(train_loader, optimizer)
            print(f"Epoch {epoch + 1} training completed.", flush=True)
            print(f"  Train loss: {train_metrics['total_loss']:.4f}", flush=True)
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
            if test_loader is not None:
                print(f"  Starting validation...", flush=True)
                val_metrics = self._validate_epoch(test_loader)
                print(f"  Validation completed. Val loss: {val_metrics['total_loss']:.4f}", flush=True)
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
                            print(f"  Saving model...", flush=True)
                            torch.save(self.state_dict(), self.model)
                            print(f"  Model saved successfully.", flush=True)
                            if self.verbose and epoch % 10 == 0:
                                print(f"Model saved (best val loss: {best_val_loss:.6f})", flush=True)
                        except Exception as e:
                            print(f"[WARN] Could not save model: {e}", flush=True)
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
                print(f"  Creating progress bar...", flush=True)
                if self.verbose:
                    progress_bar = self._create_progress_bar(epoch + 1, n_epochs, 40)
                    print(f"Epoch {epoch + 1:3d}/{n_epochs} {progress_bar} | "
                          f"Train: {current_train_loss:.4f} | Val: {current_val_loss:.4f} | "
                          f"Recon: {train_metrics['recon_loss']:.4f} | KL: {train_metrics['kl_loss']:.4f} | "
                          f"Time: {epoch_time:.1f}s | ETA: {eta_minutes:.1f}m", flush=True)

                # Save checkpoint if requested
                if checkpoint_freq and checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    torch.save(self.state_dict(), checkpoint_path)
                    print(f"  [CHECKPOINT] Saved: {checkpoint_path}", flush=True)

                # Call epoch callback if provided
                if epoch_callback is not None:
                    try:
                        callback_result = epoch_callback(epoch + 1, self, history)
                        if callback_result == "stop":
                            history['stopped_early'] = True
                            history['stop_reason'] = 'mmd_early_stop'
                            break
                    except Exception as e:
                        print(f"  [WARN] Epoch callback failed: {e}", flush=True)

                # Early stopping conditions
                if patience_counter >= patience:
                    print(f"\n[STOP] Early stopping: No improvement for {patience} epochs", flush=True)
                    print(f"Best model from epoch {best_epoch} (val loss: {best_val_loss:.6f})", flush=True)
                    history['stopped_early'] = True
                    history['stop_reason'] = 'patience_exceeded'
                    break

                if overfitting_detected:
                    print(f"\n[OVERFIT] Overfitting detected: Training loss improving but validation loss stagnant", flush=True)
                    print(f"Best model from epoch {best_epoch} (val loss: {best_val_loss:.6f})", flush=True)
                    history['stopped_early'] = True
                    history['stop_reason'] = 'overfitting_detected'
                    break

                # Check for loss explosion
                if (np.isnan(current_val_loss) or np.isnan(current_train_loss) or
                        current_val_loss > 1e6 or current_train_loss > 1e6):
                    print(f"\n[ERROR] Training unstable: Loss values became extreme or NaN", flush=True)
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
                    print(f"Epoch {epoch + 1:3d}/{n_epochs} {progress_bar} | "
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

        print("\n" + "=" * 60, flush=True)
        print("[DONE] CVAE Training completed!", flush=True)
        print(f"Total training time: {total_time / 60:.1f} minutes", flush=True)
        print(f"Total epochs completed: {len(history['train_loss'])}", flush=True)

        if test_loader is not None:
            print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})", flush=True)
            print(f"Final validation loss: {history['val_loss'][-1]:.6f}", flush=True)

        print(f"Final training loss: {history['train_loss'][-1]:.6f}", flush=True)

        # Save final model if no validation was used
        if save_model and test_loader is None:
            try:
                torch.save(self.state_dict(), self.model)
                print(f"Final model saved to: {self.model}", flush=True)
            except Exception as e:
                print(f"[WARN] Error saving final model: {e}", flush=True)
        elif test_loader is not None:
            print(f"Best model saved to: {self.model}", flush=True)

        # Save training metrics
        try:
            # Convert numpy values to regular Python types for JSON serialization
            history_serializable = {}
            for key, value in history.items():
                if isinstance(value, list):
                    history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in
                                                 value]
                elif isinstance(value, (np.floating, np.integer)):
                    history_serializable[key] = float(value)
                else:
                    history_serializable[key] = value

            with open(self.metrics_path, 'w') as f:
                json.dump(history_serializable, f, indent=2)
            print(f"Training metrics saved to: {self.metrics_path}", flush=True)

        except Exception as e:
            print(f"[WARN] Could not save training metrics: {e}", flush=True)

        return history

    def _create_progress_bar(self, current, total, width=40):
        """Create a simple progress bar"""
        progress = current / total
        filled = int(width * progress)
        bar = '#' * filled + '-' * (width - filled)
        return f"[{bar}] {progress * 100:.1f}%"

    def _train_epoch(self, data_loader, optimizer):
        """Train for one epoch"""
        self.train()
        epoch_metrics = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

        batch_count = 0
        for x_train, y_train in data_loader:
            batch_count += 1

            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            optimizer.zero_grad()
            X_reconstructed, mean, logvar, z = self.forward(x_train, y_train)
            loss_dict = self.criterion(X_reconstructed, x_train, mean, logvar)
            loss_dict['total_loss'].backward()
            optimizer.step()

            # Store metrics
            for key in epoch_metrics:
                epoch_metrics[key].append(loss_dict[key].item())

            # Show progress every 200 batches
            if batch_count % 200 == 0:
                avg_loss = np.mean(epoch_metrics['total_loss'][-200:])
                print(f"  Batch {batch_count}/{len(data_loader)} | Loss: {avg_loss:.4f}", flush=True)

        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}

    def _validate_epoch(self, data_loader):
        """Validate for one epoch"""
        self.eval()
        epoch_metrics = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

        batch_count = 0
        with torch.no_grad():
            for x_test, y_test in data_loader:
                batch_count += 1
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                X_reconstructed, mean, logvar, z = self.forward(x_test, y_test)
                loss_dict = self.criterion(X_reconstructed, x_test, mean, logvar)

                for key in epoch_metrics:
                    epoch_metrics[key].append(loss_dict[key].item())

                # Show progress every 100 batches during validation
                if batch_count % 100 == 0:
                    avg_loss = np.mean(epoch_metrics['total_loss'])
                    print(f"    Val batch {batch_count}/{len(data_loader)} | Loss: {avg_loss:.4f}", flush=True)

        return {key: np.mean(values) for key, values in epoch_metrics.items()}

    def test_model(self, test_loader):
        """ Evaluate the model on the test dataset """
        self.eval()
        test_metrics = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                x_reconstructed, mean, logvar, z = self.forward(x_test, y_test)
                loss_dict = self.criterion(x_reconstructed, x_test, mean, logvar)

                for key in test_metrics:
                    test_metrics[key].append(loss_dict[key].item())

        return {key: np.mean(values) for key, values in test_metrics.items()}

    def generate(self, c, num_samples):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_size).to(self.device)
            c = c.repeat(num_samples, 1)
            # print("Printing Condition")
            # print(c)
            # print(c.shape)
            samples = self.decode(z, c)
        return samples

    def load_training_metrics(self, metrics_path=None):
        """
        Load saved training metrics

        Args:
            metrics_path: Path to metrics file. If None, uses self.metrics_path

        Returns:
            dict: Training history and metrics
        """
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
        Architecture: {self.n_genes}+{self.n_labels} -> [800,400] -> {self.latent_size}
        Total Epochs: {len(history['train_loss'])}
        Training Time: {history.get('total_training_time', 0) / 60:.1f} minutes
        Early Stopping: {'Yes' if history.get('stopped_early', False) else 'No'}
        Stop Reason: {history.get('stop_reason', 'completed')}
        Device: {self.device}
        Beta: {self.beta}

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
        Total Training Time: {sum(history['epoch_times']) / 60:.1f} minutes
        """

        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.suptitle('Enhanced CVAE Training Analysis Dashboard', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training analysis plot saved to: {save_path}")

        plt.tight_layout()
        plt.show()

    def criterion(self, X_reconstructed, X_true, mean, logvar, beta=None):
        """ Enhanced VAE Loss with proper KL control """
        if beta is None:
            beta = self.beta

        batch_size = X_true.size(0)

        # KL divergence with better numerical stability
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        kl_loss = torch.mean(kld)

        # Reconstruction loss - properly normalized
        recon_loss = F.mse_loss(X_reconstructed, X_true, reduction='sum') / batch_size

        # Total loss with controlled beta
        total_loss = recon_loss + beta * kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta_used': beta
        }

    # Method 2: Smart beta annealing with KL control
    def get_beta_annealed(self, epoch, total_epochs, current_kl=None):
        """
        Smart beta annealing optimized for scRNA-seq data
        Higher beta values needed due to high dimensionality and sparsity
        """
        # Phase 1: Warmup (first 25% of training) - shorter for scRNA-seq
        warmup_epochs = int(total_epochs * 0.25)

        # Phase 2: Stable phase (next 50% of training)
        stable_epochs = int(total_epochs * 0.5)

        # Adjusted target beta for scRNA-seq complexity
        target_beta = 0.05  # Higher for scRNA-seq (was 0.01)
        max_beta = 0.2  # Higher ceiling for problematic cases

        if epoch < warmup_epochs:
            # Slower warmup from very small to target
            beta = target_beta * (epoch / warmup_epochs)
        elif epoch < warmup_epochs + stable_epochs:
            # Stable phase with target beta
            beta = target_beta
        else:
            # Adaptive phase - more conservative adjustments
            beta = target_beta

            if current_kl is not None:
                if current_kl > 120:  # Much higher threshold for scRNA-seq
                    beta = min(target_beta * 1.2, max_beta)
                elif current_kl < 5:  # Higher minimum threshold
                    beta = max(target_beta * 0.8, 0.01)
                else:
                    beta = target_beta  # Good range: 5-500 for scRNA-seq

        return beta

    # Method 3: Enhanced training epoch with KL monitoring
    def _train_epoch_with_kl_control(self, data_loader, optimizer, beta, epoch):
        """Training epoch with minimal KL monitoring"""
        self.train()
        epoch_metrics = {
            'total_loss': [], 'recon_loss': [], 'kl_loss': [],
            'kl_individual': []
        }

        for x_train, y_train in data_loader:
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)

            optimizer.zero_grad()
            X_reconstructed, mean, logvar, z = self.forward(x_train, y_train)

            loss_dict = self.criterion(X_reconstructed, x_train, mean, logvar, beta)

            # KL monitoring - individual values before averaging
            with torch.no_grad():
                kl_individual = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
                epoch_metrics['kl_individual'].extend(kl_individual.cpu().numpy())

            loss_dict['total_loss'].backward()

            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()

            for key in ['total_loss', 'recon_loss', 'kl_loss']:
                epoch_metrics[key].append(loss_dict[key].item())

        # Calculate statistics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items() if key != 'kl_individual'}

        kl_individual = epoch_metrics['kl_individual']
        avg_metrics['kl_mean'] = np.mean(kl_individual)
        avg_metrics['kl_std'] = np.std(kl_individual)
        avg_metrics['kl_max'] = np.max(kl_individual)
        avg_metrics['kl_min'] = np.min(kl_individual)

        return avg_metrics

    # Method 4: Enhanced validation epoch
    def _validate_epoch_with_kl_control(self, data_loader, beta):
        """Validation epoch with KL monitoring"""
        self.eval()
        epoch_metrics = {
            'total_loss': [], 'recon_loss': [], 'kl_loss': [],
            'kl_individual': []
        }

        with torch.no_grad():
            for x_test, y_test in data_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                X_reconstructed, mean, logvar, z = self.forward(x_test, y_test)

                loss_dict = self.criterion(X_reconstructed, x_test, mean, logvar, beta)

                # KL individual values
                kl_individual = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
                epoch_metrics['kl_individual'].extend(kl_individual.cpu().numpy())

                for key in ['total_loss', 'recon_loss', 'kl_loss']:
                    epoch_metrics[key].append(loss_dict[key].item())

        # Calculate statistics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items() if key != 'kl_individual'}

        # KL statistics
        kl_individual = epoch_metrics['kl_individual']
        avg_metrics['kl_mean'] = np.mean(kl_individual)
        avg_metrics['kl_std'] = np.std(kl_individual)
        avg_metrics['kl_max'] = np.max(kl_individual)

        return avg_metrics

    # MODIFY YOUR EXISTING train_net METHOD - ADD THESE SECTIONS:

    def train_net_with_kl_control(self, train_loader, n_epochs=100, patience=15,
                                  test_loader=None, save_model=True,
                                  overfitting_patience=8, min_delta=1e-4):
        """
        Clean CVAE training with beta annealing optimized for scRNA-seq
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8, min_lr=1e-6
        )

        history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_loss': [], 'val_recon': [], 'val_kl': [],
            'learning_rate': [], 'beta_values': [],
            'kl_mean': [], 'kl_std': [], 'kl_max': [],
            'epoch_times': [], 'best_epoch': 0, 'best_val_loss': float('inf'),
            'total_training_time': 0, 'stopped_early': False, 'stop_reason': 'completed'
        }

        # Setup directories
        model_dir = os.path.dirname(self.model)
        metrics_dir = os.path.dirname(self.metrics_path)
        for directory in [model_dir, metrics_dir]:
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        start_time = time.time()

        print("🧬 CVAE Training (scRNA-seq optimized)")
        print(f"📊 Beta: 0.000 → 0.05 → adaptive (KL target: 5-500)")
        print(f"📈 {n_epochs} epochs, patience={patience}")
        print("-" * 60)

        self.to(self.device)

        for epoch in range(n_epochs):
            epoch_start_time = time.time()

            # Smart beta annealing
            current_kl = history['kl_mean'][-1] if len(history['kl_mean']) > 0 else None
            current_beta = self.get_beta_annealed(epoch, n_epochs, current_kl)

            # Training
            train_metrics = self._train_epoch_with_kl_control(train_loader, optimizer, current_beta, epoch)
            current_lr = optimizer.param_groups[0]['lr']

            # Update history
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_recon'].append(train_metrics['recon_loss'])
            history['train_kl'].append(train_metrics['kl_loss'])
            history['learning_rate'].append(current_lr)
            history['beta_values'].append(current_beta)
            history['kl_mean'].append(train_metrics['kl_mean'])
            history['kl_std'].append(train_metrics['kl_std'])
            history['kl_max'].append(train_metrics['kl_max'])

            # MINIMAL monitoring - only show special cases
            self.monitor_training_progress(history, epoch)

            epoch_time = time.time() - epoch_start_time
            history['epoch_times'].append(epoch_time)

            # Validation
            if test_loader is not None:
                val_metrics = self._validate_epoch_with_kl_control(test_loader, current_beta)
                history['val_loss'].append(val_metrics['total_loss'])
                history['val_recon'].append(val_metrics['recon_loss'])
                history['val_kl'].append(val_metrics['kl_loss'])

                current_val_loss = val_metrics['total_loss']
                scheduler.step(current_val_loss)

                # Early stopping
                if current_val_loss < best_val_loss - min_delta:
                    best_val_loss = current_val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    history['best_epoch'] = best_epoch
                    history['best_val_loss'] = best_val_loss

                    if save_model:
                        torch.save(self.state_dict(), self.model)
                else:
                    patience_counter += 1

                # CLEAN progress display - just the essentials
                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    print(f"Epoch {epoch + 1:3d}/{n_epochs} | "
                          f"Train: {train_metrics['total_loss']:.4f} | "
                          f"Val: {current_val_loss:.4f} | "
                          f"Recon: {train_metrics['recon_loss']:.4f} | "
                          f"KL: {train_metrics['kl_mean']:.1f}")

                # Early stopping check
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping: No improvement for {patience} epochs")
                    history['stopped_early'] = True
                    history['stop_reason'] = 'patience_exceeded'
                    break

        # Clean completion summary
        total_time = time.time() - start_time
        history['total_training_time'] = total_time

        print("\n" + "=" * 60)
        print("✅ Training completed!")
        print(f"⏱️  Time: {total_time / 60:.1f}m | Best val: {best_val_loss:.4f} (epoch {best_epoch})")
        print(f"📊 Final KL: {history['kl_mean'][-1]:.1f} | Final β: {current_beta:.3f}")

        # Save metrics
        try:
            history_serializable = {}
            for key, value in history.items():
                if isinstance(value, list):
                    history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in
                                                 value]
                elif isinstance(value, (np.floating, np.integer)):
                    history_serializable[key] = float(value)
                else:
                    history_serializable[key] = value

            with open(self.metrics_path, 'w') as f:
                json.dump(history_serializable, f, indent=2)
            print(f"📈 Metrics saved to: {self.metrics_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save metrics: {e}")

        return history


    def plot_enhanced_training_history(self, history=None, save_path=None):
        """
        Plot comprehensive training curves with beta annealing and KL monitoring
        """
        if history is None:
            history = self.load_training_metrics()
            if history is None:
                print("❌ No training history available")
                return

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Total Loss (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history['train_loss'], label='Train', color='blue', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax1.plot(history['val_loss'], label='Validation', color='red', linewidth=2)
            if 'best_epoch' in history and history['best_epoch'] > 0:
                best_idx = history['best_epoch'] - 1
                if best_idx < len(history['val_loss']):
                    ax1.axvline(x=best_idx, color='green', linestyle='--', alpha=0.7, label='Best Model')
        ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Loss Components (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history['train_recon'], label='Reconstruction', color='blue', linewidth=2)
        ax2.plot(history['train_kl'], label='KL Divergence', color='orange', linewidth=2)
        if 'val_recon' in history and len(history['val_recon']) > 0:
            ax2.plot(history['val_recon'], label='Val Recon', color='lightblue', linewidth=1, alpha=0.7)
            ax2.plot(history['val_kl'], label='Val KL', color='gold', linewidth=1, alpha=0.7)
        ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. ⭐ BETA ANNEALING SCHEDULE (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'beta_values' in history and len(history['beta_values']) > 0:
            ax3.plot(history['beta_values'], color='purple', linewidth=3, marker='o', markersize=3)
            ax3.set_title('Beta Annealing Schedule', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Beta Value')
            ax3.grid(True, alpha=0.3)

            # Add annotations for key phases
            if len(history['beta_values']) > 30:
                warmup_end = len(history['beta_values']) * 0.3
                ax3.axvline(x=warmup_end, color='red', linestyle=':', alpha=0.7, label='Warmup End')
                ax3.text(warmup_end + 2, max(history['beta_values']) * 0.8,
                         'Warmup\nPhase', fontsize=10, alpha=0.7)

            # Show current beta value
            final_beta = history['beta_values'][-1]
            ax3.text(0.7, 0.9, f'Final β: {final_beta:.4f}',
                     transform=ax3.transAxes, fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='purple', alpha=0.1))
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Beta values not tracked\nin this training run',
                     transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Beta Annealing (Not Available)', fontsize=14, fontweight='bold')

        # 4. ⭐ KL DIVERGENCE MONITORING (second row left)
        ax4 = fig.add_subplot(gs[1, 0])
        if 'kl_mean' in history and len(history['kl_mean']) > 0:
            epochs = range(len(history['kl_mean']))

            # Plot KL mean with error bars if std is available
            if 'kl_std' in history and len(history['kl_std']) > 0:
                kl_mean = np.array(history['kl_mean'])
                kl_std = np.array(history['kl_std'])
                ax4.plot(epochs, kl_mean, color='red', linewidth=2, label='KL Mean')
                ax4.fill_between(epochs, kl_mean - kl_std, kl_mean + kl_std,
                                 color='red', alpha=0.2, label='±1 Std')
            else:
                ax4.plot(history['kl_mean'], color='red', linewidth=2, label='KL Mean')

            # Add healthy KL range
            ax4.axhspan(1, 100, color='green', alpha=0.1, label='Healthy Range')
            ax4.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Upper Bound')
            ax4.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Lower Bound')

            ax4.set_title('KL Divergence Monitoring', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('KL Divergence')
            ax4.set_yscale('log')  # Log scale for better visualization
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Add final KL status
            final_kl = history['kl_mean'][-1]
            kl_status = "🟢 Healthy" if 1 <= final_kl <= 100 else "🔴 Problematic"
            ax4.text(0.7, 0.9, f'Final KL: {final_kl:.1f}\n{kl_status}',
                     transform=ax4.transAxes, fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightgreen' if '🟢' in kl_status else 'lightcoral', alpha=0.3))
        else:
            ax4.text(0.5, 0.5, 'KL statistics not available\nin this training run',
                     transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('KL Monitoring (Not Available)', fontsize=14, fontweight='bold')

        # 5. Learning Rate Schedule (second row middle)
        ax5 = fig.add_subplot(gs[1, 1])
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            ax5.plot(history['learning_rate'], color='orange', linewidth=2)
            ax5.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Learning Rate')
            ax5.set_yscale('log')
            ax5.grid(True, alpha=0.3)

            # Mark LR reductions
            lr_values = np.array(history['learning_rate'])
            reductions = np.where(np.diff(lr_values) < -1e-6)[0]  # Find significant drops
            for reduction_idx in reductions:
                ax5.axvline(x=reduction_idx, color='red', linestyle=':', alpha=0.7)

            if len(reductions) > 0:
                ax5.text(0.7, 0.9, f'LR reductions: {len(reductions)}',
                         transform=ax5.transAxes, fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.1))

        # 6. Beta vs KL Relationship (second row right)
        ax6 = fig.add_subplot(gs[1, 2])
        if ('beta_values' in history and 'kl_mean' in history and
                len(history['beta_values']) > 0 and len(history['kl_mean']) > 0):
            # Create scatter plot colored by epoch
            epochs = range(min(len(history['beta_values']), len(history['kl_mean'])))
            betas = history['beta_values'][:len(epochs)]
            kls = history['kl_mean'][:len(epochs)]

            scatter = ax6.scatter(betas, kls, c=epochs, cmap='viridis', s=30, alpha=0.7)
            ax6.set_title('Beta vs KL Relationship', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Beta Value')
            ax6.set_ylabel('KL Divergence')
            ax6.set_yscale('log')
            ax6.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax6)
            cbar.set_label('Epoch')

            # Add ideal region
            ax6.axhspan(1, 100, color='green', alpha=0.1, label='Target KL')
            ax6.legend()

        # 7. Epoch Timing (third row left)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'epoch_times' in history and len(history['epoch_times']) > 0:
            ax7.plot(history['epoch_times'], color='green', linewidth=2)
            ax7.axhline(y=np.mean(history['epoch_times']), color='red', linestyle='--',
                        alpha=0.7, label=f'Mean: {np.mean(history["epoch_times"]):.1f}s')
            ax7.set_title('Epoch Training Time', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Time (seconds)')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # 8. Training Progress Summary (third row middle)
        ax8 = fig.add_subplot(gs[2, 1])
        if 'val_loss' in history and len(history['val_loss']) > 0:
            # Train vs Val loss correlation
            min_len = min(len(history['train_loss']), len(history['val_loss']))
            train_subset = history['train_loss'][:min_len]
            val_subset = history['val_loss'][:min_len]

            ax8.scatter(train_subset, val_subset, alpha=0.6, c=range(min_len),
                        cmap='viridis', s=30)

            # Add diagonal line (perfect correlation)
            min_loss = min(min(train_subset), min(val_subset))
            max_loss = max(max(train_subset), max(val_subset))
            ax8.plot([min_loss, max_loss], [min_loss, max_loss], 'r--', alpha=0.7,
                     label='Perfect Correlation')

            ax8.set_title('Train vs Validation Loss', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Training Loss')
            ax8.set_ylabel('Validation Loss')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        # 9. Training Summary (bottom row spans all)
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')

        # Create comprehensive summary text
        summary_text = f"""
        ENHANCED CVAE TRAINING SUMMARY
        ===============================
        Architecture: {getattr(self, 'n_genes', 'N/A')}+{getattr(self, 'n_labels', 'N/A')} → [800,400] → {getattr(self, 'latent_size', 'N/A')}
        Total Epochs: {len(history['train_loss'])}
        Training Time: {history.get('total_training_time', 0) / 60:.1f} minutes
        Early Stopping: {'Yes' if history.get('stopped_early', False) else 'No'}
        Stop Reason: {history.get('stop_reason', 'completed')}
    
        FINAL METRICS
        =============
        Final Train Loss: {history['train_loss'][-1]:.6f}
        Final Recon Loss: {history['train_recon'][-1]:.6f}
        Final KL Loss: {history['train_kl'][-1]:.6f}
        """

        if 'val_loss' in history and len(history['val_loss']) > 0:
            summary_text += f"""
        Final Val Loss: {history['val_loss'][-1]:.6f}
        Best Val Loss: {history.get('best_val_loss', 'N/A'):.6f} (Epoch {history.get('best_epoch', 'N/A')})
        """

        if 'beta_values' in history and len(history['beta_values']) > 0:
            summary_text += f"""
        Final Beta: {history['beta_values'][-1]:.4f}
        Beta Range: {min(history['beta_values']):.4f} → {max(history['beta_values']):.4f}
        """

        if 'kl_mean' in history and len(history['kl_mean']) > 0:
            final_kl = history['kl_mean'][-1]
            kl_healthy = "🟢 HEALTHY" if 1 <= final_kl <= 100 else "🔴 NEEDS ATTENTION"
            summary_text += f"""
        Final KL Divergence: {final_kl:.1f} ± {history.get('kl_std', [0])[-1]:.1f} {kl_healthy}
        KL Range: {min(history['kl_mean']):.1f} → {max(history['kl_mean']):.1f}
        """

        if 'epoch_times' in history and len(history['epoch_times']) > 0:
            summary_text += f"""
        Avg Epoch Time: {np.mean(history['epoch_times']):.1f}s
        Total Training Time: {sum(history['epoch_times']) / 60:.1f} minutes
        """

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.suptitle('Enhanced CVAE Training Analysis with Beta Annealing & KL Control',
                     fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enhanced training analysis plot saved to: {save_path}")

        plt.tight_layout()
        plt.show()


    # ALSO ADD THIS METHOD FOR REAL-TIME BETA AND KL MONITORING:

    def monitor_training_progress(self, history, epoch):
        """
        Minimal training monitoring - only show important info at key moments
        """
        if len(history['beta_values']) == 0 or len(history['kl_mean']) == 0:
            return

        current_beta = history['beta_values'][-1]
        current_kl = history['kl_mean'][-1]

        # Only print in these specific cases:
        show_info = False
        message = ""

        # 1. Warmup phase - show beta progress occasionally
        total_epochs = 120  # Adjust to your actual total
        warmup_end = int(total_epochs * 0.25)

        if epoch <= warmup_end and epoch % 20 == 0:
            progress = (epoch / warmup_end) * 100
            show_info = True
            message = f"🔥 Beta warmup: {progress:.0f}% complete (β={current_beta:.3f})"

        # 2. End of warmup phase
        elif epoch == warmup_end:
            show_info = True
            message = f"✅ Warmup complete! Final β={current_beta:.3f}, KL={current_kl:.1f}"

        # 3. Only show KL issues if they're really bad
        elif current_kl > 1000:  # Much higher threshold
            show_info = True
            message = f"⚠️  KL very high: {current_kl:.0f} (β={current_beta:.3f})"

        # 4. Posterior collapse detection
        elif current_kl < 1 and epoch > warmup_end:
            show_info = True
            message = f"⚠️  Possible collapse: KL={current_kl:.2f} (β={current_beta:.3f})"

        if show_info:
            print(f"   {message}")