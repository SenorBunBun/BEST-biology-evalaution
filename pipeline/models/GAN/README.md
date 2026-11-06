# Multi-Conditional scGAN

A multi-conditional GAN for generating single-cell RNA-seq data with multiple mixed conditions (categorical and continuous).

## Features

- Supports mixed condition types (categorical, continuous, binary)
- Conditional batch normalization for generator
- Projection method for critic
- WGAN-GP training framework
- Direct h5ad file loading
- Comprehensive condition evaluation

## Installation

```bash
# Activate virtual environment
source ~/.scgan_venv_home/bin/activate

# For GPU support, set up CUDA environment
source ~/setup_cuda_env.sh
```

## Quick Start

### 1. Training

#### Basic Training
```bash
# Start new training with default parameters
python multi_cond/train.py

# Start training with custom configuration
python multi_cond/train.py --config multi_cond/configs/parameters.json

# Start training with custom run directory
python multi_cond/train.py --run-dir multi_cond/checkpoints/experiment1
```

#### Resume Training
```bash
# Resume from best model checkpoint
python multi_cond/train.py \
    --resume multi_cond/checkpoints/learning_rate_adj/best_model

# Resume from specific checkpoint
python multi_cond/train.py \
    --resume multi_cond/checkpoints/run_20250926_141822/checkpoint_step_10000

# Resume with different configuration
python multi_cond/train.py \
    --resume multi_cond/checkpoints/experiment1/best_model \
    --config multi_cond/configs/parameters_gp_fix.json
```

### 2. Generation

#### Basic Generation
```bash
# Generate synthetic data matching reference distribution
python multi_cond/generate.py \
    --checkpoint multi_cond/checkpoints/learning_rate_adj/best_model \
    --scale 1.0

# Generate with 50% of reference cell count
python multi_cond/generate.py \
    --checkpoint multi_cond/checkpoints/learning_rate_adj/best_model \
    --scale 0.5 \
    --output synthetic_half.h5ad

# Generate with custom reference data
python multi_cond/generate.py \
    --checkpoint multi_cond/checkpoints/learning_rate_adj/best_model \
    --reference data/my_reference.h5ad \
    --scale 2.0 \
    --output synthetic_double.h5ad
```

### 3. Condition Evaluation

The condition evaluation script tests how well the model learns and respects conditions through 4 comprehensive tests:

#### Basic Evaluation
```bash
# Evaluate best model
python multi_cond/evaluate_conditions.py \
    --checkpoint multi_cond/checkpoints/learning_rate_adj/best_model

# Evaluate specific checkpoint with output directory
python multi_cond/evaluate_conditions.py \
    --checkpoint multi_cond/checkpoints/run_20250926_141822/checkpoint_step_10000 \
    --output multi_cond/evaluation_reports/

# Evaluate with custom reference data
python multi_cond/evaluate_conditions.py \
    --checkpoint multi_cond/checkpoints/experiment1/best_model \
    --reference data/test_reference.h5ad \
    --config multi_cond/configs/parameters_gp_fix.json
```

#### What It Tests:
1. **Consistency**: Generates multiple samples with same conditions, checks variance
2. **Discrimination**: Tests if critic can distinguish real vs fake per condition
3. **Specificity**: Verifies different conditions produce different outputs
4. **Distribution**: Compares generated vs real expression distributions

#### Example Output:
```
TEST 1: Condition Consistency
  Mean variance: 0.234 ✅ (low = good)

TEST 2: Condition Discrimination
  Mean score: -0.8 ✅ (near zero = good)

TEST 3: Condition Specificity
  Flight responsive genes: 324/2000 (16.2%) ✅
  Age responsive genes: 287/2000 (14.4%) ✅

TEST 4: Distribution Matching
  Mean KS statistic: 0.21 ✅ (low = good)

Overall Quality: 4/4
🎉 Excellent condition learning!
```

### 4. Monitoring Training

#### TensorBoard
```bash
# View training metrics
tensorboard --logdir multi_cond/logs

# View specific run
tensorboard --logdir multi_cond/logs/learning_rate_adj
```

#### Check Training Status
```bash
# List all checkpoints
ls -la multi_cond/checkpoints/*/

# Check latest checkpoint
ls -t multi_cond/checkpoints/*/checkpoint_step_* | head -1

# View training state
cat multi_cond/checkpoints/learning_rate_adj/best_model/training_state.json
```

## Configuration Files

### Available Configurations
- `parameters.json` - Default parameters
- `parameters_gp_fix.json` - Optimized for gradient penalty issues
- `parameters_balanced.json` - Balanced training with GP~0.35

### Key Parameters
```json
{
  "data": {
    "batch_size": 64,          // Batch size for training
    "train_split": 0.85        // Train/validation split
  },
  "model": {
    "latent_dim": 128,         // Noise vector dimension
    "lambda_gp": 10.0          // Gradient penalty weight
  },
  "training": {
    "max_steps": 300000,       // Maximum training steps
    "critic_iters": 5,         // Critic iterations per generator update
    "learning_rate_g": 0.0002, // Generator learning rate
    "learning_rate_c": 0.0002, // Critic learning rate
    "save_freq": 1000          // Checkpoint frequency
  }
}
```

## Understanding Loss Values

### Ideal Ranges
- **Generator Loss**: -2 to -8 (negative is good)
- **Critic Loss**: -1 to -5 (negative shows discrimination)
- **Gradient Penalty**: 0.5 to 1.5 (ideally near 1.0)

### Example Good Values
- C_loss: -2.7, G_loss: -5.9 → Excellent balance
- C_loss: -3.1, G_loss: -0.5 → Good convergence

## Advanced Usage

### Custom Condition Generation
```python
from multi_cond.generate import SyntheticDataGenerator

# Load generator
gen = SyntheticDataGenerator('multi_cond/checkpoints/best_model')
gen.load_model()
gen.load_reference_data()

# Generate with specific conditions
conditions = [
    {'Strain': 'C57BL/6', 'Age at Launch': 32.0, 'Duration': 30.0, 'Flight': 1},
    {'Strain': 'BALB/c', 'Age at Launch': 16.0, 'Duration': 60.0, 'Flight': 0}
]
synthetic = gen.generate_custom(conditions, n_cells_per_condition=100)
gen.save_synthetic_data(synthetic, 'custom_conditions.h5ad')
```

## Troubleshooting

### Memory Issues
- Reduce batch_size in config (32 or 16)
- Use gradient accumulation
- Clear GPU cache between runs

### Training Instability
- Check gradient penalty (should be 0.3-1.5)
- Reduce learning rates
- Increase critic_iters if generator dominates
- Decrease critic_iters if critic dominates

### Poor Generation Quality
- Run condition evaluation to identify issues
- Check if losses are balanced (both negative)
- Ensure sufficient training (>50 epochs)
- Verify data preprocessing is correct

## File Structure

```
multi_cond/
├── configs/              # Configuration files
├── models/              # Model architectures
├── utils/               # Utility functions
├── checkpoints/         # Saved models
├── logs/               # TensorBoard logs
├── generated/          # Generated samples
├── train.py            # Main training script
├── generate.py         # Generation script
└── evaluate_conditions.py  # Condition evaluation
```

## Citation

If you use this code, please cite:
- Original scGAN paper
- Your NASA spaceflight publication

## Support

For issues or questions, check the training logs and TensorBoard metrics first.