# Multi-Conditional scGAN - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Set Up Environment
```bash
# Use the existing environment
source ~/.scgan_venv_home/bin/activate

# OR create new environment if needed
python3 -m venv scgan_env
source scgan_env/bin/activate
pip install tensorflow==2.13.0 numpy scipy pandas h5py anndata scanpy scikit-learn matplotlib seaborn
```

### 2. Verify Installation
```bash
# Check structure is intact
python multi_cond/test_simple.py

# Test models with real data
python multi_cond/test_small.py
```

### 3. Start Training
```bash
python multi_cond/train.py
```

## 📊 Monitor Progress
```bash
# In a new terminal
tensorboard --logdir multi_cond/logs
# Open browser to http://localhost:6006
```

## 🧬 Generate Synthetic Cells

### Generate with matched conditions (recommended)
```bash
python multi_cond/generate.py \
    --checkpoint multi_cond/checkpoints/latest \
    --n_cells 10000 \
    --match_conditions true
```

### Generate with specific conditions
```bash
python multi_cond/generate.py \
    --checkpoint multi_cond/checkpoints/latest \
    --n_cells 1000 \
    --strain "BALB/c" \
    --age_at_launch 12.0 \
    --duration 41.0 \
    --flight 0
```

## 📁 Key Files

- **Data**: `data/raw_0.997_filtered_hvg_2000_augmented_combined_norm.h5ad`
- **Config**: `multi_cond/configs/parameters.json`
- **Models**: `multi_cond/models/generator.py`, `multi_cond/models/critic.py`
- **Training**: `multi_cond/train.py`
- **Generation**: `multi_cond/generate.py`

## ⚙️ Configuration

Edit `multi_cond/configs/parameters.json` to adjust:
- `batch_size`: 128 (reduce if OOM)
- `learning_rate_g`: 0.0001 (generator)
- `learning_rate_c`: 0.0004 (critic)
- `epochs`: 10000 (with early stopping)
- `checkpoint_freq`: 50

## 🔧 Troubleshooting

### Out of Memory
```json
{
  "training": {
    "batch_size": 64
  }
}
```

### TensorFlow Not Found
```bash
pip install tensorflow==2.13.0
# OR for GPU:
pip install tensorflow[and-cuda]==2.13.0
```

### Slow Training
- Ensure GPU is available: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Reduce model size in config
- Use smaller subset for testing

## 📈 Expected Results

- **Training time**: 2-3 hours (GPU), 24+ hours (CPU)
- **Convergence**: Usually by epoch 500-800
- **Quality check**: Generated cells should cluster with real cells of same condition
- **File size**: ~2GB per 100k generated cells

## 🎯 Conditions

The model handles 4 conditions:
- **Strain**: BALB/c, B6129SF2/J, C57BL/6NTac
- **Age at Launch**: 0, 12, 14 weeks
- **Duration**: 28, 39, 41, 42, 53 days
- **Flight**: 0 (ground), 1 (spaceflight)

## 💡 Tips

1. Start with small batches to verify setup
2. Monitor tensorboard for training stability
3. Save checkpoints frequently (every 50 epochs)
4. Generate small test batches before large-scale generation
5. Validate generated data with standard scRNA-seq QC tools

## 📖 More Information

See `IMPLEMENTATION_SUMMARY.md` for detailed technical documentation.