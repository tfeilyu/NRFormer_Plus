# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Training Models
```bash
# Train a model with specific parameters
python train.py --model_des <model_description> --dataset <dataset_name> --epochs 300 --model_name <model_name>

# Quick training with random model description (using go.sh)
bash go.sh

# Example training commands
python train.py --model_des 111 --dataset '1D-data' --model_name 'NRFormer_Plus' --epochs 300
```

### Testing Models
```bash
# Test a trained model
python test.py --model_des <model_description> --dataset <dataset_name> --model_name <model_name>

# Test with sudden change analysis
python test.py --sudden_change True --model_des 123 --dataset '1D-data' --model_name 'NRFormer_Plus_v3'
```

### Hyperparameter Search (New)
```bash
# Quick hyperparameter testing (recommended for initial exploration)
python quick_hyperparameter_test.py --test_type architecture --epochs 50

# Systematic hyperparameter search
python hyperparameter_search.py --search_type important_only --max_combinations 20 --epochs 150

# Full experimental suite (for TKDE paper)
bash run_hyperparameter_experiments.sh

# Ablation studies
python quick_hyperparameter_test.py --test_type ablation --epochs 100
```

### Available Models
- `NRFormer`: **KDD 2025 published** base transformer model for nuclear radiation forecasting
- `NRFormer_Plus`: **TKDE submission** enhanced version with physics-guided components (PGRT2 class)
- Model variants: `NRFormer_Plus_v1`, `NRFormer_Plus_v2`, `NRFormer_Plus_v3`, `NRFormer_Plus_v4`, `NRFormer_Plus_v5`

## Architecture Overview

### Core Structure
This is a **nuclear radiation spatiotemporal forecasting system** using transformer-based neural networks with physics-guided components.

**Key Components:**
- **Data Processing** (`src/DataProcessing.py`): Handles radiation sensor data preprocessing and creates adjacency matrices for spatial relationships
- **Model Architecture** (`src/model/`): Two main model families
  - `NRFormer.py`: Base attention-based spatiotemporal transformer
  - `NRFormer_Plus.py`: Physics-Guided Radiation Transformer (PGRT2) with atmospheric diffusion modeling
- **Training Engine** (`src/trainer.py`): Handles model training, validation, and optimization
- **Utilities** (`src/utils.py`): Metrics, data scaling, and helper functions

### Data Flow
1. **Input**: Spatiotemporal radiation sensor data (3627 sensors, 24 time steps)
2. **Processing**: Creates spatial adjacency matrices based on sensor distances, applies normalization
3. **Model**: Transformer-based architectures with spatial-temporal attention mechanisms
4. **Output**: Multi-step ahead radiation level predictions (up to 24 steps)

### Configuration System
Models are configured via YAML files in `model_settings/`:
- `NRFormer.yaml`: Base model configuration
- `NRFormer_Plus.yaml`: Enhanced model configuration

Key hyperparameters include batch size, learning rates, early stopping, spatial/temporal attention layers, and meteorological feature flags.

### Physics-Guided Components (NRFormer_Plus)
- **Atmospheric diffusion modeling**: Physics-informed neural network components
- **Meteorological integration**: Wind, temperature, humidity data incorporation
- **Multi-scale spatial modeling**: Local and global radiation propagation pathways
- **RevIN normalization**: Reversible instance normalization for temporal stability

### Training Features
- **Weights & Biases integration**: Automatic experiment tracking and model artifacts
- **Early stopping**: Configurable patience-based training termination
- **Multi-run experiments**: Statistical significance testing across multiple runs
- **Model checkpointing**: Automatic best model saving in `logs/` directory

### Evaluation Metrics
- **MAE** (Mean Absolute Error): Primary metric for model performance
- **MAPE** (Mean Absolute Percentage Error): Relative error measurement  
- **RMSE** (Root Mean Square Error): Penalizes large prediction errors
- **Multi-horizon evaluation**: Performance across different prediction horizons (3, 6, 9, 12, 24 steps)

## Research Context & Hyperparameter Tuning

### Publication Status
- **NRFormer**: Published at KDD 2025 as the base model
- **NRFormer_Plus**: Extended version prepared for TKDE submission with enhanced physics-guided components

### Key Configuration Changes
The default `model_name` in `train.py` has been updated to `'NRFormer_Plus'` (line 28) to focus on the TKDE submission model.

### Hyperparameter Search Tools
Three comprehensive tools are available for systematic hyperparameter optimization:

1. **`hyperparameter_search.py`**: Core search engine with grid/random/important parameter search modes
2. **`run_hyperparameter_experiments.sh`**: Automated experimental suite covering:
   - Important parameter screening
   - Architecture depth search  
   - Physics feature ablation studies
   - Training hyperparameter optimization
3. **`quick_hyperparameter_test.py`**: Lightweight tool for rapid configuration testing

### Search Space Coverage
- **Architecture**: Hidden dimensions, attention layers, MLP configurations
- **Physics Features**: Wind angle/speed, temperature, dew point, RevIN ablation
- **Training**: Learning rates, batch sizes, weight decay, early stopping
- **Temporal Features**: Time/day/month embedding combinations

All experiments automatically log results to JSON/CSV formats with Weights & Biases integration for systematic analysis supporting TKDE publication requirements.