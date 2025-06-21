# Spike Threshold Adaptive Learning for Image Classification with SNNs

This project contains the proceedings of my Bachelor Thesis at Radboud University. It contains the official implementation of ConvSTAL and ConvSTAL-PS, two encoder networks that convert images to binary spike trains (inspired by [STAL](https://arxiv.org/pdf/2407.08362)).

Classifiers: 
-  __Spiking Recurrent Neural Network__ (proposed [here](https://arxiv.org/pdf/2407.08362))
- __Spiking Vision Transformer__ (proposed [here](https://arxiv.org/abs/2209.15425))

Encoders:
- Rate Coding
- Latency Coding
- STAL
- ConvSTAL
- ConvSTAL-PS

Loss functions:
- Encoder loss (proposed [here](https://arxiv.org/pdf/2407.08362))
- SpikeSparsityLoss (SS)
- Cross-Entropy + Spike Sparsity (CE+SS) loss

Experiments:
- Training of STAL with Encoder and MultiChannelEncoder losses
- Training of RSNN with different encoders (Rate Coding, Latency Coding, STAL, ConvSTAL)
- Training of Spikformer with different encoders (SPS, ConvSTAL-PS)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Or using conda
conda create -n myenv python=3.12
conda activate myenv
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── config/                 # Configuration files for different experiments described in the thesis
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── training/          # Training utilities
│   └── data/              # Data loading and processing
├── weights/               # Saved model weights
├── experiments/           # Log files of wandb and MLflow
└── notebooks/            # Jupyter notebooks
```

## Training Models

### Basic Training

To train a model, use the following command:

```bash
python src/train.py --config config/<config_name>.yaml
```

Available model configurations:
- `config/spikformer_experiment/`: Configuration files related to the experiment comparing ConvSTAL-PS against SPS
- `config/srnn_experiment/`:Configuration files related to the experiment comparing ConvSTAL against Rate coding and STAL

### Hyperparameter Optimization

To perform hyperparameter optimization:

```bash
python src/training/utils/hyperparameter_search.py --config config/optimize.yaml
```

This will optimize the omega and concentration weight parameters for the CE_SpikeSparsityLoss function.

### Command Line Arguments

You can override configuration parameters using command line arguments:

```bash
python src/train.py --config config/train_spikformer.yaml \
    --model.learning_rate 0.001 \
    --training.batch_size 32 \
    --training.epochs 100
```

## Configuration Files

Configuration files are in YAML format and contain settings for:
- Model architecture and parameters
- Training hyperparameters
- Data loading
- Logging configuration

## Logging

The framework supports two logging backends:
1. Weights & Biases (wandb) (main)
2. MLFlow

Configure logging in the config file:
```yaml
training:
  logger:
    name: wandb  # or mlflow
    save_dir: ./logs
    project: your_project_name
```

To use MLflow, you need to first initialize it with the following commant
```bash 
mlflow server --backend-store-uri sqlite:///<ROOT_DIR>\experiments\mlflow\mlflow.db --default-artifact-root ./mlflow/artifacts --host 127.0.0.1 
```