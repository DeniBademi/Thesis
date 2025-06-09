# Neural Network Training Framework

This project contains the proceedings of my Bachelor Thesis at Radboud University "Spike Threshold Adaptive Learning for
Image Classification with SNNs". It contains the official implementation of ConvSTAL and STAL-PS, two encoder networks that convert images to binary spike trains (inspired by [STAL](https://arxiv.org/pdf/2407.08362)). It includes

Implementations of:

Classifiers: 
-  __Spiking Recurrent Neural Network__ (proposed [here](https://arxiv.org/pdf/2407.08362))
- __Spiking Vision Transformer__ (proposed [here](https://arxiv.org/abs/2209.15425))


Encoders:
- Rate Coding
- Latency Coding
- STAL
- ConvSTAL
- STAL-PS

Loss functions:
- Encoder loss (proposed [here](https://arxiv.org/pdf/2407.08362))
- MultiChannelEncoderLoss
- SpikeSparsityLoss (SS)
- Cross Entropy+SS

Experiments:
- Training of STAL with Encoder and MultiChannelEncoder losses
- Training of RSNN with different encoders (Rate Coding, Latency Coding, STAL, ConvSTAL)
- Training of Spikformer with different encoders (SPS, STAL-PS)

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
├── experiments/           # Experiment results
└── notebooks/            # Jupyter notebooks
```

## Training Models

### Basic Training

To train a model, use the following command:

```bash
python src/train.py --config config/<config_name>.yaml
```

Available model configurations:
- `train_spikformer.yaml`: Spikformer model
- `train_rsnn_convstal.yaml`: SRNN with ConvSTAL encoder
- `train_rsnn_stal.yaml`: SRNN with STAL encoder
- `train_conv_stal.yaml`: ConvSTAL model
- `train_stal.yaml`: STAL model
- `train_rsnn_latency_rate.yaml`: SRNN with rate/latency encoding

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
- Data loading and preprocessing
- Logging configuration

## Logging

The framework supports two logging backends:
1. Weights & Biases (wandb)
2. MLFlow

Configure logging in the config file:
```yaml
training:
  logger:
    name: wandb  # or mlflow
    save_dir: ./logs
    project: your_project_name
```