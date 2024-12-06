# Language Models for Molecular Dynamics
This repository contains the official implementation of the paper [Language Models for Molecular Dynamics](https://www.biorxiv.org/content/10.1101/2024.11.25.625337v1) (bioRxiv 2024).

## Overview
This codebase implements Molecular Dynamics Language Models (MDLMs), a novel approach that uses GPT-J model to explore the conformational space of Chignolin. The model is trained on a short classical MD trajectory and maintains structural accuracy through kernel density estimations derived from extensive MD datasets.

## System Requirements
### Hardware Requirements
- GPU with minimum 8GB VRAM
- Tested on Intel® Core™ i7-12700H CPU, NVIDIA GeForce RTX 4060 Laptop GPU
- 16GB RAM recommended
- 15GB free disk space

### Software Requirements
- Python 3.8+
- PyTorch 2.4+
- CUDA 12.7+

### Operating Systems
- Ubuntu 20.04 or later
- macOS 12.0 or later
- Windows 10/11 with WSL2

Typical install time: 10-15 minutes

## Repository Structure
```
├── dataset/                     # Dataset directory
│   ├── train.txt               # Training data
│   └── valid.txt               # Validation data
├── inference/                   # Inference code
│   ├── pretrained_model/       # Pretrained model weights
│   ├── angle_utils.py          # Angle processing utilities
│   ├── model_setup.py          # Model initialization
│   └── sampler.py              # Sampling implementation
├── KDE/                        # Kernel Density Estimation
│   ├── KDE220.py
│   └── low_memory_kde_functions.pkl
│   └── kde_functions_from_md.pkl  # Download from Zenodo
├── model/                      # Model files
│   ├── custom_trainer.py       # Custom training implementation
│   ├── data_processing.py      # Data processing utilities
│   ├── helpers.py             
│   └── model_config.py         # Model configuration
├── tokenizer/                  # Tokenizer files
│   └── vocab.txt
├── utils/                      # Utility functions
│   ├── training_utils.py
│   ├── config.py
│   └── generate.py
└── train.py                    # Main training script
```

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mdlm.git
cd mdlm

# Create and activate virtual environment (recommended)
python -m venv mdlm_env
source mdlm_env/bin/activate  # On Windows use: mdlm_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Download KDE functions
wget https://zenodo.org/records/14263500/files/kde_functions_from_md.pkl -O KDE/kde_functions_from_md.pkl
```

## Demo Data
The `dataset` directory contains example tokenized trajectory data:
- `train.txt`: Training set of tokenized conformations
- `valid.txt`: Validation set of tokenized conformations

Each line represents one protein conformation, where tokens follow the format:
`XyyZ` where:
- X, Z are amino acid identities
- yy represents discretized φ-ψ angles between residues
- `.` marks the end of each conformation

Example: `GaeY YdnD DfnP PemE EdgT TkhG GciT TbeW WnaG .`
represents a complete conformation of Chignolin.
Expected runtime with demo data: ~14 hours for training and inference

## Usage
### Training
To train the model:
```bash
python train.py --tokenizer ./tokenizer/ \
                --train_data ./dataset/train.txt \
                --valid_data ./dataset/valid.txt \
                --output_dir ./output
```

### Inference
To use the pretrained weights and generate conformations for Chignolin:
```bash
python generate.py 
```
This will generate conformations saved as a NumPy file (.npy) with shape (X, 2, 9), where:
- X is the number of generated conformations
- 2 represents φ and ψ angles
- 9 represents the number of angle pairs in Chignolin

## Reproducibility
To reproduce the results from the paper:
1. Train the model using provided demo data (~6 hours)
2. Run inference for sampling (~8 hours)
3. Generated conformations will be saved as an npy file as described in the inference section.

## Citation
If you use this code in your research, please cite:
```bibtex
@article{Murtada2024.11.25.625337,
    author = {Murtada, Mhd Hussein and Brotzakis, Z. Faidon and Vendruscolo, Michele},
    title = {Language Models for Molecular Dynamics},
    elocation-id = {2024.11.25.625337},
    year = {2024},
    doi = {10.1101/2024.11.25.625337},
    publisher = {Cold Spring Harbor Laboratory},
    journal = {bioRxiv}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Contact
For questions about the code, please open an issue.
