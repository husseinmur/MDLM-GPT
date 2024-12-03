# Language Models for Molecular Dynamics

This repository contains the official implementation of the paper [Language Models for Molecular Dynamics](https://www.biorxiv.org/content/10.1101/2024.11.25.625337v1) (bioRxiv 2024).

## Overview
This codebase implements Molecular Dynamics Language Models (MDLMs), a novel approach that uses GPT-J model to explore the conformational space of Chignolin. The model is trained on a short classical MD trajectory and it maintains structural accuracy through kernel density estimations derived from extensive MD datasets.

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

# Install requirements
pip install -r requirements.txt

# Download KDE functions
wget https://zenodo.org/records/14263500/files/kde_functions_from_md.pkl -O KDE/kde_functions_from_md.pkl
```

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
