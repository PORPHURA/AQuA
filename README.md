# Autonomous Quality and Hallucination Assessment for Virtual Tissue Staining and Digital Pathology
Luzhe Huang, Yuzhu Li, Nir Pillar, Tal Keidar Haran, William Dean Wallace, Aydogan Ozcan. *Nature Biomedical Engineering*, 2025

This code repository includes the testing codes, pre-trained models and demo data for autonomous quality and hallucination assessment (AQuA).

This version of code repository is for review purpose only. Upon acceptance, the training codes and more demo data will be uploaded, and this repository will also be open-sourced in GitHub.

## Installation
The codes were composed and tested on Linux and Windows platforms with the following software environment:
- Python 3.9.16
- CUDA 12.2
- PyTorch 1.13.0
- the rest packages listed in `requirements.txt`
After setting up the environment, there is no additional installation steps. The codes are ready to run.

## Reproduction
About the dataset preparation for VS and VAF models, refer to the Methods section. The full training codes for VS models are provided in our previous work https://github.com/liyuzhu1998/Autopsy-Virtual-Staining/tree/main

### AQUA model
- `./functions.py` includes basic components of the ResNet backbone and classification head in AQuA-Net.
- `./ckpts` includes C=5 pre-trained ensembles with T=5 for human lung tissue samples. Checkpoints can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ztfS6hTkyU-mUrXuHI6LAgUFJ1-hcEyY?usp=share_link)

### Demo data
- `./demo_data/test_vs_54` and `./demo_data/test_vs_1098` contain cyclic inference results of a poor-staining and a good-staining VS models, respectively. Demo data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ztfS6hTkyU-mUrXuHI6LAgUFJ1-hcEyY?usp=share_link)

### Test
Run
```
python test.py
```
to test AQuA ensembles and generate predictions for each VS images in the demo data.
The predictions will be saved to `./predictions`.
