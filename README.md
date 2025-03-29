 # Autonomous Quality and Hallucination Assessment for Virtual Tissue Staining and Digital Pathology
Luzhe Huang, Yuzhu Li, Nir Pillar, Tal Keidar Haran, William Dean Wallace, Aydogan Ozcan, *Nature Biomedical Engineering*, 2025

This code repository includes the training and testing codes for autonomous quality and hallucination assessment (AQuA).

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
- `./ckpts` includes C=5 pre-trained ensembles with T=5 for human lung tissue samples.

### Demo data
- `./demo_data/test_vs_54` and `./demo_data/test_vs_1098` contain cyclic inference results of a poor-staining and a good-staining VS models, respectively.

Checkpoints and demo data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ztfS6hTkyU-mUrXuHI6LAgUFJ1-hcEyY?usp=share_link). After downloading, place them into corresponding folders.

### Train
Run
```
python train.py
```
to train an AQUA model. The dataset needs to be organized in the following way:
- training positive sample folder: *.mat, ...
- training negative sample folder: *.mat, ...
- validation positive sample folder: *.mat, ...
- validation negative sample folder: *.mat, ...

Each .mat file should contain the following variables of cyclic inference results:
- he_outputs: [T+1, 3, H, W], cyclic inference in HE domain,
- dapi_outputs: [T+1, 1, H, W], cyclic inference in DAPI domain, the first frame is measurement
- tissue_masks: [T+1, H, W], tissue mask from HE images
- nuclei_masks: [T+1, H, W], nuclei mask from HE images

Here T is the maximum cycle number, H and W are image dimensions.

### Test
Run
```
python test.py
```
to test AQuA ensembles and generate predictions for each VS images in the demo data.
The predictions will be saved to `./predictions`.
