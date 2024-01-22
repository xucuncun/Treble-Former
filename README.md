# Treble-Transformer

## Quick start

### 1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

### 2. [Install PyTorch](https://pytorch.org/get-started/locally/)

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Preparing datasets and pre-training models
(1) If training, validation and testing are done on the same dataset (Kvasir or CVC-ClinicDB is recommended for this 
dataset), put the dataset into "./data1", and train_with_data1.py will automatically split the dataset into training, 
validation and testing according to 8:1:1.\
(2) The datasets used in this study are publicly available at: \
Kvasir-SEG: https://datasets.simula.no/kvasir-seg/. \
CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/. \
(3) Pre-training models should be downloaded via their github connection and placed in location "./Models" after downloading.\
https://github.com/microsoft/Swin-Transformer \
https://github.com/whai362/PVT \
https://github.com/zengjixiangnfft/ESFPNet (Not official MixTransformer, but ESFPNet is excellent!) \

### 5. run training:
```
python train.py 
```
