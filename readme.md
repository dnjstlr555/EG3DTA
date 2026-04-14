# Efficient Graph 3D Convolution Network for Explainable Cross-Subject Rehabilitation Assessment


Official PyTorch implementation of the paper **"Efficient Graph 3D Convolution Network for Explainable Cross-Subject Rehabilitation Assessment"**.


## Installation
Use dockerfile to automatically set up the environment, or follow the instructions below to manually set up the environment:
```bash
# Clone this repository
git clone [https://github.com/dnjstlr555/EG3DTA.git](https://github.com/dnjstlr555/EG3DTA.git)
cd EG3DTA

# Create a conda environment
conda create -n eg3d python=3.8 -y
conda activate eg3d

# Download torch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Download openmim
pip install -U openmim
mim install mmengine "mmcv>=2.0.1"

# Install dependencies
pip install -r requirements.txt

# Test
python docker-test.py
```

## Data Preparation
Please download them from the official sources:

1. **KiMoRe Dataset:** [Download Link](https://drive.google.com/drive/folders/1b1anGzSytePiCUyoz8AlGMV2G54e_ZeC?usp=share_link)
2. **UI-PRMD Dataset:** [Download Link](https://www.idahofallshighered.org/vakanski/ui-prmd.html) (Download the reduced dataset)

After downloading, organize the data as follows:
```text
data/
  ├── kimore/
  │   ├── CG/
  │   └── GPP/
  └── uiprmd/
      ├── Correct Movements/
      └── ...
```
Run the data preprocessing script:
```bash
python utils/preprocess_kimore.py
python utils/preprocess_uiprmd.py
```

## Training

To train the E-G3D model from scratch using subject-wise 5-fold cross-validation:

```bash
# Train on KiMoRe
python main.py --data_path ./data/kimore_kfold_norm.pkl --graph_type nturgb+d --phase train

# Train on UI-PRMD
python main.py --data_path ./data/uiprmd_kfold.pkl --graph_type vicon --phase train
```

## Evaluation & Pretrained Models

To evaluate a trained model on unseen subjects:

```bash
# Evaluate on KiMoRe Dataset
python main.py --data_path ./data/kimore_kfold_norm.pkl --graph_type nturgb+d --phase eval
```

## Visualization
To visualize the learned temporal edges and attention maps:
```bash
python visualize.py --ex 1 --fold 0
```

## Contact
For any questions or discussions, please feel free to open an issue or contact Won-Sik Oh at dnjstlr555@gmail.com.