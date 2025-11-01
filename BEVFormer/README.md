# BEVFormer: a Cutting-edge Baseline for Camera-based Detection

This is the customized version of BEVFormer for evaluation on the nuCarla dataset. For full details, please refer to the original work: https://github.com/fundamentalvision/BEVFormer/tree/master.

# Installation

## Tested Environment

- **CUDA**: 12.8
- **Nvidia Driver**: 575  
- **Python**: 3.10  
- **OS**: Ubuntu 22.04
- **GCC**: 11.2

**a. Create a conda environment.**
```shell
conda create -n bevformer python=3.10 -y
conda activate bevformer
```

**b. Install cuda torch and torchvision.**
```shell
# Required torch>=2.0
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**c. Install Detectron2.**
```shell
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**d. Compile MMCV from source (modified based on v1.7.2).**
```shell
cd mmcv
MMCV_WITH_OPS=1 pip install . --no-cache-dir -v
cd ..
```

**e. Install MMDetection3d (modified based on 1.0.0rc6).**
```shell
pip install -e .
```

# Prepare Data

Create symbolic link to the nuCarla dataset.
```
mkdir data
ln -s path/to/nucarla data/nuscenes
```

Generate annotation files for model training and evaluation.
```
python tools/create_data.py --version v1.0
```

# Train and Test

Train BEVFormer (adjust config file and the number of GPUs).
```
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 1
```

To resume training from a previously saved checkpoint:
```
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 1 --resume-from work_dirs/bevformer_base/latest.pth
```

Evaluate BEVFormer (adjust config file and the number of GPUs).
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_base_ep24.pth 1 --out results_bevformer.pkl --eval bbox
```

# Visualization
Update the results_path to your evaluation.
```
python tools/visual.py --results_path test/bevformer_base/Sun_Oct_19_17_49_38_2025/pts_bbox/results_nusc.json --version v1.0-trainval
```