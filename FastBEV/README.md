# Fast-BEV: A Fast and Strong Bird’s-Eye View Perception Baseline

This is the customized version of FastBEV for evaluation on the nuCarla dataset. For full details, please refer to the original work: https://github.com/ymlab/advanced-fastbev.

## Tested Environment

- **CUDA**: 12.8
- **Nvidia Driver**: 570
- **Python**: 3.10
- **OS**: Ubuntu 22.04

## Installation

**a. Create a conda environment.**
```shell
conda create -n fastbev python=3.10 -y
conda activate fastbev
```

**b. Install Torch and Torchvision.**
```shell
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**c. Compile MMCV (modified based on v1.7.2) from source, this can take a few minutes.**
```shell
cd mmcv
MMCV_WITH_OPS=1 pip install . --no-cache-dir -v
cd ..
```

**d. Install MMDetection3d (modified based on 1.0.0rc4).**
```shell
pip install .
```

## Prepare Data

Create symbolic link to the nuCarla dataset.
```
mkdir data
ln -s path/to/nuCarla data/nuscenes
```

Generate annotation files for model training and evaluation.
```
python tools/create_data.py
```

## Train and Test

Download pretrained Resnet image backbone.

```shell
mkdir ckpts && cd ckpts

wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

Train FastBEV (adjust config file and the number of GPUs).
```
./tools/dist_train.sh ./configs/fastbev/fastbev-r50-cbgs-4d.py 1
```

Evaluate FastBEV (adjust config file, checkpoint, and the number of GPUs).
```
./tools/dist_test.sh ./configs/fastbev/fastbev-r50-cbgs-4d.py ./ckpts/fastbev-r50-cbgs-4d.pth 1 --eval mAP
```

## Visualization
```
python tools/visual.py --results_path test/bevformer_base/Sun_Oct_19_17_49_38_2025/pts_bbox/results_nusc.json --version v1.0-trainval
```