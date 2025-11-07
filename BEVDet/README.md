# BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View

This is the customized version of BEVDet for evaluation on the nuCarla dataset. For full details, please refer to the original work: https://github.com/HuangJunJie2017/BEVDet.

## Tested Environment

- **CUDA**: 12.8
- **Nvidia Driver**: 570
- **Python**: 3.10
- **OS**: Ubuntu 22.04

## Installation

**a. Create a conda environment.**
```shell
conda create -n petr python=3.10 -y
conda activate petr
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

Generate annotation files for model training and evaluation.
```
python tools/create_data.py
```

## Train and Test

Train BEVDet (adjust config file and the number of GPUs).
```
./tools/dist_train.sh configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py 1
```

Evaluate BEVDet (adjust config file, checkpoint, and the number of GPUs).
```
./tools/dist_test.sh ./configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py ./ckpts/bevdet-ep24.pth 1 --eval mAP
```

#### Visualize the predicted result.

```shell
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```
