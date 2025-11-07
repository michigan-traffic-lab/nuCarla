# Position Embedding Transformation for Multi-View 3D Object Detection

This is the customized version of PETR for evaluation on the nuCarla dataset. For full details, please refer to the original work: https://github.com/megvii-research/PETR.

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

**d. Install MMDetection3d (modified based on 1.0.0rc6).**
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
python tools/create_data.py --version v1.0
```

## Train and Test

Download pretrained Resnet image backbone.

```shell
mkdir ckpts && cd ckpts

wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

Train PETR (adjust config file and the number of GPUs).
```
tools/dist_train.sh ./projects/configs/petr/petr_vovnet_gridmask_p4_1600x640.py 1 --work-dir work_dirs/petr
```

Evaluate PETR (adjust config file, checkpoint, and the number of GPUs).
```
tools/dist_test.sh ./projects/configs/petr/petr_vovnet_gridmask_p4_1600x640.py ./ckpts/petr_vovnet_gridmask_p4_1600x640_ep24.pth 1 --eval bbox
```

## Visualize
You can generate the reault json following:
```bash
./tools/dist_test.sh projects/configs/petr/petr_vovnet_gridmask_p4_800x320.py work_dirs/petr_vovnet_gridmask_p4_800x320/latest.pth 1 --out work_dirs/pp-nus/results_eval.pkl --format-only --eval-options 'jsonfile_prefix=work_dirs/pp-nus/results_eval'
```
You can visualize the 3D object detection following:
```bash
python3 tools/visualize.py
```