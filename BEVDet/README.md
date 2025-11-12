# BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View

This is the customized version of BEVDet for evaluation on the nuCarla dataset. For full details, please refer to the original work: https://github.com/HuangJunJie2017/BEVDet.


## Installation

**a. Create a conda environment.**
```shell
conda create -n bevdet python=3.10 -y
conda activate bevdet
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
pip install -e .
```

## Prepare Data

Generate annotation files for model training and evaluation.
```
# v1.0-mini for quick start up
# v1.0-trainval for training and validation
# v1.0-test for unseen maps test only

python tools/create_data.py --version v1.0-trainval
```

This will create `bevdetv3-nuscenes_infos_train.pkl` and `bevdetv3-nuscenes_infos_val.pkl`.

## Train and Test

Train BEVDet.
```
# Adjust config file and the number of GPUs
# (optional) --work-dir: directory to save the model and logs
# (optional) --resume-from: resume from a previous checkpoint

./tools/dist_train.sh ./configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py 1
```

Evaluate BEVDet.
```
# Adjust config file, saved checkpoint, and the number of GPUs

./tools/dist_test.sh ./configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py ./ckpts/bevdet-r50-4dlongterm-stereo-cbgs.pth 1
```

#### Visualize the predicted result.

Plots will be saved in `test/pts_bbox/plots`.
```
python tools/visual.py ./test/pts_bbox/results_nusc.json --version v1.0-trainval --workers 4
```
