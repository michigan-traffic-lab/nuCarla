# Fast-BEV: A Fast and Strong Bird’s-Eye View Perception Baseline

This is the customized version of FastBEV for evaluation on the nuCarla dataset. For full details, please refer to the original work: https://github.com/ymlab/advanced-fastbev.


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
pip install -e .
```

## Prepare Data

Create symbolic link to the nuCarla dataset.
```
mkdir data
ln -s path/to/nuCarla data/nuscenes
```

Generate annotation files for model training and evaluation.
```
# v1.0-mini for quick start up
# v1.0-trainval for training and validation
# v1.0-test for unseen maps test only

python tools/create_data.py --version v1.0-trainval
```

This will create `bevdetv3-nuscenes_infos_train.pkl` and `bevdetv3-nuscenes_infos_val.pkl`.

## Train and Test

Download pretrained resnet image backbone.

```shell
mkdir ckpts && cd ckpts

wget https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/resnet50-19c8e357.pth
```

Train FastBEV.
```
# Adjust config file and the number of GPUs
# (optional) --work-dir: directory to save the model and logs
# (optional) --resume-from: resume from a previous checkpoint

./tools/dist_train.sh ./configs/fastbev/fastbev-r50-cbgs-4d.py 1
```

Evaluate FastBEV.
```
# Adjust config file, saved checkpoint, and the number of GPUs

./tools/dist_test.sh ./configs/fastbev/fastbev-r50-cbgs-4d.py ./ckpts/fastbev-r50-cbgs-4d.pth 1
```

## Visualization

Plots will be saved in `test/pts_bbox/plots`.
```
python tools/visual.py ./test/pts_bbox/results_nusc.json --version v1.0-trainval --workers 4
```