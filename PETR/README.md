# Position Embedding Transformation for Multi-View 3D Object Detection

This is the customized version of PETR for evaluation on the nuCarla dataset. For full details, please refer to the original work: https://github.com/megvii-research/PETR.


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

This will create `nuscenes_infos_train.pkl` and `nuscenes_infos_val.pkl`.

## Train and Test

Download pretrained resnet image backbone.

```shell
mkdir ckpts && cd ckpts

wget https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/fcos3d-vovnet-imgbackbone-remapped.pth
```

Train PETR.
```
# Adjust config file and the number of GPUs
# (optional) --work-dir: directory to save the model and logs
# (optional) --resume-from: resume from a previous checkpoint

tools/dist_train.sh ./projects/configs/petr/petr-vovnet-gridmask-p4-1600x640.py 1
```

Evaluate PETR.
```
# Adjust config file, saved checkpoint, and the number of GPUs

tools/dist_test.sh ./projects/configs/petr/petr-vovnet-gridmask-p4-1600x640.py ./ckpts/petr-vovnet-gridmask-p4-1600x640.pth 1
```


## Visualization

Plots will be saved in `test/pts_bbox/plots`.
```
python tools/visual.py ./test/pts_bbox/results_nusc.json --version v1.0-trainval --workers 4
```