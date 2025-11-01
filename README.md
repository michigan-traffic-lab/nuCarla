# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

## Environment Setup

- **CUDA (nvcc)**: 12.8
- **Driver Version**: 575  
- **Python**: 3.10  
- **OS**: Ubuntu 22.04
- **gcc**: 11.2

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n nuCarla python=3.10 -y
conda activate nuCarla
```

**b. Clone BEVFormer.**
```
git clone git@github.com:QZJGeorge/nuCarla.git
cd nuCarla
```

**c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
# Recommended torch>=2.0
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**e. Install Detectron2.**
```shell
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**f. Install mmcv (modified based on v1.7.2).**
```shell
cd mmcv
MMCV_WITH_OPS=1 pip install . --no-cache-dir -v
cd ..
```

**h. Prepare pretrained models.**
```shell
mkdir ckpts && cd ckpts

wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
```

note: this pretrained model is the same model used in [detr3d](https://github.com/WangYueFt/detr3d)
