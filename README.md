# nuCarla: A nuScenes-Style Bird’s-Eye View Perception Dataset for CARLA Simulation


## Introduction

nuCarla ([paper.pdf](https://github.com/user-attachments/files/23537819/nuCarla_Open.pdf)) is a large-scale, nuScenes-compatible, camera-based perception dataset developed in the CARLA simulator, featuring 9 distinct maps, 14 weather conditions, and 6 object classes. It is designed to facilitate training of effective perception representations for end-to-end autonomous driving development. The dataset contains 1,000 scenarios in total, with 700 for training, 150 for validation, and 150 for testing, matching the nuScenes split. To thoroughly validate nuCarla, we train and evaluate four state-of-the-art Bird's Eye View (BEV) perception models: [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [PETR](https://github.com/megvii-research/PETR), [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [FastBEV](https://github.com/ymlab/advanced-fastbev), and demonstrate that all models achieve competitive performance using the official nuScenes detection metrics.

https://github.com/user-attachments/assets/4e1b57fd-4a80-4032-9c23-efc975fa57e1


## Key Features

🚗 The first large-scale **CARLA-based perception dataset** with full compatibility to the nuScenes format.

⚙️ **MMDetection3D-1.0 fully upgraded** for usage with the latest PyTorch (2.7+) and CUDA (12.8+).

📦 Open-source **dataset and pretrained model weights** provided.


## Tested Environment

The following environment settings are applied across all models in this repository:

- **CUDA**: 12.8
- **Nvidia Driver**: 570
- **Python**: 3.10
- **OS**: Ubuntu 22.04


## Data Preparation

The dataset is publicly available at [Hugging Face](https://huggingface.co/datasets/zhijieq/nuCarla). Download it using `download.sh` and organize it in the following structure.

```
nuCarla
├─ maps/
├─ samples/
├─ v1.0-mini/
├─ v1.0-test/
├─ v1.0-trainval/
```

**Note**: This is a camera-based perception dataset. The LiDAR files in the sample folder are dummy placeholders provided solely for compatibility with the MMDetection3D framework conventions.


## Training and Evaluation

For complete installation, training, and evaluation workflows, please refer to the individual folder corresponding to each model: [BEVFormer](BEVFormer/), [PETR](PETR/), [BEVDet](BEVDet/), [FastBEV](FastBEV/).


## Model Zoo

**IMPORTANT**: The following results are evaluated on the nuCarla *validation* set. All metrics are post-computed based on the six available classes (car, truck, bus, pedestrian, motorcycle, bicycle) and are not the direct output from the nuScenes console.

Due to time and resource considerations, we only conducted thorough evaluations for the selected model variants within each family. For users interested in other variants, please refer to the respective original code repositories.


| Method | Schedule | VRAM | GPU Hours | mAP | NDS | Config | Download
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BEVFormer (Base) | 24ep | 30500M | 300 | 0.813 | 0.778 | [config](BEVFormer/projects/configs/bevformer/bevformer-base.py) | [model](https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/bevformer-base.pth)/[log](https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/bevformer-base.log) |
| PETR (VovNet-Grid<br>Mask-P4-1600x640) | 24ep | 9500M | 150 | 0.745 | 0.710 | [config](PETR/projects/configs/petr/petr-vovnet-gridmask-p4-1600x640.py) | [model](https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/petr-vovnet-gridmask-p4-1600x640.pth)/[log](https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/petr-vovnet-gridmask-p4-1600x640.log) |
| BEVDet (R50-4DLong<br>Term-Stereo-CBGS) | 24ep | 8500M | 300 | 0.811 | 0.753 | [config](BEVDet/configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py) | [model](https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/bevdet-r50-4dlongterm-stereo-cbgs.pth)/[log](https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/bevdet-r50-4dlongterm-stereo-cbgs.log) |
| FastBEV (R50-CBGS-4D) | 24ep | 14000M | 50 | 0.777 | 0.728 |  [config](FastBEV/configs/fastbev/fastbev-r50-cbgs-4d.py) | [model](https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/fastbev-r50-cbgs-4d.pth)/[log](https://github.com/michigan-traffic-lab/nuCarla/releases/download/v1.0/fastbev-r50-cbgs-4d.log) |


## Acknowledgement

Many thanks to these excellent open source projects:

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [PETR](https://github.com/megvii-research/PETR)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [FastBEV](https://github.com/ymlab/advanced-fastbev)

## Citation

If you use nuCarla in your research, please consider citing:

```bibtex
@article{nucarla,
      title={nuCarla: A nuScenes-Style Bird’s-Eye View Perception Dataset for CARLA Simulation}, 
      author={Qiao, Zhijie and Cao, Zhong and Liu, Henry X.},
      year={2025},
}
