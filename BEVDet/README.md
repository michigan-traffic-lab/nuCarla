# BEVDet

#### Train model
```shell
# multiple gpu
./tools/dist_train.sh configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py 1
```

#### Test model
```shell
# multiple gpu
./tools/dist_test.sh configs/bevdet/bevdet-r50-4dlongterm-stereo-cbgs.py ckpts/bevdet_ep24.pth 1 --eval mAP
```

#### Visualize the predicted result.

```shell
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```
