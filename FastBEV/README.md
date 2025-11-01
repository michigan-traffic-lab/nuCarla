# FastBEV++


python tools/create_data_bevdet.py


## Get Started

```shell
./tools/dist_train.sh configs/fastbev/paper/fastbev-r50-cbgs-4d.py 8
```

test script:

```shell
./tools/dist_test.sh configs/fastbev/paper/fastbev-r50-cbgs.py work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth 8 --eval mAP 2>&1 | tee work_dirs/fastbev-r50-cbgs/epoch_20_ema.pth.log
```

python tools/test.py configs/fastbev/paper/fastbev-r50-cbgs-4d.py ckpts/fastbev-r50-cbgs-4d_ep24.pth --eval mAP