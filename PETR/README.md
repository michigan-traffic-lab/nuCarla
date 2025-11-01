## Environment Setup

### Create data
python tools/create_data.py --version v1.0


## Train & inference
```bash
You can train the model following:
```bash
tools/dist_train.sh projects/configs/petr/petr_vovnet_gridmask_p4_1600x640.py 1 --work-dir work_dirs/petr_vovnet_gridmask_p4_1600x640/
# --resume-from work_dirs/petr_vovnet_gridmask_p4_1600x640/latest.pth

```
You can evaluate the model following:
```bash
tools/dist_test.sh projects/configs/petr/petr_vovnet_gridmask_p4_1600x640.py ckpts/petr_vovnet_gridmask_p4_1600x640_ep24.pth 1 --eval bbox
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