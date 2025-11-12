import os
import mmcv
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import Box
import gc


cams = [
  'CAM_FRONT',
  'CAM_FRONT_RIGHT',
  'CAM_BACK_RIGHT',
  'CAM_BACK',
  'CAM_BACK_LEFT',
  'CAM_FRONT_LEFT'
]


def render_annotation(anntoken: str, margin: float = 10) -> None:
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes, select_cams = [], []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=BoxVisibility.ANY,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)

    num_cam = len(all_bboxes)
    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    plt.subplots_adjust(wspace=0.08, hspace=0.12)
    select_cams = [sample_record['data'][cam] for cam in select_cams]

    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])

    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], colors=(c, c, c))
        corners = view_points(boxes[0].corners(), np.eye(4), False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    plt.close()
    gc.collect()


def get_color(category_name: str):
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']
    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def get_predicted_data(sample_data_token: str, pred_anns=None):
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    boxes = pred_anns
    box_list = []
    for box in boxes:
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        if sensor_record['modality'] == 'camera' and not box_in_image(box, cam_intrinsic, imsize):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def render_sample_data(sample_token: str, pred_data=None, out_path: str = None, verbose: bool = True) -> None:
    sample = nusc.get('sample', sample_token)
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]
    _, ax = plt.subplots(4, 3, figsize=(24, 18))
    plt.subplots_adjust(wspace=0.07, hspace=0.12)  # wider vertical margin, same horizontal
    j = 0
    for ind, cam in enumerate(cams):
        sample_data_token = sample['data'][cam]
        sd_record = nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality == 'camera':
            boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                         name=record['detection_name'], token='predicted')
                     for record in pred_data['results'][sample_token]
                     if record['detection_score'] > 0.2]

            data_path, boxes_pred, camera_intrinsic = get_predicted_data(sample_data_token, pred_anns=boxes)
            _, boxes_gt, _ = nusc.get_sample_data(sample_data_token)
            if ind == 3:
                j += 1
            ind = ind % 3
            data = Image.open(data_path)

            ax[j, ind].imshow(data)
            ax[j + 2, ind].imshow(data)
            for box in boxes_pred:
                c = np.array(get_color(box.name)) / 255.0
                box.render(ax[j, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))
            for box in boxes_gt:
                c = np.array(get_color(box.name)) / 255.0
                box.render(ax[j + 2, ind], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            ax[j, ind].set_xlim(0, data.size[0])
            ax[j, ind].set_ylim(data.size[1], 0)
            ax[j + 2, ind].set_xlim(0, data.size[0])
            ax[j + 2, ind].set_ylim(data.size[1], 0)
            ax[j, ind].axis('off')
            ax[j, ind].set_title(f'PRED: {sd_record["channel"]}')
            ax[j + 2, ind].axis('off')
            ax[j + 2, ind].set_title(f'GT: {sd_record["channel"]}')
        else:
            raise ValueError("Error: Unknown sensor modality!")

    if out_path is not None:
        plt.savefig(out_path + '_camera', bbox_inches='tight', pad_inches=0.2, dpi=200, facecolor='white')
    if verbose:
        plt.show()

    plt.close()
    gc.collect()


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Render NuScenes sample data with predicted results.")
    parser.add_argument('results_path', type=str, help='Path to the results_nusc.json file.')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='NuScenes dataset version.')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers.')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot='./data/nuscenes', verbose=False)
    bevformer_results = mmcv.load(args.results_path)
    sample_token_list = list(bevformer_results['results'].keys())

    scene_to_samples = {}
    for scene in nusc.scene:
        first = scene['first_sample_token']
        samples = []
        while first != "":
            sample = nusc.get('sample', first)
            samples.append(first)
            first = sample['next']
        scene_to_samples[scene['token']] = {"name": scene['name'], "samples": samples}

    token_to_name = {}
    for scene_idx, (scene_token, scene_info) in enumerate(scene_to_samples.items(), start=1):
        scene_name = f"scene_{scene_idx:04d}"
        for frame_idx, token in enumerate(scene_info["samples"], start=1):
            token_to_name[token] = f"{scene_name}_frame_{frame_idx:04d}"

    ordered_tokens = [t for t in token_to_name.keys() if t in sample_token_list]
    print(f"Total sorted samples: {len(ordered_tokens)}")

    base_dir = os.path.dirname(args.results_path)
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    def process_token(token):
        name = token_to_name[token]
        out_path = os.path.join(out_dir, name)
        try:
            render_sample_data(token, pred_data=bevformer_results, out_path=out_path, verbose=False)
        except Exception as e:
            print(f"Error rendering {name}: {e}")

    with mp.Pool(processes=args.workers) as pool:
        list(tqdm(pool.imap_unordered(process_token, ordered_tokens),
                  total=len(ordered_tokens), desc="Rendering samples"))
