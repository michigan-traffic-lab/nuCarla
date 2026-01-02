"""
Visualize 3D bounding boxes on camera images.
"""

import os
import json
import argparse

import cv2
import yaml
import numpy as np
from pyquaternion import Quaternion

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    CENTER_TO_WHEELBASE = config["data"]["center_to_wheelbase"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def quat_trans_to_matrix(q: Quaternion, t: np.ndarray):
    """Convert quaternion and translation to 4x4 transformation matrix."""
    R = q.rotation_matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def get_sensor_extrinsic(extrinsic_info):
    """Compute sensor extrinsic matrix from calibration info."""
    rotation = extrinsic_info["rotation"]
    q = Quaternion(w=rotation["w"], x=rotation["x"], y=rotation["y"], z=rotation["z"])

    yaw_q = Quaternion(axis=[0, 0, 1], degrees=90)
    roll_q = Quaternion(axis=[1, 0, 0], degrees=90)
    q = q * roll_q * yaw_q

    location = extrinsic_info["location"]
    t = np.array([location["x"] - CENTER_TO_WHEELBASE, location["y"], location["z"]])
    return quat_trans_to_matrix(q, t)


def cam_to_image(pt_cam, intrinsic):
    """Project a 3D point in camera coordinates to 2D image coordinates."""
    if pt_cam[2] <= 0.1:
        return None
    proj = intrinsic @ pt_cam
    u, v = proj[0] / proj[2], proj[1] / proj[2]
    return int(u), int(v)


def draw_box(img, corners_2d, color=(0, 255, 0), thickness=2):
    """Draw a 3D bounding box on the image."""
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    for i1, i2 in edges:
        if corners_2d[i1] is None or corners_2d[i2] is None:
            continue
        cv2.line(img, corners_2d[i1], corners_2d[i2], color, thickness)


def get_color_by_category(category):
    """Get visualization color based on actor category."""
    if category in ["vehicle.car", "vehicle.truck", "vehicle.bus.rigid"]:
        return (0, 255, 0)
    elif category in ["vehicle.motorcycle", "vehicle.bicycle"]:
        return (255, 0, 0)
    elif category in ["human.pedestrian.adult", "human.pedestrian.child"]:
        return (0, 0, 255)

    return (128, 128, 128)


def process_scene(scene_dir):
    """Process a scene and generate 3D bounding box visualizations."""
    calibration = load_json(os.path.join(scene_dir, "calibrated_sensor.json"))
    rgb_dir = os.path.join(scene_dir, "rgb")
    annotation_dir = os.path.join(scene_dir, "annotation")

    cameras = [d for d in os.listdir(rgb_dir) if os.path.isdir(os.path.join(rgb_dir, d))]
    annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith(".json")])

    for ann_file in annotation_files:
        frame_id = os.path.splitext(ann_file)[0]
        annotation = load_json(os.path.join(annotation_dir, ann_file))

        # Build ego transform (world to ego)
        ego_data = annotation["actors"]["ego"]
        ego_trans = ego_data["translation"]
        ego_rot = ego_data["rotation"]
        q_ego = Quaternion(w=ego_rot["w"], x=ego_rot["x"], y=ego_rot["y"], z=ego_rot["z"])
        t_ego = np.array([ego_trans["x"], ego_trans["y"], ego_trans["z"]])
        T_ego_world = quat_trans_to_matrix(q_ego, t_ego)
        T_world_ego = np.linalg.inv(T_ego_world)

        for camera in cameras:
            calib = calibration[camera]
            intrinsic = np.array(calib["intrinsic"])
            T_cam_ego = get_sensor_extrinsic(calib["extrinsic"])
            T_ego_cam = np.linalg.inv(T_cam_ego)

            img = cv2.imread(os.path.join(rgb_dir, camera, f"{frame_id}.jpg"))

            for actor_id, actor_data in annotation["actors"].items():
                if actor_id == "ego":
                    continue

                if actor_data.get("visibility", {}).get(camera, 0) == 0:
                    continue

                trans = actor_data["translation"]
                rot = actor_data["rotation"]
                size = actor_data["size"]

                q = Quaternion(w=rot["w"], x=rot["x"], y=rot["y"], z=rot["z"])
                t = np.array([trans["x"], trans["y"], trans["z"]])
                l, w, h = size["x"], size["y"], size["z"]

                # Pedestrians have their origin at center, so shift z down by half height
                category = actor_data.get("category", "")
                if category in ["human.pedestrian.adult", "human.pedestrian.child"]:
                    z_offset = -h / 2
                else:
                    z_offset = 0

                corners = np.array([
                    [l / 2, w / 2, z_offset], [l / 2, -w / 2, z_offset],
                    [-l / 2, -w / 2, z_offset], [-l / 2, w / 2, z_offset],
                    [l / 2, w / 2, h + z_offset], [l / 2, -w / 2, h + z_offset],
                    [-l / 2, -w / 2, h + z_offset], [-l / 2, w / 2, h + z_offset],
                ])
                corners_world = (q.rotation_matrix @ corners.T).T + t

                corners_2d = []
                for pt_world in corners_world:
                    pt_h = np.array([*pt_world, 1.0])
                    pt_ego = (T_world_ego @ pt_h)[:3]
                    pt_h = np.array([*pt_ego, 1.0])
                    pt_cam = (T_ego_cam @ pt_h)[:3]
                    pt_cam = np.array([-pt_cam[1], -pt_cam[2], pt_cam[0]])
                    corners_2d.append(cam_to_image(pt_cam, intrinsic))

                color = get_color_by_category(actor_data.get("category", ""))
                draw_box(img, corners_2d, color=color, thickness=2)

            out_dir = os.path.join(scene_dir, "visualize", camera)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{frame_id}.jpg"), img)

        print(f"Processed frame {frame_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, default="scene_001")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        data_dir = config["data"]["data_dir"]

    scene_dir = os.path.join(data_dir, args.scene_dir)
    process_scene(scene_dir)
