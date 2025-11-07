import os
import json
import cv2
import yaml
import numpy as np
from pyquaternion import Quaternion


# ============================================================
# ==== Utility Functions ====
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def quat_trans_to_matrix(q: Quaternion, t: np.ndarray):
    """Convert quaternion + translation into a 4x4 transform matrix."""
    R = q.rotation_matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def get_sensor_extrinsic(sensor_info):
    q = Quaternion(sensor_info["rotation"])

    # axis adjustments to match image coordinates
    yaw_q = Quaternion(axis=[0, 0, 1], degrees=90)
    roll_q = Quaternion(axis=[1, 0, 0], degrees=90)
    q = q * roll_q * yaw_q

    translation = sensor_info["translation"]
    t = (translation[0], translation[1], translation[2])
    return quat_trans_to_matrix(q, t)

def world_to_ego(pt_world, T_world_ego):
    pt_h = np.array([*pt_world, 1.0])
    return (T_world_ego @ pt_h)[:3]

def ego_to_cam(pt_ego, T_ego_cam):
    pt_h = np.array([*pt_ego, 1.0])
    return (T_ego_cam @ pt_h)[:3]

def cam_to_image(pt_cam, intrinsic):
    if pt_cam[2] <= 0.1:
        return None
    proj = intrinsic @ pt_cam
    u, v = proj[0] / proj[2], proj[1] / proj[2]
    return int(u), int(v)

def draw_box(img, corners_2d, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box edges projected to 2D image."""
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    for i1, i2 in edges:
        if corners_2d[i1] is None or corners_2d[i2] is None:
            continue
        cv2.line(img, corners_2d[i1], corners_2d[i2], color, thickness)
    return img

# ============================================================
# ==== Main Processing ====
# ============================================================

attributes = {
    "cb5118da1ab342aa947717dc53544259": "vehicle",
    "c3246a1e22a14fcb878aa61e69ae3329": "vehicle",
    "a14936d865eb4216b396adae8cb3939c": "cycle",
    "ab83627ff28b465b85c427162dec722f": "pedestrian"
}


def process_data(save_dir, version):
    """
    Read all metadata JSONs and render 3D bounding boxes onto camera images.
    Saves output under save_dir/3D_BOX/<camera>/.
    """

    # === Load JSON files ===
    sample_data = load_json(os.path.join(save_dir, version, "sample_data.json"))
    ego_pose = load_json(os.path.join(save_dir, version, "ego_pose.json"))
    sample_annotation = load_json(os.path.join(save_dir, version, "sample_annotation.json"))
    calibrated_sensor = load_json(os.path.join(save_dir, version, "calibrated_sensor.json"))

    # === Create quick lookup dictionaries ===
    ego_pose_map = {e["token"]: e for e in ego_pose}
    calib_map = {c["token"]: c for c in calibrated_sensor}

    # group annotations by sample_token
    ann_by_sample = {}
    for ann in sample_annotation:
        s_token = ann["sample_token"]
        ann_by_sample.setdefault(s_token, []).append(ann)

    os.makedirs(os.path.join(save_dir, "3D_BOX"), exist_ok=True)

    # === Iterate through sample_data ===
    for sd in sample_data:
        filename = sd["filename"]
        sample_token = sd["sample_token"]
        ego_pose_token = sd["ego_pose_token"]
        calib_token = sd["calibrated_sensor_token"]

        img_path = os.path.join(save_dir, filename)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        # === Get ego pose and calibrated sensor ===
        ego = ego_pose_map[ego_pose_token]
        q_ego = Quaternion(ego["rotation"])
        t_ego = np.array(ego["translation"])
        T_ego_world = quat_trans_to_matrix(q_ego, t_ego)
        T_world_ego = np.linalg.inv(T_ego_world)

        calib = calib_map[calib_token]

        intrinsic = np.array(calib["camera_intrinsic"])

        T_cam_ego = get_sensor_extrinsic(calib)
        T_ego_cam = np.linalg.inv(T_cam_ego)

        # === Get annotations belonging to this sample ===
        anns = ann_by_sample.get(sample_token, [])
        if len(anns) == 0:
            continue

        for ann in anns:
            attribute_token = ann["attribute_tokens"]
            attribute = attributes[attribute_token[0]]

            if attribute == "vehicle":
                color = (0, 255, 0)
            elif attribute == "cycle":
                color = (255, 0, 0)
            elif attribute == "pedestrian":
                color = (0, 0, 255)
            else:
                raise ValueError

            translation = np.array(ann["translation"])
            rotation = Quaternion(ann["rotation"])
            size = ann["size"]
            w, l, h = size

            # 3D bounding box corners in box frame
            half = np.array([
                [ l/2,  w/2, -h/2],
                [ l/2, -w/2, -h/2],
                [-l/2, -w/2, -h/2],
                [-l/2,  w/2, -h/2],
                [ l/2,  w/2,  h/2],
                [ l/2, -w/2,  h/2],
                [-l/2, -w/2,  h/2],
                [-l/2,  w/2,  h/2],
            ])

            # Transform corners to world coordinates
            R_box = rotation.rotation_matrix
            corners_world = (R_box @ half.T).T + translation

            # === Project corners onto the image ===
            corners_2d = []
            for pt_world in corners_world:
                pt_ego = world_to_ego(pt_world, T_world_ego)
                pt_cam = ego_to_cam(pt_ego, T_ego_cam)
                # Remap axes as in original code convention
                pt_cam = np.array([-pt_cam[1], -pt_cam[2], pt_cam[0]])
                proj = cam_to_image(pt_cam, intrinsic)
                corners_2d.append(proj)

            img = draw_box(img, corners_2d, color=color, thickness=2)

        # === Save output ===
        out_dir = os.path.join(save_dir, "3D_BOX", os.path.dirname(filename))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "3D_BOX", filename)
        cv2.imwrite(out_path, img)
        print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        # save_dir = config["parameters"]["save_dir"]
        # version = config["parameters"]["version"]

    save_dir = '/media/zhijie/Disk2/distributed/Mcity'
    version = 'v1.0-trainval'

    process_data(save_dir, version)
