import os
import json
import uuid
import yaml
import shutil
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

# ============================================================
# =============== Helper Functions ============================
# ============================================================

def generate_token():
    """Generate a random 32-character hex token."""
    return uuid.uuid4().hex

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def dumps_nuscenes_style(obj):
    """Convert dict to NuScenes-style JSON text."""
    lines = []
    for key, value in obj.items():
        if key in ["rotation", "translation", "size"]:
            arr_lines = ",\n".join(f"{v:.15g}" for v in value)
            arr_str = "[\n" + arr_lines + "\n]"
            lines.append(f"\"{key}\": {arr_str}")
        elif isinstance(value, str):
            lines.append(f"\"{key}\": \"{value}\"")
        else:
            lines.append(f"\"{key}\": {json.dumps(value)}")
    return "{\n" + ",\n".join(lines) + "\n}"

def save_json_nuscenes_style(data, path):
    """Save JSON array where each object follows NuScenes formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("[\n")
        for i, obj in enumerate(data):
            f.write(dumps_nuscenes_style(obj))
            if i != len(data) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]")

# ============================================================
# =============== Scene Renaming =============================
# ============================================================

def rename_scenes(source_dir, version):
    from nuscenes.utils import splits

    # if version == 'v1.0-trainval':
    #     target_names = splits.train + splits.val
    # elif version == 'v1.0-mini':
    #     target_names = splits.mini_train + splits.mini_val
    # else:
    #     raise ValueError('unknown nuscenes version')

    target_names = splits.val
    
    target_names = sorted(target_names)
    target_names = target_names[75:150]
    
    scene_dirs = sorted([d for d in os.listdir(source_dir) if d.startswith("scene_")])
    for i, scene_name in enumerate(tqdm(scene_dirs, desc="Renaming scenes")):
        if i < len(target_names):
            new_name = target_names[i]
        else:
            break
        old_path = os.path.join(source_dir, scene_name)
        new_path = os.path.join(source_dir, new_name)
        os.rename(old_path, new_path)

# ============================================================
# =============== Sample-Level Processing ====================
# ============================================================

def process_sample_files(annotation_dir, scene_token, global_sample_list, global_ego_pose_list):
    sample_files = sorted(
        [f for f in os.listdir(annotation_dir) if f.endswith(".json")],
        key=lambda x: int(x.replace(".json", ""))
    )

    scene_samples = []
    sample_tokens = [generate_token() for _ in sample_files]
    ego_pose_tokens = []

    for i, filename in enumerate(sample_files):
        path = os.path.join(annotation_dir, filename)
        data = load_json(path)
        timestamp = data["meta"]["timestamp"]

        prev_token = "" if i == 0 else sample_tokens[i - 1]
        next_token = "" if i == len(sample_files) - 1 else sample_tokens[i + 1]

        entry = {
            "token": sample_tokens[i],
            "timestamp": timestamp,
            "prev": prev_token,
            "next": next_token,
            "scene_token": scene_token
        }
        scene_samples.append(entry)
        global_sample_list.append(entry)

        ego = data["actors"]["ego"]
        rotation = ego["rotation"]
        translation = ego["translation"]
        ego_token = generate_token()
        ego_pose_tokens.append(ego_token)

        R = Quaternion([rotation["w"], rotation["x"], rotation["y"], rotation["z"]]).rotation_matrix
        shift_local = np.array([-1.317, 0.0, 0.0])
        shift_world = R @ shift_local
        tx, ty, tz = np.array([translation["x"], translation["y"], translation["z"]]) + shift_world

        ego_entry = {
            "token": ego_token,
            "timestamp": timestamp,
            "rotation": [
                rotation["w"], rotation["x"], rotation["y"], rotation["z"]
            ],
            "translation": [tx, ty, tz]
        }
        global_ego_pose_list.append(ego_entry)

    return {
        "nbr_samples": len(scene_samples),
        "first_sample_token": scene_samples[0]["token"],
        "last_sample_token": scene_samples[-1]["token"],
        "sample_tokens": sample_tokens,
        "ego_pose_tokens": ego_pose_tokens,
        "sample_files": sample_files,
    }

# ============================================================
# =============== Scene-Level Processing =====================
# ============================================================

def process_scene(source_dir, scene_name, global_sample_list, global_ego_pose_list,
                  global_sample_data_list, global_sample_annotation_list,
                  global_instance_list, sensing_range):
    
    scene_path = os.path.join(source_dir, scene_name)
    annotation_dir = os.path.join(scene_path, "annotation")
    rgb_dir = os.path.join(scene_path, "RGB")
    scene_token = generate_token()
    sample_info = process_sample_files(annotation_dir, scene_token, global_sample_list, global_ego_pose_list)

    build_sample_data_entries(source_dir, save_dir, scene_name, rgb_dir, sample_info, global_sample_data_list)

    scene_annos, scene_instances = build_sample_annotation_and_instance_entries(
        annotation_dir, sample_info, sensing_range
    )
    global_sample_annotation_list.extend(scene_annos)
    global_instance_list.extend(scene_instances)

    weather_file = os.path.join(scene_path, "weather.txt")
    with open(weather_file, "r") as f:
        weather_description = f.read().strip()

    scene_summary = {
        "token": scene_token,
        "log_token": "",
        "nbr_samples": sample_info["nbr_samples"],
        "first_sample_token": sample_info["first_sample_token"],
        "last_sample_token": sample_info["last_sample_token"],
        "name": scene_name,
        "description": f"Mcity, {weather_description}"
    }

    return scene_summary

# ============================================================
# =============== sample_data.json ===========================
# ============================================================

calibrated_sensor_tokens = {
    "CAM_FRONT": "a894159f414f4fc9acf1b6e79b8f109a",
    "CAM_FRONT_LEFT": "f4d1e4e9e3764ff4b6166c754568da52",
    "CAM_FRONT_RIGHT": "72b8a10d51384b43aee498b2d3e3cce3",
    "CAM_BACK": "ab0a4a5aa1f04a5fbfac42bc4ca385f7",
    "CAM_BACK_LEFT": "f2d4ddfd079549bf8b628cb809690d7a",
    "CAM_BACK_RIGHT": "1786859cc6224cac9fbf330d608b6f1f",
    "LIDAR_TOP": "a183049901c24361a6b0b11b8013137c"
}

category_tokens = {
    "vehicle.car": "fd69059b62a3469fbaef25340c0eab7f",
    "vehicle.motorcycle": "dfd26f200ade4d24b540184e16050022",
    "vehicle.bicycle": "fc95c87b806f48f8a1faea2dcc2222a4",
    "vehicle.bus.rigid": "fedb11688db84088883945752e480c2c",
    "vehicle.truck": "6021b5187b924d64be64a702e5570edf",
    "vehicle.emergency.police": "7b2ff083a64e4d53809ae5d9be563504",

    "human.pedestrian.adult": "1fa93b757fc74fb197cdd60001ad8abf",
    "human.pedestrian.child": "b1c6de4c57f14a5383d9f963fbdcb5cb",
    "human.pedestrian.police_officer": "bb867e2064014279863c71a29b1eb381"
}

attribute_tokens = {
    "vehicle.moving": "cb5118da1ab342aa947717dc53544259",
    "vehicle.stopped": "c3246a1e22a14fcb878aa61e69ae3329",
    "cycle.with_rider": "a14936d865eb4216b396adae8cb3939c",
    "pedestrian.moving": "ab83627ff28b465b85c427162dec722f",
}

def build_sample_data_entries(source_dir, save_dir, scene_name, rgb_dir, sample_info, global_sample_data_list):
    samples_dir = os.path.join(save_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sensors = list(calibrated_sensor_tokens.keys())
    for s in sensors:
        os.makedirs(os.path.join(samples_dir, s), exist_ok=True)

    lidar_dir = os.path.join(samples_dir, "LIDAR_TOP")
    os.makedirs(lidar_dir, exist_ok=True)

    sensor_entries = {sensor: [] for sensor in sensors}

    for idx, filename in enumerate(sample_info["sample_files"]):
        ann_path = os.path.join(source_dir, scene_name, "annotation", filename)
        ann_data = load_json(ann_path)
        datetime_str = ann_data["meta"]["datetime"]
        timestamp = ann_data["meta"]["timestamp"]
        sample_token = sample_info["sample_tokens"][idx]
        ego_pose_token = sample_info["ego_pose_tokens"][idx]

        lidar_filename = f"n001-{datetime_str}__LIDAR_TOP__{timestamp}.pcd.bin"
        lidar_abs_path = os.path.join(lidar_dir, lidar_filename)
        open(lidar_abs_path, "wb").close()

        lidar_entry = {
            "token": generate_token(),
            "sample_token": sample_token,
            "ego_pose_token": ego_pose_token,
            "calibrated_sensor_token": calibrated_sensor_tokens["LIDAR_TOP"],
            "timestamp": timestamp,
            "fileformat": "pcd.bin",
            "is_key_frame": True,
            "height": 0,
            "width": 0,
            "filename": os.path.join("samples", "LIDAR_TOP", lidar_filename)
        }
        sensor_entries["LIDAR_TOP"].append(lidar_entry)

        for s in sensors:
            if s == "LIDAR_TOP":
                continue
            src_img_path = os.path.join(rgb_dir, s, filename.replace(".json", ".jpg"))
            dst_filename = f"n001-{datetime_str}__{s}__{timestamp}.jpg"
            dst_rel_path = os.path.join("samples", s, dst_filename)
            dst_abs_path = os.path.join(save_dir, dst_rel_path)
            shutil.copy(src_img_path, dst_abs_path)

            entry = {
                "token": generate_token(),
                "sample_token": sample_token,
                "ego_pose_token": ego_pose_token,
                "calibrated_sensor_token": calibrated_sensor_tokens[s],
                "timestamp": timestamp,
                "fileformat": "jpg",
                "is_key_frame": True,
                "height": 900,
                "width": 1600,
                "filename": dst_rel_path
            }
            sensor_entries[s].append(entry)

    for _, entries in sensor_entries.items():
        for i, entry in enumerate(entries):
            entry["prev"] = "" if i == 0 else entries[i - 1]["token"]
            entry["next"] = "" if i == len(entries) - 1 else entries[i + 1]["token"]
            global_sample_data_list.append(entry)

# ============================================================
# =============== sample_annotation + instance.json ==========
# ============================================================

def build_sample_annotation_and_instance_entries(annotation_dir, sample_info, sensing_range):
    scene_sample_annotation_list = []
    scene_instance_list = []

    instance_map = {}
    completed_instances = []
    prev_annotations = {}

    for idx, filename in enumerate(sample_info["sample_files"]):
        path = os.path.join(annotation_dir, filename)
        anno = load_json(path)
        sample_token = sample_info["sample_tokens"][idx]
        ego = anno["actors"]["ego"]
        ego_pos = np.array([ego["translation"]["x"], ego["translation"]["y"], ego["translation"]["z"]])

        current_ids = set(anno["actors"].keys()) - {"ego"}

        # Remove instances that disappeared
        for actor_id in list(instance_map.keys()):
            if actor_id not in current_ids:
                last_ann_token = prev_annotations[actor_id]
                for entry in scene_sample_annotation_list:
                    if entry["token"] == last_ann_token:
                        entry["next"] = ""
                        break
                completed_instances.append(instance_map.pop(actor_id))
                prev_annotations.pop(actor_id)

        # Process visible actors
        for actor_id, actor_data in anno["actors"].items():
            if actor_id == "ego":
                continue

            visibility = actor_data["visibility"]
            visible = False
            for cam, pixels in visibility.items():
                if pixels > 150:
                    visible = True
                    break

            if not visible:
                print(f"filter out actor", {actor_id})
                continue

            pos = np.array([
                actor_data["translation"]["x"],
                actor_data["translation"]["y"],
                actor_data["translation"]["z"]
            ])
            distance = np.linalg.norm(pos - ego_pos)
            if distance > sensing_range:
                continue

            category = actor_data["category"]
            attribute = actor_data["attribute"]

            if attribute == "pedestrian.moving": 
                translation_z = actor_data["translation"]["z"]
            else:
                translation_z = actor_data["translation"]["z"] + actor_data["size"]["z"]/2

            category_token = category_tokens[category]
            attribute_token = attribute_tokens[attribute]

            if actor_id not in instance_map:
                instance_token = generate_token()
                instance_map[actor_id] = {
                    "token": instance_token,
                    "category_token": category_token,
                    "annotation_tokens": []
                }
                prev_token = ""
            else:
                prev_token = prev_annotations[actor_id]

            ann_token = generate_token()
            entry = {
                "token": ann_token,
                "sample_token": sample_token,
                "instance_token": instance_map[actor_id]["token"],
                "visibility_token": "4",
                "attribute_tokens": [attribute_token],
                "translation": [
                    actor_data["translation"]["x"],
                    actor_data["translation"]["y"],
                    translation_z
                ],
                "size": [
                    actor_data["size"]["y"],
                    actor_data["size"]["x"],
                    actor_data["size"]["z"]
                ],
                "rotation": [
                    actor_data["rotation"]["w"],
                    actor_data["rotation"]["x"],
                    actor_data["rotation"]["y"],
                    actor_data["rotation"]["z"]
                ],
                "num_lidar_pts": 100,
                "num_radar_pts": 100,
                "prev": prev_token,
                "next": ""
            }

            if prev_token:
                for ann in scene_sample_annotation_list:
                    if ann["token"] == prev_token:
                        ann["next"] = ann_token
                        break

            prev_annotations[actor_id] = ann_token
            instance_map[actor_id]["annotation_tokens"].append(ann_token)
            scene_sample_annotation_list.append(entry)

    for actor_id, inst in instance_map.items():
        last_token = prev_annotations[actor_id]
        for entry in scene_sample_annotation_list:
            if entry["token"] == last_token:
                entry["next"] = ""
                break
        completed_instances.append(inst)

    for inst in completed_instances:
        tokens = inst["annotation_tokens"]
        if len(tokens) == 0:
            continue
        scene_instance_list.append({
            "token": inst["token"],
            "category_token": inst["category_token"],
            "nbr_annotations": len(tokens),
            "first_annotation_token": tokens[0],
            "last_annotation_token": tokens[-1]
        })

    # return local lists
    return scene_sample_annotation_list, scene_instance_list

# ============================================================
# =============== Dataset-Level Orchestration ================
# ============================================================

def convert_to_nuscenes(source_dir, save_dir, version, sensing_range):
    rename_scenes(source_dir, version)

    # copy pre-configured json files
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, "v1.0-trainval")
    dst_dir = os.path.join(save_dir, version)
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

    scene_dirs = sorted([d for d in os.listdir(source_dir) if d.startswith("scene-")])

    all_scene_summaries = []
    all_sample_summaries = []
    all_ego_pose_entries = []
    all_sample_data_entries = []
    all_sample_annotation_entries = []
    all_instance_entries = []

    for scene_name in tqdm(scene_dirs, desc="Processing scenes"):
        scene_summary = process_scene(source_dir, scene_name,
                                      all_sample_summaries,
                                      all_ego_pose_entries,
                                      all_sample_data_entries,
                                      all_sample_annotation_entries,
                                      all_instance_entries,
                                      sensing_range)
        all_scene_summaries.append(scene_summary)

    save_json_nuscenes_style(all_scene_summaries, os.path.join(save_dir, version, "scene.json"))
    save_json_nuscenes_style(all_sample_summaries, os.path.join(save_dir, version, "sample.json"))
    save_json_nuscenes_style(all_ego_pose_entries, os.path.join(save_dir, version, "ego_pose.json"))
    save_json_nuscenes_style(all_sample_data_entries, os.path.join(save_dir, version, "sample_data.json"))
    save_json_nuscenes_style(all_sample_annotation_entries, os.path.join(save_dir, version, "sample_annotation.json"))
    save_json_nuscenes_style(all_instance_entries, os.path.join(save_dir, version, "instance.json"))

    print(f"\n[INFO] Process completed.")
    print(f"[INFO] Total scenes processed: {len(all_scene_summaries)}")
    print(f"[INFO] Total samples processed: {len(all_sample_summaries)}")
    print(f"[INFO] Total ego poses saved: {len(all_ego_pose_entries)}")
    print(f"[INFO] Total sample data saved: {len(all_sample_data_entries)}")
    print(f"[INFO] Total sample annotations saved: {len(all_sample_annotation_entries)}")
    print(f"[INFO] Total instances saved: {len(all_instance_entries)}")

# ============================================================
# =============== Entry Point ================================
# ============================================================

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        sensing_range = config["parameters"]["sensing_range"]

    source_dir = '/media/zhijie/Disk2/data/Mcity'
    save_dir = '/media/zhijie/Disk2/distributed/Mcity'
    version = 'v1.0-trainval'

    convert_to_nuscenes(source_dir, save_dir, version, sensing_range)
