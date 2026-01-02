"""
Generate annotation files from ground truth and instance segmentation data.
"""

import os
import json
from multiprocessing import Pool, cpu_count

import yaml
import numpy as np
from tqdm import tqdm


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def process_scene(args):
    """Process a single scene to generate annotations."""
    scene_dir, cameras = args

    gt_dir = os.path.join(scene_dir, "gt")
    instance_to_actor = load_json(os.path.join(scene_dir, "id_mapping.json"))["instance_to_actor"]
    cav_info = load_json(os.path.join(scene_dir, "cav_info.json"))
    cav_id = str(cav_info["cav_id"])

    # Read from rgb folder
    rgb_dir = os.path.join(scene_dir, "rgb", cameras[0])
    frame_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])

    annotation_dir = os.path.join(scene_dir, "annotation")
    os.makedirs(annotation_dir, exist_ok=True)

    for rgb_filename in frame_files:
        frame_id = os.path.splitext(rgb_filename)[0]
        out_path = os.path.join(annotation_dir, f"{frame_id}.json")

        # Skip if JSON already exists
        if os.path.exists(out_path):
            continue

        gt_data = load_json(os.path.join(gt_dir, f"{frame_id}.json"))

        # Collect visibility counts: actor_id -> {camera: pixel_count}
        visible_actors = {}

        for camera in cameras:
            instance_path = os.path.join(scene_dir, "instance", camera, f"{frame_id}.npy")
            mask = np.load(instance_path)[:, :, :3].reshape(-1, 3)
            colors, counts = np.unique(mask, axis=0, return_counts=True)

            for color, count in zip(colors, counts):
                key = f"{color[0]}-{color[1]}-{color[2]}"
                actor_id = instance_to_actor.get(key, None)
                if actor_id is None or actor_id == -1 or str(actor_id) == cav_id:
                    continue
                if str(actor_id) not in gt_data["actors"]:
                    continue

                if actor_id not in visible_actors:
                    visible_actors[actor_id] = {cam: 0 for cam in cameras}
                visible_actors[actor_id][camera] = int(count)

        timestamp = gt_data["meta"]["timestamp"]
        datetime = gt_data["meta"]["datetime"]

        # Build annotation JSON
        annotation = {
            "meta": {
                "datetime": datetime,
                "timestamp": timestamp,
                "num_visible_actors": len(visible_actors),
            },
            "actors": {},
        }

        # Ego vehicle
        annotation["actors"]["ego"] = dict(gt_data["actors"][cav_id])

        # Background actors
        for actor_id, actor_dict in visible_actors.items():
            vcopy = dict(gt_data["actors"][str(actor_id)])
            vcopy["visibility"] = actor_dict
            annotation["actors"][str(actor_id)] = vcopy

        save_json(annotation, out_path)

    print(f"Process {scene_dir} completed", flush=True)
    return scene_dir


def process_all_scenes(data_path, cameras):
    """Process all scenes in parallel."""
    scenes = [os.path.join(data_path, d) for d in sorted(os.listdir(data_path)) if d.startswith("scene_")]
    print(f"Found {len(scenes)} scenes to process.")

    with Pool(cpu_count()) as pool:
        args_list = [(scene, cameras) for scene in scenes]
        for _ in tqdm(pool.imap_unordered(process_scene, args_list), total=len(scenes), desc="Processing Scenes"):
            pass

    print("Annotation generation completed.")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        data_dir = config["data"]["data_dir"]
        cameras = [cam["channel"] for cam in config["sensors"]]

    process_all_scenes(data_dir, cameras)
