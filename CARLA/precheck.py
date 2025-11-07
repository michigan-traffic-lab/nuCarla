import os
import json
import shutil
from tqdm import tqdm
import yaml


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def process_scene(scene_dir, required_items, cameras):
    scene_name = os.path.basename(scene_dir)
    reasons = []

    # --- Check required items ---
    missing = [item for item in required_items if not os.path.exists(os.path.join(scene_dir, item))]
    if missing:
        reasons.append(f"missing items: {missing}")

    # --- Check for collision.txt ---
    collision_file = os.path.join(scene_dir, "collision.txt")
    if os.path.exists(collision_file):
        reasons.append("collision.txt detected")

    # --- Check RGB folder structure ---
    rgb_dir = os.path.join(scene_dir, "RGB")
    if os.path.exists(rgb_dir):
        for cam in cameras:
            cam_dir = os.path.join(rgb_dir, cam)
            if not os.path.exists(cam_dir):
                reasons.append(f"missing RGB camera folder {cam}")
                continue

            files = [f for f in os.listdir(cam_dir) if os.path.isfile(os.path.join(cam_dir, f)) and f.endswith(".jpg")]
            if len(files) != 40:
                reasons.append(f"{cam} in RGB has {len(files)} files (expected 40)")
    else:
        reasons.append("RGB folder missing")

    # --- Check INSTANCE folder structure ---
    instance_dir = os.path.join(scene_dir, "INSTANCE")
    if os.path.exists(instance_dir):
        for cam in cameras:
            cam_dir = os.path.join(instance_dir, cam)
            if not os.path.exists(cam_dir):
                reasons.append(f"missing INSTANCE camera folder {cam}")
                continue
            files = [f for f in os.listdir(cam_dir) if os.path.isfile(os.path.join(cam_dir, f))]
            if len(files) != 80:
                reasons.append(f"{cam} in INSTANCE has {len(files)} files (expected 80)")
    else:
        reasons.append("INSTANCE folder missing")

    # --- Check GT and ego vehicle consistency ---
    gt_dir = os.path.join(scene_dir, "GT")
    cav_info_path = os.path.join(scene_dir, "cav_info.json")

    if os.path.exists(gt_dir) and os.path.exists(cav_info_path):
        cav_info = load_json(cav_info_path)
        cav_id = str(cav_info.get("cav_id", ""))
        cam_front_dir = os.path.join(rgb_dir, "CAM_FRONT")
        missing_gt_count = 0
        missing_ego_count = 0

        if os.path.exists(cam_front_dir):
            rgb_frames = [os.path.splitext(f)[0] for f in os.listdir(cam_front_dir) if f.endswith(".jpg")]
            for frame in rgb_frames:
                gt_path = os.path.join(gt_dir, f"{frame}.json")
                if not os.path.exists(gt_path):
                    missing_gt_count += 1
                    continue
                gt_data = load_json(gt_path)
                actors = gt_data.get("actors", {})
                if cav_id not in actors:
                    missing_ego_count += 1

            if missing_gt_count > 0:
                reasons.append(f"missing {missing_gt_count} GT files")
            if missing_ego_count > 0:
                reasons.append(f"{missing_ego_count} GT files missing ego vehicle id {cav_id}")
        else:
            reasons.append("CAM_FRONT folder missing (for GT check)")
    else:
        reasons.append("missing GT or cav_info.json for vehicle check")

    # Return deletion info
    return (scene_name, reasons) if reasons else None


def process_all_scenes(data_path, required_items, cameras):
    scenes = [os.path.join(data_path, d) for d in sorted(os.listdir(data_path)) if d.startswith("scene_")]
    print(f"Found {len(scenes)} scenes to check.")
    record = []

    for scene in tqdm(scenes, desc="Prechecking Scenes"):
        result = process_scene(scene, required_items, cameras)
        if result:
            record.append((scene, result[1]))

    print(f"\nPrecheck completed: {len(scenes)} total scenes, {len(record)} problematic scenes.")

    if record:
        print("\nProblematic scenes detected (to be deleted if confirmed):")
        for s, reasons in record:
            print(f"  - {os.path.basename(s)}:")
            for r in reasons:
                print(f"      - {r}")

        confirm = input("\nDo you want to delete all these problematic scenes? (Y/N): ").strip().lower()
        if confirm == "y":
            print("\nDeleting problematic scenes:")
            for s, _ in record:
                print(f"  - {os.path.basename(s)}", flush=True)
                shutil.rmtree(s)
            print(f"Deleted {len(record)} problematic scenes.")
        else:
            print("\nDeletion skipped by user confirmation.")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        source_dir = config["parameters"]["source_dir"]
        cameras = [cam["channel"] for cam in config["sensors"]]

    required_items = ["cav_info.json", "id_mapping.json", "GT", "RGB", "INSTANCE"]
    map_dir = os.path.join(source_dir, "Town10HD")

    process_all_scenes(map_dir, required_items, cameras)