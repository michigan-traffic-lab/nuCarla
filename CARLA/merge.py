import os
import json
import shutil


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


def merge_json_files(root_path, subfolders, save_path, version_folder="v1.0-trainval"):
    """
    Merge specific JSON files across multiple subfolders and save with NuScenes-style formatting.
    """
    merged_dir = os.path.join(save_path, version_folder)
    os.makedirs(merged_dir, exist_ok=True)

    target_files = [
        "ego_pose.json",
        "instance.json",
        "sample.json",
        "sample_annotation.json",
        "scene.json",
        "sample_data.json"
    ]

    for file_name in target_files:
        merged_list = []

        for subfolder in subfolders:
            json_path = os.path.join(root_path, subfolder, version_folder, file_name)
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_list.extend(data)
                    else:
                        print(f"[WARNING] {json_path} does not contain a list.")
            else:
                print(f"[WARNING] Missing: {json_path}")

        out_path = os.path.join(merged_dir, file_name)
        save_json_nuscenes_style(merged_list, out_path)
        print(f"[INFO] Saved merged {file_name} to {out_path} ({len(merged_list)} entries)")


def merge_samples_folders(root_path, subfolders, save_path):
    """
    Merge sample sensor folders across all subfolders by copying all files.
    """
    sensor_names = [
        "CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT", "CAM_BACK_LEFT",
        "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "LIDAR_TOP"
    ]

    samples_dir = os.path.join(save_path, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    for sensor in sensor_names:
        dst_sensor_dir = os.path.join(samples_dir, sensor)
        os.makedirs(dst_sensor_dir, exist_ok=True)

        for subfolder in subfolders:
            src_sensor_dir = os.path.join(root_path, subfolder, "samples", sensor)
            if os.path.exists(src_sensor_dir):
                for file_name in os.listdir(src_sensor_dir):
                    src_file = os.path.join(src_sensor_dir, file_name)
                    dst_file = os.path.join(dst_sensor_dir, file_name)
                    if os.path.exists(dst_file):
                        name, ext = os.path.splitext(file_name)
                        dst_file = os.path.join(dst_sensor_dir, f"{subfolder}_{name}{ext}")
                    shutil.copy2(src_file, dst_file)
                print(f"[INFO] Copied all files from {src_sensor_dir}")
            else:
                print(f"[WARNING] Missing sensor folder: {src_sensor_dir}")

    print(f"[INFO] Completed merging samples into {samples_dir}")


if __name__ == "__main__":
    root_path = "/media/zhijie/Disk2/distributed"
    save_path = "/media/zhijie/Disk3/nuCarla_test"
    subfolders = ["Town10", "Mcity"]

    merge_json_files(root_path, subfolders, save_path, version_folder='v1.0-trainval')
    merge_samples_folders(root_path, subfolders, save_path)
