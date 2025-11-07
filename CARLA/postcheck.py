import os
import json
import yaml
import shutil
from tqdm import tqdm


def count_actors_in_scene(scene_path):
    """Count average number of actors across all frames (JSON files) in a scene."""
    annotation_dir = os.path.join(scene_path, "annotation")
    if not os.path.isdir(annotation_dir):
        return 0

    total_actors = 0
    frame_count = 0

    for file_name in os.listdir(annotation_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(annotation_dir, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                if "actors" in data:
                    total_actors += len(data["actors"])
                    frame_count += 1

    if frame_count == 0:
        return 0

    return total_actors / frame_count


def main(root_dir):
    """Iterate over all scenes, compute average actors, print results, and handle deletions."""
    scene_actor_counts = {}
    record = []

    scenes = [s for s in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, s)) and s.startswith("scene_")]

    print("\n[INFO] Counting average actors per scene...")
    for scene_name in tqdm(scenes, desc="Processing scenes", unit="scene"):
        scene_path = os.path.join(root_dir, scene_name)
        avg = count_actors_in_scene(scene_path)
        scene_actor_counts[scene_name] = avg

        if avg > 71.7:
            record.append((scene_path, avg))

    # Sort by average actor count (ascending)
    sorted_scenes = sorted(scene_actor_counts.items(), key=lambda x: x[1])

    print("\n=== Average actor Count per Scene (Low → High) ===")
    for scene_name, avg in sorted_scenes:
        print(f"{scene_name}: {avg:.2f}")

    # Print problematic scenes
    if record:
        print("\n=== Problematic Scenes (Avg < 10 actors) ===")
        for s, avg in record:
            print(f"  - {os.path.basename(s)} (avg={avg:.2f}, reason: too few actors)")

        confirm = input("\nDo you want to delete all these problematic scenes? (Y/N): ").strip().lower()
        if confirm == "y":
            print("\nDeleting problematic scenes:")
            for s, _ in tqdm(record, desc="Deleting", unit="scene"):
                shutil.rmtree(s)
            print(f"\nDeleted {len(record)} problematic scenes.")
        else:
            print("\nDeletion skipped by user confirmation.")
    else:
        print("\nNo problematic scenes found.")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        source_dir = config["parameters"]["source_dir"]

    source_dir = '/media/zhijie/Disk2/data/Mcity'

    main(source_dir)