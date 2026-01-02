"""
CARLA sensor management: spawning camera sensors and saving sensor data.
"""

import os
import json
import time
import argparse

import carla
import yaml
import numpy as np
import cv2
from pyquaternion import Quaternion


def spawn_sensor(world, CAV, blueprint_id, transform, key_name, options, storage):
    """Spawn a sensor and attach it to the CAV with a callback to store data."""
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find(blueprint_id)

    for key, val in options.items():
        if blueprint.has_attribute(key):
            blueprint.set_attribute(key, str(val))

    sensor = world.spawn_actor(blueprint, transform, attach_to=CAV)

    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = array.reshape((image.height, image.width, 4))
        storage[key_name] = {"frame": image.frame, "data": img.copy()}

    sensor.listen(callback)
    return sensor, key_name


def spawn_camera_sensors(world, CAV, config, storage):
    """Spawn RGB and instance segmentation cameras based on config."""
    sensors = []
    channels = []
    calibration_dict = {}

    center_to_wheelbase = config["data"]["center_to_wheelbase"]

    for sensor_info in config["sensors"]:
        channel = sensor_info["channel"]
        location = sensor_info["transform"]["location"]
        rotation = sensor_info["transform"]["rotation"]
        options = sensor_info["options"]

        # Convert nuScenes quaternion to CARLA rotation
        nus_q = Quaternion(
            w=rotation.get("w", 1.0),
            x=rotation.get("x", 0.0),
            y=rotation.get("y", 0.0),
            z=rotation.get("z", 0.0),
        )
        yaw_q = Quaternion(axis=[0, 0, 1], degrees=-90)
        roll_q = Quaternion(axis=[1, 0, 0], degrees=-90)
        q_new = yaw_q * roll_q * nus_q.inverse
        yaw, pitch, roll = q_new.yaw_pitch_roll
        pitch, yaw, roll = np.degrees([pitch, yaw, roll])

        transform = carla.Transform(
            carla.Location(
                x=location["x"] - center_to_wheelbase,
                y=-location["y"],
                z=location["z"],
            ),
            carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
        )

        # Compute intrinsic matrix
        w_img, h_img = options["image_size_x"], options["image_size_y"]
        fov_rad = np.deg2rad(options["fov"])
        focal = w_img / (2.0 * np.tan(fov_rad / 2.0))
        intrinsic_matrix = [
            [focal, 0.0, w_img / 2.0],
            [0.0, focal, h_img / 2.0],
            [0.0, 0.0, 1.0],
        ]

        # Spawn RGB and instance segmentation sensors
        for blueprint_id, subfolder, key_suffix in [
            ("sensor.camera.rgb", "rgb", "rgb"),
            ("sensor.camera.instance_segmentation", "instance", "instance"),
        ]:
            key_name = f"{channel}_{key_suffix}"
            sensor, _ = spawn_sensor(
                world,
                CAV,
                blueprint_id,
                transform,
                key_name,
                options,
                storage,
            )
            sensors.append(sensor)
            channels.append((subfolder, channel, key_name))

            calibration_dict[channel] = {
                "intrinsic": intrinsic_matrix,
                "extrinsic": sensor_info["transform"],
            }

    return sensors, channels, calibration_dict


def save_all_sensors(frame_id, save_dir, storage, channels):
    """Save all sensor data for a given frame. Returns 1 if saved, 0 otherwise."""
    # Check if all data available
    if not all(storage.get(key_name) is not None for _, _, key_name in channels):
        return 0

    # Save all
    for subfolder, ch_name, key_name in channels:
        content = storage[key_name]
        img = content["data"]
        out_dir = os.path.join(save_dir, subfolder, ch_name)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{frame_id}.jpg"), img[:, :, :3])

        # Save raw files for later annotation
        if subfolder == "instance":
            np.save(os.path.join(out_dir, f"{frame_id}.npy"), img)

    print(f"Saved all sensor data for frame {frame_id}")

    # Clear after saving
    for _, _, key_name in channels:
        storage[key_name] = None

    return 1


def spawn_collision_sensor(world, cav, save_dir):
    """
    Attach a collision sensor to the CAV.
    When a collision occurs, create an empty collision.txt file in the scene folder.
    """
    blueprint_library = world.get_blueprint_library()
    collision_bp = blueprint_library.find("sensor.other.collision")
    sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=cav)

    def on_collision(event):
        file_path = os.path.join(save_dir, "collision.txt")
        if not os.path.exists(file_path):
            open(file_path, "w").close()
            print("[INFO] Collision detected! Created collision.txt")

    sensor.listen(on_collision)
    return sensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Sensor Manager")
    parser.add_argument("--scene_dir", type=str, default="scene_001")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        data_dir = config["data"]["data_dir"]
        nbr_samples = config["data"]["nbr_samples"]
        delta_T = config["simulator"]["sync_step"]
        time_res = config["data"]["time_res"]
        save_freq = int(time_res / delta_T)

    scene_dir = os.path.join(data_dir, args.scene_dir)

    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    # Find CAV
    cav = None
    attempts = 0
    while cav is None:
        print("CAV not found. Waiting...")
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name", "") == "cav":
                cav = actor

        attempts += 1
        if attempts >= 100:
            print("[ERROR] CAV not found after 100 attempts. Exiting program.")
            exit(1)

        time.sleep(0.1)

    print("Found CAV, attaching sensors...")

    # Add camera sensors
    camera_storage = {}
    sensors, channels, calibration_dict = spawn_camera_sensors(world, cav, config, camera_storage)

    # Add collision sensor
    collision_sensor = spawn_collision_sensor(world, cav, scene_dir)

    with open(os.path.join(scene_dir, "calibrated_sensor.json"), "w") as f:
        json.dump(calibration_dict, f, indent=4)

    print(f"[INFO] Started saving sensor data into {scene_dir}...")

    try:
        total_samples = 0
        while True:
            snapshot = world.get_snapshot()
            frame_id = snapshot.frame
            if frame_id % save_freq == 0:
                total_samples += save_all_sensors(frame_id, scene_dir, camera_storage, channels)
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("[INFO] Interrupted, stopping sensor logging.")

    finally:
        for s in sensors:
            s.stop()
            s.destroy()
        if collision_sensor:
            collision_sensor.stop()
            collision_sensor.destroy()
        print("[INFO] All sensors destroyed. Exiting cleanly.")
