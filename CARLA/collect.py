import subprocess
import time
import os
import sys
import carla
import yaml
import random
import argparse


weathers = [
    ("ClearNoon", carla.WeatherParameters.ClearNoon),
    ("CloudyNoon", carla.WeatherParameters.CloudyNoon),
    ("WetNoon", carla.WeatherParameters.WetNoon),
    ("WetCloudyNoon", carla.WeatherParameters.WetCloudyNoon),
    ("MidRainyNoon", carla.WeatherParameters.MidRainyNoon),
    ("HardRainNoon", carla.WeatherParameters.HardRainNoon),
    ("SoftRainNoon", carla.WeatherParameters.SoftRainNoon),
    ("ClearSunset", carla.WeatherParameters.ClearSunset),
    ("CloudySunset", carla.WeatherParameters.CloudySunset),
    ("WetSunset", carla.WeatherParameters.WetSunset),
    ("WetCloudySunset", carla.WeatherParameters.WetCloudySunset),
    ("MidRainSunset", carla.WeatherParameters.MidRainSunset),
    ("HardRainSunset", carla.WeatherParameters.HardRainSunset),
    ("SoftRainSunset", carla.WeatherParameters.SoftRainSunset),
]


def cleanup():
    print("\n[INFO] Cleaning up child processes...")
    subprocess.run(["pkill", "-9", "-f", "setup_traffic.py"], check=False)
    subprocess.run(["pkill", "-9", "-f", "sensors.py"], check=False)
    subprocess.run(["pkill", "-9", "-f", "CarlaUE4"], check=False)


def main(scene_dir, map_name):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        carla_path = config["parameters"]["carla_path"]
        source_dir = config["parameters"]["source_dir"]

    start_time = time.time()
    scene = f"scene_{int(start_time * 1e6)}"
    print(f"[INFO] Scene name: {scene}")

    # === Create scene folder ===
    scene_dir = os.path.join(source_dir, map_name, scene)
    os.makedirs(scene_dir, exist_ok=True)

    ground_truth_dir = os.path.join(scene_dir, "GT")
    os.makedirs(ground_truth_dir, exist_ok=True)

    try:
        # 1. Launch CARLA
        carla_proc = subprocess.Popen(
            [carla_path, "-RenderOffScreen"],
            preexec_fn=os.setsid
        )
        print("[INFO] CarlaUE4 started")
        time.sleep(5.0)

        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        client.load_world(map_name)
        print(f"Loaded {map_name}.")

        name, weather = random.choice(weathers)
        print(f"Selected weather: {name}")

        # Save weather.txt in scene folder
        weather_path = os.path.join(scene_dir, "weather.txt")
        with open(weather_path, "w") as wf:
            wf.write(name + "\n")

        world = client.get_world()
        world.set_weather(weather)
        time.sleep(2.0)

        # 2. Launch traffic
        traffic_proc = subprocess.Popen(
            ["python3", "setup_traffic.py", "--scene_dir", scene_dir],
            preexec_fn=os.setsid
        )
        print("[INFO] setup_traffic.py started")
        
        time.sleep(15.0)

        # 3. Launch sensors
        sensors_proc = subprocess.Popen(
            ["python3", "sensors.py", "--scene_dir", scene_dir],
            preexec_fn=os.setsid
        )
        sensors_proc.wait()
        print("[INFO] sensors collection finished")

        cleanup()
        elapsed = time.time() - start_time
        print(f"[INFO] ===== Run complete (elapsed {elapsed:.1f}s) =====")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected, stopping...")
        cleanup()
        sys.exit(0)

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, default="scene_001")
    parser.add_argument("--map_name", type=str, default="Town01")
    args = parser.parse_args()

    main(args.scene_dir, args.map_name)
