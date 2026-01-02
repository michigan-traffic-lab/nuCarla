"""
CARLA traffic management: spawning vehicles, pedestrians, and saving ground truth.
"""

import os
import json
import random
import datetime
import time
import argparse

import carla
import yaml
import numpy as np

from blueprints import (
    adult_pedestrian_pool,
    child_pedestrian_pool,
    car_vehicle_pool,
    truck_vehicle_pool,
    bus_vehicle_pool,
    motorcycle_vehicle_pool,
    bicycle_vehicle_pool,
)


def setup_mapping_camera(world, image_event):
    """Setup an overhead camera for instance segmentation mapping."""
    blueprint_library = world.get_blueprint_library()
    cam_bp = blueprint_library.find("sensor.camera.instance_segmentation")
    cam_bp.set_attribute("image_size_x", "800")
    cam_bp.set_attribute("image_size_y", "800")
    cam_bp.set_attribute("fov", "90")

    transform = carla.Transform(
        carla.Location(x=-15.0, y=0.0, z=1000.5),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )

    camera = world.spawn_actor(cam_bp, transform)

    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        image_event["image"] = array
        image_event["frame"] = image.frame

    camera.listen(callback)
    return camera


def save_background_instance(world, image_event, instance_to_actor):
    """Record the background instance color."""
    while image_event["image"] is None:
        world.tick()

    # Compute center pixel for background ID
    h, w, _ = image_event["image"].shape
    cy, cx = h // 2, w // 2
    r, g, b, _ = image_event["image"][cy, cx]
    background_id = f"{r}-{g}-{b}"
    instance_to_actor[background_id] = -1


def setup_cav(world, spawn_points):
    """Spawn the Connected Autonomous Vehicle (CAV)."""
    blueprint_library = world.get_blueprint_library()

    cav_blueprint = blueprint_library.find("vehicle.nissan.micra")
    cav_blueprint.set_attribute("role_name", "cav")
    cav = None

    while not cav:
        spawn_point = spawn_points.pop()
        cav = world.try_spawn_actor(cav_blueprint, spawn_point)

    print("Successfully spawned CAV")
    cav.set_simulate_physics(True)
    cav.set_autopilot(True)

    return cav


def setup_vehicle_actor(world, blueprint, spawn_points, image_event, instance_to_actor, vehicle_list):
    """Spawn a vehicle and record its instance segmentation color."""
    bp = random.choice(blueprint)

    if bp.has_attribute("color"):
        color = random.choice(bp.get_attribute("color").recommended_values)
        bp.set_attribute("color", color)

    # Temporarily spawn high above ground for segmentation mapping
    init_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=1000.0),
        carla.Rotation(0.0, 0.0, 0.0),
    )

    vehicle = world.try_spawn_actor(bp, init_transform)
    if not vehicle:
        print("Failed to spawn vehicle")
        return

    vehicle_list.append(vehicle)
    vehicle.set_simulate_physics(False)

    wait_for_record = True
    while wait_for_record:
        world.tick()

        img = image_event["image"]
        h, w, _ = img.shape
        center_col = w // 2

        # Extract the vertical center line
        center_line = img[:, center_col, :3]
        unique_colors = np.unique(center_line, axis=0)

        # Maximum 2 colors (background + vehicle)
        for color in unique_colors:
            r, g, b = color
            key = f"{r}-{g}-{b}"

            if key not in instance_to_actor:
                instance_to_actor[key] = vehicle.id
                wait_for_record = False

    # Move to final spawn position
    vehicle_spawn = spawn_points.pop()
    vehicle.set_transform(vehicle_spawn)


def setup_pedestrian_actor(world, image_event, instance_to_actor, pedestrian_list):
    """Spawn a pedestrian and record its instance segmentation color."""
    blueprint_library = world.get_blueprint_library()

    pedestrian_blueprints = [
        bp for bp in blueprint_library.filter("walker.pedestrian.*")
        if bp.id in adult_pedestrian_pool or bp.id in child_pedestrian_pool
    ]

    walker_bp = random.choice(pedestrian_blueprints)

    # Choose a random spawn location
    spawn_location = None
    while spawn_location is None:
        spawn_location = world.get_random_location_from_navigation()

    # Temporarily spawn high above ground for segmentation mapping
    init_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=1000.0),
        carla.Rotation(0.0, 0.0, 0.0),
    )

    walker = world.try_spawn_actor(walker_bp, init_transform)
    if not walker:
        print("Failed to spawn walker")
        return

    walker.set_simulate_physics(False)

    wait_for_record = True
    while wait_for_record:
        world.tick()
        img = image_event["image"]
        h, w, _ = img.shape
        center_col = w // 2

        # Extract the vertical center line
        center_line = img[:, center_col, :3]
        unique_colors = np.unique(center_line, axis=0)

        # Maximum 2 colors (background + actor)
        for color in unique_colors:
            r, g, b = color
            key = f"{r}-{g}-{b}"
            if key not in instance_to_actor:
                instance_to_actor[key] = walker.id
                wait_for_record = False

    # Move to final spawn location
    walker.set_transform(carla.Transform(spawn_location))
    pedestrian_list.append(walker)


def rpy_to_quaternion(roll, pitch, yaw):
    """Convert roll-pitch-yaw angles to quaternion (w, x, y, z)."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


def check_flipover(world, angle_threshold=15.0):
    """Detect and remove flipped or invalid actors from the world."""
    vehicles = world.get_actors().filter("vehicle.*")
    pedestrians = world.get_actors().filter("walker.pedestrian*")

    actors = list(vehicles) + list(pedestrians)

    for actor in actors:
        if not actor.is_alive:
            continue

        transform = actor.get_transform()
        roll = abs(transform.rotation.roll)
        pitch = abs(transform.rotation.pitch)

        if roll > angle_threshold or pitch > angle_threshold:
            print(f"[WARN] Removing flipped over actor {actor.id} (roll={roll:.1f}, pitch={pitch:.1f})")
            actor.destroy()


def save_ground_truth(world, ground_truth_dir, frame_id):
    """Save ground truth data for all actors in the world."""
    vehicles = world.get_actors().filter("vehicle.*")
    pedestrians = world.get_actors().filter("walker.pedestrian*")

    actors = list(vehicles) + list(pedestrians)
    num_actors = len(actors)

    # Meta data
    snapshot = world.get_snapshot()
    timestamp = int(snapshot.timestamp.elapsed_seconds * 1e6)  # Unix timestamp in microseconds

    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    time_str = now.strftime("%Y-%m-%d-%H-%M-%S%z")

    gt_data = {
        "meta": {
            "datetime": time_str,
            "timestamp": timestamp,
            "num_actors": num_actors,
        },
        "actors": {},
    }

    # Actor data
    for actor in actors:
        actor_type = actor.type_id

        # Get velocity vector
        vel = actor.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Determine category
        if actor_type in car_vehicle_pool:
            category = "vehicle.car"
        elif actor_type in truck_vehicle_pool:
            category = "vehicle.truck"
        elif actor_type in bus_vehicle_pool:
            category = "vehicle.bus.rigid"
        elif actor_type in motorcycle_vehicle_pool:
            category = "vehicle.motorcycle"
        elif actor_type in bicycle_vehicle_pool:
            category = "vehicle.bicycle"
        elif actor_type in adult_pedestrian_pool:
            category = "human.pedestrian.adult"
        elif actor_type in child_pedestrian_pool:
            category = "human.pedestrian.child"
        else:
            category = None
            print(actor_type, "unknown")

        transform = actor.get_transform()
        bbox = actor.bounding_box.extent

        # Negate y axis due to CARLA convention
        roll = transform.rotation.roll / 180.0 * np.pi
        pitch = -transform.rotation.pitch / 180.0 * np.pi
        yaw = -transform.rotation.yaw / 180.0 * np.pi

        w, x, y, z = rpy_to_quaternion(roll, pitch, yaw)

        gt_data["actors"][actor.id] = {
            "size": {
                "x": bbox.x * 2,
                "y": bbox.y * 2,
                "z": bbox.z * 2,
            },
            "translation": {
                "x": transform.location.x,
                "y": -transform.location.y,
                "z": transform.location.z,
            },
            "rotation": {
                "w": w,
                "x": x,
                "y": y,
                "z": z,
            },
            "speed": speed,
            "category": category,
        }

    # Save to file
    filename = os.path.join(ground_truth_dir, f"{frame_id}.json")
    with open(filename, "w") as f:
        json.dump(gt_data, f, indent=4)

    return num_actors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, default="scene_001")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        map_name = config["simulator"]["map_name"]
        sync_step = config["simulator"]["sync_step"]

        time_res = config["data"]["time_res"]
        save_freq = int(time_res / sync_step)

        num_cars = config["traffic"]["num_cars"]
        num_trucks = config["traffic"]["num_trucks"]
        num_buses = config["traffic"]["num_buses"]
        num_motorcycles = config["traffic"]["num_motorcycles"]
        num_bicycles = config["traffic"]["num_bicycles"]
        num_pedestrians = config["traffic"]["num_pedestrians"]
        data_dir = config["data"]["data_dir"]

    scene_dir = os.path.join(data_dir, args.scene_dir)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    print(f"Loading map: {map_name}")
    client.load_world(map_name)
    time.sleep(5.0)

    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = sync_step
    world.apply_settings(settings)

    image_event = {"image": None, "frame": -1}
    id_camera = setup_mapping_camera(world, image_event)

    instance_to_actor = {}

    save_background_instance(world, image_event, instance_to_actor)

    blueprint_library = world.get_blueprint_library()

    car_blueprints = [bp for bp in blueprint_library.filter("vehicle.*") if bp.id in car_vehicle_pool]
    truck_blueprints = [bp for bp in blueprint_library.filter("vehicle.*") if bp.id in truck_vehicle_pool]
    bus_blueprints = [bp for bp in blueprint_library.filter("vehicle.*") if bp.id in bus_vehicle_pool]
    motorcycle_blueprints = [bp for bp in blueprint_library.filter("vehicle.*") if bp.id in motorcycle_vehicle_pool]
    bicycle_blueprints = [bp for bp in blueprint_library.filter("vehicle.*") if bp.id in bicycle_vehicle_pool]

    vehicle_list = []
    pedestrian_list = []

    # Prepare all spawn points
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    cav = setup_cav(world, spawn_points)
    vehicle_list.append(cav)

    print("Begin spawning pedestrian actors...")
    world.set_pedestrians_cross_factor(0.0)  # No road crossing
    for _ in range(num_pedestrians):
        setup_pedestrian_actor(world, image_event, instance_to_actor, pedestrian_list)

    print("Begin spawning car actors...")
    for _ in range(num_cars):
        setup_vehicle_actor(world, car_blueprints, spawn_points, image_event, instance_to_actor, vehicle_list)

    print("Begin spawning truck actors...")
    for _ in range(num_trucks):
        setup_vehicle_actor(world, truck_blueprints, spawn_points, image_event, instance_to_actor, vehicle_list)

    print("Begin spawning bus actors...")
    for _ in range(num_buses):
        setup_vehicle_actor(world, bus_blueprints, spawn_points, image_event, instance_to_actor, vehicle_list)

    print("Begin spawning motorcycle actors...")
    for _ in range(num_motorcycles):
        setup_vehicle_actor(world, motorcycle_blueprints, spawn_points, image_event, instance_to_actor, vehicle_list)

    print("Begin spawning bicycle actors...")
    for _ in range(num_bicycles):
        setup_vehicle_actor(world, bicycle_blueprints, spawn_points, image_event, instance_to_actor, vehicle_list)

    num_unique_actors = len(set(instance_to_actor.values()))
    print(f"Recorded instance mapping for {num_unique_actors} unique actors (including background)")

    for ped in pedestrian_list:
        ped.set_simulate_physics(True)

        # Setup controller
        controller_bp = blueprint_library.find("controller.ai.walker")
        controller = world.spawn_actor(controller_bp, carla.Transform(), ped)
        world.tick()
        controller.start()

        # Set controller destination
        destination = world.get_random_location_from_navigation()
        if destination:
            controller.go_to_location(destination)
        controller.set_max_speed(random.uniform(1.0, 2.0))

    for veh in vehicle_list:
        veh.set_simulate_physics(True)
        veh.set_autopilot(True)

    # Create Traffic Manager
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    tm.global_percentage_speed_difference(50.0)  # Global 50% slower
    tm.set_global_distance_to_leading_vehicle(6.0)  # Car following distance

    print("Traffic setup complete.")

    ground_truth_dir = os.path.join(scene_dir, "gt")
    os.makedirs(ground_truth_dir, exist_ok=True)

    # Save instance_to_actor mapping
    with open(os.path.join(scene_dir, "id_mapping.json"), "w") as f:
        json.dump({"instance_to_actor": instance_to_actor}, f, indent=4)
    print(f"Saved dictionary to {os.path.join(scene_dir, 'id_mapping.json')}")

    cav_data = {"cav_id": cav.id}
    cav_info_dir = os.path.join(scene_dir, "cav_info.json")
    with open(cav_info_dir, "w") as f:
        json.dump(cav_data, f, indent=4)

    try:
        while True:
            frame_id = world.tick()
            check_flipover(world, angle_threshold=15.0)

            if frame_id % save_freq == 0:
                num_cars = save_ground_truth(world, ground_truth_dir, frame_id)
                print(f"Saved ground truth information for frame {frame_id} with {num_cars} actors")

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print("Cleaning up actors...")

        # Get all actors
        all_actors = world.get_actors()
        vehicles = all_actors.filter("vehicle.*")
        walkers = all_actors.filter("walker.pedestrian.*")
        controllers = all_actors.filter("controller.ai.walker")

        # Destroy controllers first
        for ctrl in controllers:
            if ctrl.is_alive:
                ctrl.stop()
                ctrl.destroy()

        # Then destroy walkers
        for ped in walkers:
            if ped.is_alive:
                ped.destroy()

        # Then destroy vehicles
        for veh in vehicles:
            if veh.is_alive:
                veh.destroy()

        print(f"Destroyed {len(controllers)} controllers, {len(walkers)} walkers, and {len(vehicles)} vehicles.")
        print("All actors cleaned up safely.")


if __name__ == "__main__":
    main()
