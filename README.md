# CARLA Data Collection Pipeline

A framework for automated traffic simulation and multi-camera data collection in the CARLA simulator. The pipeline spawns traffic actors, attaches camera sensors to an ego vehicle, and extracts ground truth annotations for all visible actors. While designed as a minimal setup for single-scenario data collection, the modular architecture allows users to extend it for continuous collection and annotation workflows.

---

## Repository Structure

| File | Description |
|------|-------------|
| `config.yaml` | Central configuration file for simulation parameters, traffic actor counts, output directories, and camera sensor specifications. |
| `blueprints.py` | Defines CARLA blueprint pools for different actor categories (pedestrians, cars, trucks, buses, motorcycles, bicycles). |
| `traffic.py` | Spawns traffic actors and records their metadata for downstream processing. |
| `sensors.py` | Attaches a 6-camera sensor suite to the ego vehicle. Each camera pair consists of an RGB camera for raw images and an instance segmentation camera for ground truth extraction. |
| `annotation.py` | Detects visible traffic actors in each frame using instance segmentation images, retrieves corresponding actor metadata, and generates final annotations. |
| `visualize.py` | Renders 3D bounding boxes on RGB images for visual verification. Transforms actor coordinates from world space to camera space and projects them onto 2D images with category-based color coding. |

---

## Limitations

CARLA's instance segmentation camera returns Unreal Engine object IDs rather than CARLA actor IDs, with no direct API mapping between them. As a workaround, we position a dummy instance segmentation camera high above the ground (see `traffic.py`) and temporarily teleport each newly spawned actor to this isolated location to capture its Unreal ID, which is then mapped to the corresponding CARLA actor ID. If you know a better approach, please open an issue to share with the community!

---

## Installation

### Tested Environment

The following environment has been tested; others may work as well.

- **Python**: 3.10
- **OS**: Ubuntu 22.04

### CARLA Download

Download our customized [CARLA 0.9.16](https://huggingface.co/datasets/zhijieq/nuCarla/blob/main/CARLA_0.9.16.zip) package from Hugging Face. The only difference from the official release is the removal of static vehicle meshes. For more information, see Section 3.6 of our [paper](https://arxiv.org/pdf/2511.13744).

### Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Data Collection

Run the following steps in separate terminal windows. There is no automatic exit trigger; press Ctrl+C to stop recording once enough frames have been collected.

**1. Launch CARLA Server**

```bash
./path/to/CarlaUE4.sh
```

**2. Spawn Traffic**

Wait for all traffic actors to spawn and begin moving, then proceeed to next step.

```bash
python traffic.py
```

**3. Attach Sensors**

Attaches camera sensors to the ego vehicle and starts recording.

```bash
python sensors.py
```

---

### Data Processing

**1. Generate Annotations**

Extracts visible actor information and compiles ground truth labels. This step can take some time due to pixel-level processing.

```bash
python annotation.py
```

**2. Visualize Results (Optional)**

After annotation is complete, run this script to render 3D bounding boxes overlaid on saved frames for visual verification.

```bash
python visualize.py
```