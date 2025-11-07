#!/bin/bash

set -e

num_run=90
maps=("Town10HD" "McityMap_Main")

# Loop through each map
for map in "${maps[@]}"; do
    for ((i=1; i<=num_run; i++)); do
        echo "[INFO] --- Run ${i}/${num_run} for map ${map} ---"

        # Generate timestamp for recording
        ts=$(python3 -c "import time; print(int(time.time() * 1e6))")
        save_dir="scene_${ts}"

        # Run collection with map argument
        python3 collect.py --scene_dir "${save_dir}" --map "${map}"

        echo "[INFO] Run ${i} finished."
        sleep 3
    done
done

echo "[INFO] All maps completed successfully."