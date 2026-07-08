#!/bin/bash
set -e

# Acquire a lock file to prevent parallel docker compose runs
LOCKFILE="./.grid_search.lock"
echo "Waiting for lock on ${LOCKFILE} to prevent parallel runs..."
exec 9>"$LOCKFILE"
flock -x 9
echo "Lock acquired. Starting experiments."

for i in {0..10}; do
    # Calculate cutoff: 0 -> 0.0, 1 -> 0.1, ..., 10 -> 1.0
    if [ "$i" -eq 10 ]; then
        cutoff="1.0"
    else
        cutoff="0.$i"
    fi

    # cutoff_str: 0.1 -> 010, 1.0 -> 100
    cutoff_val=$(( i * 10 ))
    cutoff_str=$(printf "%03d" "$cutoff_val")

    for ((end_step=0; end_step<=250; end_step+=25)); do
        step_str=$(printf "%03d" "$end_step")
        
        run_name="translate_ffhq256_to_anime256_100000_eta08_free_inv_fbsdiff${cutoff_str}_${step_str}stp_250stp_025rstp"
        exp_name="danbooru_100000steps_no_freeinv_rev"

        OUTPUT_DIR="output/${exp_name}/${run_name}"
        DEBUG_DIR="debug/${exp_name}/${run_name}"
        
        # Check if experiment has already completed successfully
        if [ -f "${OUTPUT_DIR}/eval_results.json" ]; then
            echo ">>> Skipping already completed experiment: ${run_name}"
            continue
        fi
        
        echo -e "\n>>> Running: ${run_name} (cutoff=${cutoff}, end_step=${end_step})"
        
        GPU_ID=2 docker compose run --rm \
            -e RUN_NAME="${run_name}" \
            -e FBSDIFF_CUTOFF="${cutoff}" \
            -e FBSDIFF_END_STEP="${end_step}" \
            -e OUTPUT_DIR="${OUTPUT_DIR}" \
            -e INTERMEDIATE_DIR="${DEBUG_DIR}" \
            app
        
        # Cooling down delay to prevent overheating and allow OS/GPU to reclaim VRAM
        sleep 5
    done
done
