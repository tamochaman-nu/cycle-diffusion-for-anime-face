import subprocess
import os
import numpy as np

def run_experiment(cutoff, end_step):
    # cutoff format: 0.1 -> 010, 1.0 -> 100
    cutoff_str = f"{int(round(cutoff * 100)):03d}"
    # end_step format: 10 -> 010
    step_str = f"{int(end_step):03d}"
    
    run_name = f"translate_ffhq256_to_anime256_10000_eta0001_free_inv_fbsdiff{cutoff_str}_{step_str}stp_100stp_020rstp_wvloss001"
    
    env = os.environ.copy()
    env["RUN_NAME"] = run_name
    env["FBSDIFF_CUTOFF"] = str(cutoff)
    env["FBSDIFF_END_STEP"] = str(int(end_step))
    
    print(f"\n>>> Running: {run_name} (cutoff={cutoff}, end_step={end_step})")
    
    cmd = ["GPU_ID=0", "docker", "compose", "run", "--rm", "-e", f"RUN_NAME={run_name}", "-e", f"FBSDIFF_CUTOFF={cutoff}", "-e", f"FBSDIFF_END_STEP={int(end_step)}", "app-wavelet"]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {run_name}: {e}")

def main():
    cutoffs = [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)]
    end_steps = [20]
    
    for cutoff in cutoffs:
        for end_step in end_steps:
            run_experiment(cutoff, end_step)

if __name__ == "__main__":
    main()
