# File: main.py
# Description:
# End-to-end coordinator for the Wind-Weighted AOD Transport Efficiency (WWTE) pipeline.
# Runs both the spatial transport calculation and the premium geospatial visualizer.
#
# Author: Hossein Lotfi — Research Scientist
#
# How to run:
# python main.py
#

import os
import sys
import subprocess
import time
import json

def run_script(script_path):
    """
    Executes a python script inside the 'resources' directory and prints output in real-time.
    """
    script_name = os.path.basename(script_path)
    print("\n" + "="*80)
    print(f"🚀 RUNNING PIPELINE COMPONENT: {script_name}")
    print("="*80)
    
    start_time = time.time()
    try:
        # Launch process using the active python interpreter
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Real-time stdout streaming
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
            
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n✅ {script_name} finished successfully in {elapsed:.2f} seconds.")
            return True
        else:
            print(f"\n❌ {script_name} crashed with return code {process.returncode} after {elapsed:.2f} seconds.")
            return False
            
    except Exception as e:
        print(f"\n❌ Failed to run {script_name}: {e}")
        return False

def main():
    """
    Main entry point coordinates both analysis and plotting components end-to-end.
    """
    config_path = os.path.join("config", "config.json")
    wind_banner = "unknown"
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        wind_banner = str(cfg.get("active_wind_type", "unknown"))
    except Exception:
        pass

    print("\n" + "#"*80)
    print(f"      WWTE PIPELINE END-TO-END COORDINATOR (Wind: {wind_banner})")
    print("#"*80)
    
    pipeline_start = time.time()
    
    # 1. Run final spatial transport analysis
    analysis_script = os.path.join("resources", "wwte_aod_index_analysis.py")
    if not run_script(analysis_script):
        print("\n❌ Pipeline aborted: Spatial analysis stage failed.")
        sys.exit(1)
        
    # 2. Run advanced plotting script
    plotting_script = os.path.join("resources", "plot_climatology.py")
    if not run_script(plotting_script):
        print("\n❌ Pipeline aborted: Plotting stage failed.")
        sys.exit(1)
        
    total_elapsed = time.time() - pipeline_start
    print("\n" + "#"*80)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS!")
    print(f"📂 Inputs directory:  inputs/")
    print(f"📂 Output results:   outputs/results/")
    print(f"📂 Output maps:      outputs/plots/")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
