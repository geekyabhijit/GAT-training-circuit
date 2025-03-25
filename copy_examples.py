#!/usr/bin/env python3
import os
import shutil
import sys

def main():
    # Create the data directory structure
    data_dirs = {
        "simple_current_mirror": "data/simple_current_mirror",
        "cascaded_current_mirror": "data/cascaded_current_mirror",
        "differential_pair": "data/differential_pair",
        "single_stage_differential_amplifier": "data/single_stage_differential_amplifier",
        "two_stage_amplifier": "data/two_stage_amplifier",
        "ldo": "data/ldo",
        "bandgap_reference": "data/bandgap_reference",
        "comparator": "data/comparator"
    }
    
    # Create directories if they don't exist
    for directory in data_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Copy example netlists to their respective directories
    examples = {
        "simple_current_mirror.sp": data_dirs["simple_current_mirror"],
        "differential_pair.sp": data_dirs["differential_pair"]
    }
    
    for netlist, dest_dir in examples.items():
        src_path = os.path.join("examples", netlist)
        if os.path.exists(src_path):
            dst_path = os.path.join(dest_dir, netlist)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        else:
            print(f"Example netlist {src_path} not found.", file=sys.stderr)
    
    print("\nExample netlists have been copied to their respective data directories.")
    print("To train the model with these example netlists, run:")
    print("  python train_with_netlists.py")
    print("\nNote: This is just a demonstration with minimal examples.")
    print("For actual training, you should provide many more netlists for each topology.")

if __name__ == "__main__":
    main() 