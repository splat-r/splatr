import json
import gzip
import os
import glob
import natsort

# save_path = "rearrange/metrics_all/"
save_path = "rearrange/metrics_new/"
metric_files = natsort.natsorted(glob.glob(save_path + "*.json.gz"))
print(metric_files)
fixed_strict = 0
misplaced = 0
success = 0
energy = 0
fixed = 0
for metric_file in metric_files:
    with gzip.open(metric_file, "rb") as f:
        metrics = json.load(f)
    print("\n\n")
    print(metric_file)
    for key in metrics:
        print(metrics[key]['scene'])
        print(metrics[key]['unshuffle/prop_fixed_strict'])
        print(metrics[key]['unshuffle/prop_misplaced'])
        print(metrics[key]['unshuffle/success'])
        print(metrics[key]['unshuffle/energy_prop'])

        fixed_strict += metrics[key]['unshuffle/prop_fixed_strict']
        fixed += metrics[key]['unshuffle/prop_fixed']
        misplaced += metrics[key]['unshuffle/prop_misplaced']
        success += metrics[key]['unshuffle/success']
        energy += metrics[key]['unshuffle/energy_prop']
    print("\n\n")

fixed_strict /= len(metric_files)
fixed /= len(metric_files)
misplaced /= len(metric_files)
success /= len(metric_files)
energy /= len(metric_files)

print("final fixed strict : ", fixed_strict)
print("final fixed : ", fixed)
print("final misplaced : ", misplaced)
print("final success", success)
print("final energy", energy)

print(len(metric_files))
