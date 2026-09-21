import os
from huggingface_hub import hf_hub_download
mods = ["L1", "O1", "Q08", "Q20", "G2"]
layers = {"L1": [8, 9, 10, 11], "O1": [8, 9, 10, 11], "Q08": [18, 19, 20, 21, 22],
          "Q20": [11, 12, 13, 14, 15], "G2": [11, 12, 13, 14, 15, 16]}
for m in mods:
    os.makedirs("fetch_spectra/" + m, exist_ok=True)
    for l in layers[m]:
        hf_hub_download("amkkk/AblationWriting-H2-" + m + "-cone",
                        "basis/full_layer" + str(l).zfill(2) + ".npz",
                        repo_type="model", local_dir="fetch_spectra/" + m)
print("npz-done")