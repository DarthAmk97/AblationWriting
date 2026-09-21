import os
from huggingface_hub import hf_hub_download
for l in [11, 12, 13, 14, 15]:
    os.makedirs("fetch_basis/Q20", exist_ok=True)
    hf_hub_download("amkkk/AblationWriting-H2-Q20-cone", "basis/full_layer%02d.npz" % l,
                    repo_type="model", local_dir="fetch_basis/Q20")
print("basis-done")