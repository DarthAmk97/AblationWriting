from huggingface_hub import HfApi
from pathlib import Path
tok = Path("/root/AblationWriting/.hf_token").read_text().strip()
api = HfApi(token=tok)
try:
    col = api.create_collection(title="AblationWriting", description="H2/H3 selective ablation of post-training writing collapse. Training-free.", private=False)
    print(f"CREATED {col.slug} {col.url}")
except Exception as e:
    print(f"create failed: {e}")
    cols = api.list_collections(owner="amkkk")
    for c in cols:
        print(f"EXISTING {c.title} slug={c.slug}")
