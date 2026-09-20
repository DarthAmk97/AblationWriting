from huggingface_hub import HfApi
from pathlib import Path
tok = Path("/root/AblationWriting/.hf_token").read_text().strip()
api = HfApi(token=tok)
# create collection AblationWriting
try:
    col = api.create_collection(title="AblationWriting", description="Selective low-rank / concept-cone ablation of post-training writing collapse (H2 then H3, then completion). Training-free inference-time geometry. Writeup html in repos.", private=False)
    print(f"CREATED {col.slug} {col.url}")
except Exception as e:
    print(f"create failed (may exist): {e}")
    # list existing
    try:
        cols = api.list_collections(owner="amkkk")
        for c in cols:
            print(f"EXISTING {c.title} slug={c.slug} url={c.url}")
    except Exception as e2:
        print(f"list fail {e2}")
