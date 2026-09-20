from huggingface_hub import HfApi
from pathlib import Path
tok = Path("/root/AblationWriting/.hf_token").read_text().strip()
api = HfApi(token=tok)
print(api.whoami())
