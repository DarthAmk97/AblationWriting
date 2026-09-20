#!/bin/bash
# H2 AblationWriting env install — runs on VAST RTX 3090
# Operates ONLY within /root/AblationWriting
set -x
export HF_HOME=/root/AblationWriting/.hf_cache
export HF_HUB_CACHE=/root/AblationWriting/.hf_cache/hub
export HUGGINGFACE_HUB_CACHE=/root/AblationWriting/.hf_cache/hub
export HF_DATASETS_CACHE=/root/AblationWriting/.hf_cache/datasets
export UV_CACHE_DIR=/root/AblationWriting/.hf_cache/uv_cache
export PIP_CACHE_DIR=/root/AblationWriting/.hf_cache/pip_cache
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"
mkdir -p /root/AblationWriting/logs
LOG=/root/AblationWriting/logs/install.log
exec > >(tee -a "$LOG") 2>&1
echo "=== H2 install start $(date -u) ==="
source /venv/main/bin/activate
python --version
which uv
uv pip install --upgrade pip setuptools wheel
echo "=== torch cu128 ==="
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
echo "=== core stack ==="
uv pip install "transformers>=4.55" datasets accelerate safetensors sentencepiece tokenizers huggingface_hub pyarrow pandas numpy scipy scikit-learn tqdm pyyaml matplotlib seaborn einops
echo "=== eval helpers ==="
uv pip install lm-eval || echo "lm-eval optional fail"
uv pip install ifeval 2>&1 | tail -n 5 || echo "ifeval optional"
echo "=== verify ==="
python -c "import torch, transformers, datasets; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__); print('datasets', datasets.__version__)"
nvidia-smi | head -n 20
echo "=== H2 install done $(date -u) ==="
