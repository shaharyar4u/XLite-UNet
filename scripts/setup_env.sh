#!/usr/bin/env bash
# Run ONCE inside the NGC PyTorch container. Installs the few extra packages into your home
# folder (the container itself is wiped when it stops). --no-deps keeps the container's own
# CUDA build of PyTorch untouched.
set -e
LIB=${PYLIBS:-$HOME/pylibs}
mkdir -p "$LIB"
pip install --no-deps --target "$LIB" monai nibabel captum einops
export PYTHONPATH="$LIB:$PYTHONPATH"
python - <<'PY'
import importlib
need = ["torch", "monai", "nibabel", "captum", "einops", "numpy", "scipy", "pandas",
        "matplotlib", "yaml", "tqdm"]
missing = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("missing:", missing or "none")
PY
echo "If anything is missing:  pip install --no-deps --target $LIB <name>   (pyyaml for yaml)"
