#!/bin/bash
# Install Python packages into the container overlay (run by the system on a node with
# internet). Installs into the SYSTEM Python; no virtual environments.
#
# torch: CUDA 12.6 build (works with any driver >= 525 via CUDA minor-version compatibility).
# transformers is pinned for a stable CLIP API; pyarrow reads the public parquet files;
# pillow decodes the images; safetensors stores encoder weights.

set -e

uv pip install --system --index-url https://download.pytorch.org/whl/cu126 "torch==2.12.0+cu126"
uv pip install --system numpy "transformers==4.57.1" safetensors pyarrow pillow
python -c "import torch, transformers, safetensors, pyarrow, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
