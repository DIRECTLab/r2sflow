#!/usr/bin/env bash
set -e

# `conda activate` only works in a shell where conda's hook is loaded
eval "$(conda shell.bash hook)"

conda create -n r2sflow python=3.10 -y
conda activate r2sflow
pip install -r requirements.txt
# change line below for your CUDA version, see https://pytorch.org/
pip3 install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
conda install -c conda-forge sparsehash -y
CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include \
  pip install --no-build-isolation git+https://github.com/mit-han-lab/torchsparse.git@v2.0.0

# you may also need to change this CUDA version for your system, see https://whl.natten.org
pip install natten==0.21.7+torch2130cu132 -f https://whl.natten.org
# natten silently falls back to a slow, memory-hungry path if its kernels don't
# match the installed torch; this should print True
python -c "import natten; print('natten kernels loaded:', natten.HAS_LIBNATTEN)"
