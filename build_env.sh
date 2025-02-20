#!/bin/bash
set -e # this will stop the script on first error

# get the name of the current conda environment
ENV_NAME=$(basename "$CONDA_PREFIX")

# print the name of the current conda environment to the terminal
echo "Building PJM  into the environment '$ENV_NAME'"


mamba install pytorch=2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip3 install fair-esm
pip3 install einops
pip3 install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip3 install torch_geometric==2.2.0
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip3 install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip3 install biotite
pip3 install lightning
