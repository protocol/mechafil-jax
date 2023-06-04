#!/bin/bash

## NOTE: this is the preferred way to setup the environment for
# this project. The reason is that pip and conda do not seem to play
# nicely for installing JAX on macOS, which is required.

conda env create --file=environment.yaml
conda init bash
source activate mechafil_jax
pip install --no-deps -e . 