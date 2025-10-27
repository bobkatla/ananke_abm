#!/bin/bash

uv run rgen_schedule compute-pds \
  --grid src/output/rgen/train_10min.npz \
  --out src/output/rgen/runs/exp_phase2/phase2 \
  --grid-min 10 \
  --purpose-json src/output/rgen/train_10min_purpose_map.json
