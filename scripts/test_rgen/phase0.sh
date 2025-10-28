#!/bin/bash

# 1) Prepare 10-min grid from your small dev CSV
uv run rgen_schedule prepare \
  --activities "src/data/traj_processed/small_activities_homebound_wd.csv" \
  --grid 10 \
  --out src/output/rgen/train_10min.npz \
  --val-frac 0.1 \
  --seed 123

uv run rgen_schedule compute-pds \
  --grid src/output/rgen/train_10min.npz \
  --out src/output/rgen/runs/exp_phase2/phase2 \
  --grid-min 10 \
  --purpose-json src/output/rgen/train_10min_purpose_map.json
