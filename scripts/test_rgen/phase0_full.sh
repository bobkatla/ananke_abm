#!/bin/bash

# 1) Prepare 5-min grid from your small dev CSV
uv run rgen_schedule prepare \
  --activities "src/data/schedule_processed/24h_full_activities_homebound_wd.csv" \
  --grid 5 \
  --out src/output/new_rgen/full_train_5min.npz \
  --val-frac 0.1 \
  --seed 123

uv run rgen_schedule compute-pds \
  --grid src/output/new_rgen/full_train_5min.npz \
  --out src/output/new_rgen/runs/exp_phase2/phase2_full \
  --grid-min 5 \
  --purpose-json src/output/new_rgen/full_train_5min_purpose_map.json
