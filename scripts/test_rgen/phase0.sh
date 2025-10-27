#!/bin/bash

# 1) Prepare 10-min grid from your small dev CSV
uv run rgen_schedule prepare \
  --activities "src/data/traj_processed/small_activities_homebound_wd.csv" \
  --grid 10 \
  --out src/output/rgen/train_10min.npz