#!/bin/bash

# Fit Phase 1
uv run rgen_schedule fit \
  --config src/ananke_abm/models/gen_schedule/dataio/configs/phase1.yaml \
  --output-dir src/output/new_rgen/runs/base_cnn \
  --seed 123
