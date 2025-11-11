#!/bin/bash

# Fit
uv run rgen_schedule fit \
  --config src/ananke_abm/models/gen_schedule/dataio/configs/phase2.yaml \
  --output-dir src/output/new_rgen/runs/auto_pmd \
  --seed 123