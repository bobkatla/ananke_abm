#!/bin/bash

# Fit Phase 1
uv run rgen_schedule fit \
  --config src/ananke_abm/models/gen_schedule/dataio/configs/phase1.yaml \
  --output-dir src/output/new_rgen/runs/base_cnn \
  --seed 123

# Sample schedules (argmax decode) from the trained decoder
uv run rgen_schedule sample-population \
  --ckpt src/output/new_rgen/runs/base_cnn/checkpoints/last.pt \
  --num-samples 42817 \
  --outprefix src/output/new_rgen/runs/base_cnn/base_cnn \
  --decode-mode argmax \

# Evaluate the sampled schedules against the reference
uv run rgen_schedule eval-population \
  --samples src/output/new_rgen/runs/base_cnn/base_cnn.npz \
  --samples-meta src/output/new_rgen/runs/base_cnn/base_cnn_meta.json \
  --reference src/output/new_rgen/full_train_5min.npz \
  --out-json src/output/new_rgen/runs/base_cnn/eval_report.json

# Generate diagnostic plots of the learned unaries
uv run rgen_schedule viz-population \
  --samples src/output/new_rgen/runs/base_cnn/base_cnn.npz \
  --samples-meta src/output/new_rgen/runs/base_cnn/base_cnn_meta.json \
  --reference src/output/new_rgen/full_train_5min.npz \
  --outdir src/output/new_rgen/runs/base_cnn/plots
# End of script
