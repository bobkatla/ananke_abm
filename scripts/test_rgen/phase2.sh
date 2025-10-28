#!/bin/bash

# Fit
uv run rgen_schedule fit \
  --config src/ananke_abm/models/gen_schedule/dataio/configs/phase2.yaml \
  --output-dir src/output/rgen/runs/exp_phase2 \
  --seed 123

# Sample schedules (argmax decode) from the trained decoder
uv run rgen_schedule sample-population \
  --ckpt src/output/rgen/runs/exp_phase2/checkpoints/last.pt \
  --num-samples 20000 \
  --outprefix src/output/rgen/runs/exp_phase2/samples_10min

# Evaluate the sampled schedules against the reference
uv run rgen_schedule eval-population \
  --samples src/output/rgen/runs/exp_phase2/samples_10min.npz \
  --samples-meta src/output/rgen/runs/exp_phase2/samples_10min_meta.json \
  --reference src/output/rgen/train_10min.npz \
  --out-json src/output/rgen/runs/exp_phase2/eval_report.json

# Generate diagnostic plots of the learned unaries
uv run rgen_schedule viz-population \
  --samples src/output/rgen/runs/exp_phase2/samples_10min.npz \
  --samples-meta src/output/rgen/runs/exp_phase2/samples_10min_meta.json \
  --reference src/output/rgen/train_10min.npz \
  --outdir src/output/rgen/runs/exp_phase2/plots
# End of script