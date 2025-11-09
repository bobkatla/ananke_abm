#!/bin/bash

# 2) Fit Phase 1
uv run rgen_schedule fit \
  --config src/ananke_abm/models/gen_schedule/dataio/configs/phase1.yaml \
  --output-dir src/output/new_rgen/runs/exp_phase1 \
  --seed 123

# 3) Sample schedules (argmax decode) from the trained decoder
uv run rgen_schedule sample-population \
  --ckpt src/output/new_rgen/runs/exp_phase1/checkpoints/best_val.pt \
  --num-samples 42891 \
  --outprefix src/output/new_rgen/runs/exp_phase1/samples_5min \
  --decode-mode argmax \

# 4) Evaluate the sampled schedules against the reference
uv run rgen_schedule eval-population \
  --samples src/output/new_rgen/runs/exp_phase1/samples_5min.npz \
  --samples-meta src/output/new_rgen/runs/exp_phase1/samples_5min_meta.json \
  --reference src/output/new_rgen/train_5min.npz \
  --out-json src/output/new_rgen/runs/exp_phase1/eval_report.json

# 5) Generate diagnostic plots of the learned unaries
uv run rgen_schedule viz-population \
  --samples src/output/new_rgen/runs/exp_phase1/samples_5min.npz \
  --samples-meta src/output/new_rgen/runs/exp_phase1/samples_5min_meta.json \
  --reference src/output/new_rgen/train_5min.npz \
  --outdir src/output/new_rgen/runs/exp_phase1/plots
# End of script
