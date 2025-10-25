#!/bin/bash

# 1) Prepare 10-min grid from your small dev CSV
uv run rgen_schedule prepare \
  --activities "src/data/traj_processed/small_activities_homebound_wd.csv" \
  --grid 10 \
  --out src/output/rgen_phase1/train_10min.npz

# 2) Fit Phase 1
uv run rgen_schedule fit \
  --config src/ananke_abm/models/gen_schedule/dataio/configs/phase1.yaml \
  --output-dir src/output/rgen_phase1/runs \
  --run exp_phase1 --seed 123

# 3) Sample schedules (argmax decode) from the trained decoder
uv run rgen_schedule sample-population \
  --ckpt src/output/rgen_phase1/runs/exp_phase1/checkpoints/last.pt \
  --num-samples 20000 \
  --outprefix src/output/rgen_phase1/runs/exp_phase1/samples_10min

# 4) Evaluate the sampled schedules against the reference
uv run rgen_schedule eval-population \
  --samples src/output/rgen_phase1/runs/exp_phase1/samples_10min.npz \
  --samples-meta src/output/rgen_phase1/runs/exp_phase1/samples_10min_meta.json \
  --reference src/output/rgen_phase1/train_10min.npz \
  --out-json src/output/rgen_phase1/runs/exp_phase1/eval_report.json

# 5) Generate diagnostic plots of the learned unaries
uv run rgen_schedule viz-population \
  --samples src/output/rgen_phase1/runs/exp_phase1/samples_10min.npz \
  --samples-meta src/output/rgen_phase1/runs/exp_phase1/samples_10min_meta.json \
  --reference src/output/rgen_phase1/train_10min.npz \
  --outdir src/output/rgen_phase1/runs/exp_phase1/plots
# End of script
