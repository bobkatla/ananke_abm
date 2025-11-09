#!/bin/bash

# Sample using CRF decoding from the trained decoder (phase2 model)
uv run rgen_schedule sample-population \
  --ckpt src/output/new_rgen/runs/exp_phase2/checkpoints/last.pt \
  --num-samples 42891 \
  --outprefix src/output/new_rgen/runs/pmd_crf/samples_5min_crf \
  --decode-mode crf \
  --crf-path src/output/new_rgen/crf_data/crf_linear.pt
  
# Evaluate the sampled schedules against the reference
uv run rgen_schedule eval-population \
  --samples src/output/new_rgen/runs/pmd_crf/samples_5min_crf.npz \
  --samples-meta src/output/new_rgen/runs/pmd_crf/samples_5min_crf_meta.json \
  --reference src/output/new_rgen/train_5min.npz \
  --out-json src/output/new_rgen/runs/pmd_crf/eval_crf_report.json

# Generate diagnostic plots of the learned unaries
uv run rgen_schedule viz-population \
  --samples src/output/new_rgen/runs/pmd_crf/samples_5min_crf.npz \
  --samples-meta src/output/new_rgen/runs/pmd_crf/samples_5min_crf_meta.json \
  --reference src/output/new_rgen/train_5min.npz \
  --outdir src/output/new_rgen/runs/pmd_crf/plots_crf

# Sample using CRF decoding from the trained decoder (phase1 model)
# uv run rgen_schedule sample-population \
#   --ckpt src/output/rgen/runs/exp_phase1/checkpoints/last.pt \
#   --num-samples 20000 \
#   --outprefix src/output/rgen/runs/exp_phase1/samples_10min_crf \
#   --decode-mode crf \
#   --crf-path src/output/rgen/crf_data/crf_linear.pt

# Evaluate the sampled schedules against the reference
# uv run rgen_schedule eval-population \
#   --samples src/output/rgen/runs/exp_phase1/samples_10min_crf.npz \
#   --samples-meta src/output/rgen/runs/exp_phase1/samples_10min_crf_meta.json \
#   --reference src/output/rgen/train_10min.npz \
#   --out-json src/output/rgen/runs/exp_phase1/eval_crf_report.json

# Generate diagnostic plots of the learned unaries
# uv run rgen_schedule viz-population \
#   --samples src/output/rgen/runs/exp_phase1/samples_10min_crf.npz \
#   --samples-meta src/output/rgen/runs/exp_phase1/samples_10min_crf_meta.json \
#   --reference src/output/rgen/train_10min.npz \
#   --outdir src/output/rgen/runs/exp_phase1/plots_crf