#!/bin/bash

# Sample using CRF decoding from the trained decoder (phase1 model)
uv run rgen_schedule sample-population \
  --ckpt src/output/new_rgen/runs/base_cnn/checkpoints/last.pt \
  --num-samples 42817 \
  --outprefix src/output/new_rgen/runs/cnn_crf_nonhome/cnn_crf_nonhome \
  --decode-mode crf \
  --crf-path src/output/new_rgen/crf_data_base/crf_linear.pt \
  --enforce-nonhome

# Evaluate the sampled schedules against the reference
uv run rgen_schedule eval-population \
  --samples src/output/new_rgen/runs/cnn_crf_nonhome/cnn_crf_nonhome.npz \
  --samples-meta src/output/new_rgen/runs/cnn_crf_nonhome/cnn_crf_nonhome_meta.json \
  --reference src/output/new_rgen/full_train_5min.npz \
  --out-json src/output/new_rgen/runs/cnn_crf_nonhome/eval_crf_report.json

# Generate diagnostic plots of the learned unaries
uv run rgen_schedule viz-population \
  --samples src/output/new_rgen/runs/cnn_crf_nonhome/cnn_crf_nonhome.npz \
  --samples-meta src/output/new_rgen/runs/cnn_crf_nonhome/cnn_crf_nonhome_meta.json \
  --reference src/output/new_rgen/full_train_5min.npz \
  --outdir src/output/new_rgen/runs/cnn_crf_nonhome/plots_crf