#!/bin/bash

uv run rgen_schedule prepare-crf-data \
    --vae_ckpt src/output/rgen/runs/exp_phase2/checkpoints/last.pt \
    --split_pt src/output/rgen/train_10min_splits.pt \
    --outdir src/output/rgen/crf_data \
    --batch_size 64