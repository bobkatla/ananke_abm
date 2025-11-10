#!/bin/bash

uv run rgen_schedule prepare-crf-data \
    --vae_ckpt src/output/new_rgen/runs/exp_phase2/checkpoints/last.pt \
    --split_pt src/output/new_rgen/train_5min_splits.pt \
    --outdir src/output/new_rgen/crf_data \
    --batch_size 64

uv run rgen_schedule train-crf \
    --cfg src/ananke_abm/models/gen_schedule/dataio/configs/crf_config.yaml