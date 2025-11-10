#!/bin/bash

uv run rgen_schedule metric-tables \
    --ref_npz src/output/to_compare/groundtruth/full_real.npz \
    --ref_meta src/output/to_compare/groundtruth/full_real.json \
    --compare_dir src/output/to_compare/model_samples \
    --metrics all \
    --outdir src/output/to_compare/metrics_output