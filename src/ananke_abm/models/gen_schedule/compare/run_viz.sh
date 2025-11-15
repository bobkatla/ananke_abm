#!/bin/bash

uv run rgen_schedule plot-overview \
    --ref_npz src/output/to_compare/groundtruth/full_real.npz \
    --ref_meta src/output/to_compare/groundtruth/full_real.json \
    --compare_dir src/output/to_compare/model_samples \
    --outdir src/output/to_compare/viz_output