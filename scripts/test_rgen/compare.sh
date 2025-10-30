#!/bin/bash

uv run rgen_schedule compare-samples\
  --ref_npz src/output/rgen/train_10min.npz \
  --sample_dir src/output/rgen/to_compare \
  --purpose_map src/output/rgen/train_10min_purpose_map.json \
  --outdir src/output/rgen/to_compare/comparison_report
