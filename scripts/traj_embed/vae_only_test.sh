#!/bin/bash

uv run src/ananke_abm/models/traj_syn/pipeline/train_vae_only.py \
--activities_csv src/data/traj_processed/small_activities_homebound_wd.csv \
--purposes_csv src/data/traj_processed/purposes_new.csv \
--outdir src/output/test \
--mode recon+marginals --epochs 5000

uv run src/ananke_abm/models/traj_syn/pipeline/synthesize_vae.py \
--ckpt src/output/test/vae_ep5000.pt \
--out_csv src/output/test/gen_vae.csv \
--num_gen 30000

uv run src/ananke_abm/models/traj_syn/eval/analyze_vae.py \
--gen_csv src/output/test/gen_vae.csv \
--mean_probs_npy src/output/test/vae_mean_probs_ep5000.npy \
--purposes "Home,Work,Shopping,Social,Accompanying,Education,Other" \
--step_minutes 5 \
--t_alloc_minutes 1800 \
--out_json src/output/test/vae_diag.json \
--history_csv src/output/test/vae_history.csv \
--real_csv src/data/traj_processed/small_activities_homebound_wd.csv

uv run ananke visualize-combined-traj \
--traj_csv src/output/test/gen_vae.csv \
--buffer_csv "src/output/test/buffer_grid.csv" \
--out_dir "src/output/test" \
--y_work_max 0.05 \
--y_edu_max 0.05 \
--dpi 300