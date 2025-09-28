#!/bin/bash

uv run ananke traj-embed \
-av "src\data\traj_processed\small_activities_homebound_wd.csv" \
-pv "src\data\traj_processed\purposes_new.csv" \
-o "src\output\traj_embed\models\small_linear" \
-e 1000 \
-b 32 \
--lr 1e-3 \
--val_ratio 0.2 \
--crf_mode "linear"

uv run ananke gval-traj \
--activities_csv "src\data\traj_processed\small_activities_homebound_wd.csv" \
--purposes_csv "src\data\traj_processed\purposes_new.csv" \
--ckpt "src\output\traj_embed\models\small_linear\ckpt_best.pt" \
--batch_size 32 \
--num_gen 30000 \
--gen_prefix "gen" \
--gen_csv "src\output\traj_embed\inference\small_linear\gen_activities.csv" \
--val_csv "src\output\traj_embed\inference\small_linear\gen_validation.csv" \
--eval_step_minutes 5 \
--crf_mode "linear" \
--summary_json "src\output\traj_embed\inference\small_linear\summary.json"

uv run ananke visualize-combined-traj \
--traj_csv "src\output\traj_embed\inference\small_linear\gen_activities.csv" \
--buffer_csv "src\output\traj_embed\inference\small_linear\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\gen_small_linear" \
--y_work_max 0.02 \
--y_edu_max 0.002 \
--dpi 300
