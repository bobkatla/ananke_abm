#!/bin/bash

uv run ananke gval-traj \
--activities_csv "src\data\traj_processed\small_activities_homebound_wd.csv" \
--purposes_csv "src\data\traj_processed\purposes_new.csv" \
--ckpt "src\output\traj_embed\models\small\ckpt_best.pt" \
--batch_size 32 \
--num_gen 10000 \
--gen_prefix "gen" \
--gen_csv "src\output\traj_embed\inference\small\gen_activities.csv" \
--val_csv "src\output\traj_embed\inference\small\gen_validation.csv" \
--eval_step_minutes 5