#!/bin/bash

uv run ananke traj-embed \
-av "src\data\traj_processed\small_activities_homebound_wd.csv" \
-pv "src\data\traj_processed\purposes_new.csv" \
-o "src\output\traj_embed\models\small_semi" \
-e 200 \
-b 32 \
--lr 1e-3 \
--val_ratio 0.2 \
--crf_mode "semi"