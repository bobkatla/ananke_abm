#!/bin/bash

uv run ananke traj-embed \
-av "src\data\traj_processed\activities_homebound_wd.csv" \
-pv "src\data\traj_processed\purposes_new.csv" \
-o "src\output\traj_embed\models\full_linear" \
-e 1000 \
-b 32 \
--lr 1e-3 \
--val_ratio 0.1 \
--crf_mode "linear"
