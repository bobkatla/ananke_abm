#!/bin/bash
# --traj_csv "src\data\traj_processed\small_activities_homebound_wd.csv" \
uv run ananke visualize-combined-traj \
--buffer_csv "src\output\traj_embed\inference\real\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\small_real" \
--dpi 300 \
--y_work_max 0.00002 \
--y_edu_max 0.00002