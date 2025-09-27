#!/bin/bash
# 
uv run ananke visualize-combined-traj \
--traj_csv "src\data\traj_processed\activities_homebound_wd.csv" \
--buffer_csv "src\output\traj_embed\inference\real\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\real" \
--y_work_max 0.04 \
--y_edu_max 0.004 \
--dpi 300