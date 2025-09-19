#!/bin/bash

uv run ananke visualize-combined-traj \
--traj_csv "src\output\traj_embed\inference\small\gen_activities.csv" \
--buffer_csv "src\output\traj_embed\inference\small\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\gen_small" \
--y_work_max 0.02 \
--y_edu_max 0.002 \
--dpi 300