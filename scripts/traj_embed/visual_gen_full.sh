#!/bin/bash

uv run ananke visualize-combined-traj \
--traj_csv "src\output\traj_embed\inference\full_linear\gen_activities.csv" \
--buffer_csv "src\output\traj_embed\inference\full_linear\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\gen_full_linear" \
--y_work_max 0.004 \
--y_edu_max 0.0004 \
--dpi 300