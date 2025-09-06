#!/bin/bash

uv run ananke visualize-combined-traj \
--traj_csv "src\output\traj_embed\inference\full\gen_activities.csv" \
--buffer_csv "src\output\traj_embed\inference\full\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\gen_full" \
--dpi 300