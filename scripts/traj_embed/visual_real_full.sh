#!/bin/bash
# --traj_csv "src\data\traj_processed\activities_homebound_wd.csv"
uv run ananke visualize-combined-traj \
--buffer_csv "src\output\traj_embed\inference\real\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\real" \
--dpi 300