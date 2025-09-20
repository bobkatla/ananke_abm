#!/bin/bash
#SBATCH --job-name=fullMelbAct
#SBATCH --output=fullMelb.out      # Standard output
#SBATCH --error=fullMelb.err       # Standard error
#SBATCH --account=tx89
#SBATCH --time=90:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

module load miniforge3
mamba activate ananke
uv run ananke traj-embed \
-av "src/data/traj_processed/activities_homebound_wd.csv" \
-pv "src/data/traj_processed/purposes_new.csv" \
-o "src/output/traj_embed/models/full" \
-e 3000 \
-b 32 \
--lr 1e-3 \
--val_ratio 0.2 \
--crf_mode "linear"

uv run ananke gval-traj \
--activities_csv "src\data\traj_processed\activities_homebound_wd.csv" \
--purposes_csv "src\data\traj_processed\purposes_new.csv" \
--ckpt "src\output\traj_embed\models\full\ckpt_best.pt" \
--batch_size 32 \
--num_gen 100000 \
--gen_prefix "gen" \
--gen_csv "src\output\traj_embed\inference\full\gen_activities.csv" \
--val_csv "src\output\traj_embed\inference\full\gen_validation.csv" \
--eval_step_minutes 5 \
--crf_mode "linear"

uv run ananke visualize-combined-traj \
--traj_csv "src\output\traj_embed\inference\full\gen_activities.csv" \
--buffer_csv "src\output\traj_embed\inference\full\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\gen_full" \
--y_work_max 0.04 \
--y_edu_max 0.004 \
--dpi 300