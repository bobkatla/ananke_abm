#!/bin/bash
#SBATCH --job-name=fullMelbAct_linear
#SBATCH --output=fullMelb_linear.out      # Standard output
#SBATCH --error=fullMelb_linear.err       # Standard error
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
-o "src/output/traj_embed/models/full_linear" \
-e 3000 \
-b 32 \
--lr 1e-3 \
--val_ratio 0.2 \
--crf_mode "linear"

uv run ananke gval-traj \
--activities_csv "src\data\traj_processed\activities_homebound_wd.csv" \
--purposes_csv "src\data\traj_processed\purposes_new.csv" \
--ckpt "src\output\traj_embed\models\full_linear\ckpt_best.pt" \
--batch_size 32 \
--num_gen 500000 \
--gen_prefix "gen" \
--gen_csv "src\output\traj_embed\inference\full_linear\gen_activities.csv" \
--val_csv "src\output\traj_embed\inference\full_linear\gen_validation.csv" \
--eval_step_minutes 5 \
--crf_mode "linear" \
--summary_json "src\output\traj_embed\inference\full_linear\summary.json"

uv run ananke visualize-combined-traj \
--traj_csv "src\output\traj_embed\inference\full_linear\gen_activities.csv" \
--buffer_csv "src\output\traj_embed\inference\full_linear\buffer_grid.csv" \
--out_dir "src\output\traj_embed\img\gen_full_linear" \
--y_work_max 0.05 \
--y_edu_max 0.05 \
--dpi 300