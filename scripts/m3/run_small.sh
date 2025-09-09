#!/bin/bash
#SBATCH --job-name=smallMelbAct
#SBATCH --output=smallMelb.out      # Standard output
#SBATCH --error=smallMelb.err       # Standard error
#SBATCH --account=tx89
#SBATCH --time=60:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load miniforge3
mamba activate ananke
uv run ananke traj-embed \
-av "src/data/traj_processed/small_activities_homebound_wd.csv" \
-pv "src/data/traj_processed/purposes_new.csv" \
-o "src/output/traj_embed/models/small" \
-e 200 \
-b 32 \
--lr 1e-3 \
--val_ratio 0.2

