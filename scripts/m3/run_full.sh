#!/bin/bash
#SBATCH --job-name=fullMelbAct
#SBATCH --output=fullMelb.out      # Standard output
#SBATCH --error=fullMelb.err       # Standard error
#SBATCH --account=tx89
#SBATCH --time=90:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load miniforge3
mamba activate ananke
uv run ananke traj-embed \
-av "..\..\src\data\traj_processed\activities_homebound_wd.csv" \
-pv "..\..\src\data\traj_processed\purposes_new.csv" \
-o "..\..\src\output\traj_embed\models\full" \
-e 1000 \
-b 32 \
--lr 1e-3 \
--val_ratio 0.2
