#!/bin/bash
#SBATCH --job-name="smc"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="moritz.meyerzuwestram@unibe.ch"
#SBATCH --cpus-per-task=30
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=6G
module load Python
module load Workspace_Home
python3 ./scheduler.py