#!/bin/bash
#SBATCH --job-name="smc"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="moritz.meyerzuwestram@unibe.ch"
#SBATCH --cpus-per-task=20
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH --tmp=20G

module load Python
module load Workspace_Home
python3 ./scheduler.py