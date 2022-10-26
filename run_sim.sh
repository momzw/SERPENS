#!/bin/bash

#SBATCH --job-name="smc"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="moritz.meyerzuwestram@unibe.ch"
#SBATCH --cpus-per-task=20
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=3G

module load Python
module load Workspace_Home

python3 ./run_sim.py