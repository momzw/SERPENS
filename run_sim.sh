#!/bin/bash

#SBATCH --job-name="smc"
#SBATCH --mail-type=end,fail
#SBATCH --mail-user="moritz.meyerzuwestram@unibe.ch"
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G

module load Python
module load Workspace_Home

python3 ./run_sim.py