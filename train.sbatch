#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options
#SBATCH --job-name=DABERT # Job name
#SBATCH --output=./outputs/modelresults-%j.out
# Standard output and error log
#SBATCH --mem=50gb
# Job memory request 
#SBATCH --partition=standard
# Partition (queue) 
#SBATCH --time=30:00:00
# Time limit hrs:min:sec 
#SBATCH --partition=gpu-standard 

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 
echo "$USER"
# Running your code here 
python -Wignore -u main.py
# End of job info 
echo "Ending: "`date +"%D %T"`

