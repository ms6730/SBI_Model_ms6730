#!/bin/bash
#SBATCH --job-name=sinne_ens
#SBATCH --output=/home/at8471/c2_sbi_experiments/sbi_framework/%x_%j.out
#SBATCH --error=/home/at8471/c2_sbi_experiments/sbi_framework/%x_%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=16      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=16:00:00
#SBATCH --array=1-100%5

# Calculate the task ID based on the SLURM_ARRAY_TASK_ID
task_id=$((SLURM_ARRAY_TASK_ID - 1))

#set path to json file with run settings
json_path='/home/at8471/c2_sbi_experiments/hydrogen-sbi/scripts/settings.json'

# Run your Python script
python3 run_ensemble.py "$task_id" "$json_path"

