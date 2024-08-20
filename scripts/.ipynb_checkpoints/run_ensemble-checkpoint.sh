#!/bin/bash
#SBATCH --job-name=sinne_ens
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=16      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00
#SBATCH --array=1-5%3

module load parflow-shared

# Calculate the task ID based on the SLURM_ARRAY_TASK_ID
task_id=$((SLURM_ARRAY_TASK_ID - 1))

runname='sinnemahoning'
hours=840
start_date="2002-10-27"

# Generate a unique output file name based on the task ID
out_dir="/home/at8471/c2_sbi_experiments/sbi_framework/outputs/${runname}_${task_id}"

# Run your Python script with the selected input file and output file

python3 run_ensemble.py "$out_dir" "$runname" "$task_id" "$hours" "$start_date"

