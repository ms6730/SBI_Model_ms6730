#!/bin/bash
#SBATCH --job-name=mann_bl
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=16      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00

module load parflow-shared

base_dir='/home/at8471/c2_sbi_experiments/sbi_framework'
runname='sinnemahoning'
huc='02050202'
hours=840
start="2002-10-27"
end="2002-12-01"
timezone="EDT"
P=4
Q=4

# Set up and do baseline run
python3 setup_run_eval_baseline.py "$base_dir" "$runname" "$huc" "$hours" "$start" "$end" "$timezone" "$P" "$Q"

