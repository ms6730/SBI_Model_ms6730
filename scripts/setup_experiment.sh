#!/bin/bash
#SBATCH --job-name=mann_bl
#SBATCH --output=/home/at8471/c2_sbi_experiments/sbi_framework/%x_%j.out
#SBATCH --error=/home/at8471/c2_sbi_experiments/sbi_framework/%x_%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=16      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:20:00

#module load parflow-shared
#all these variables should be in a runfile that describes the run
#add function that takes json (etc) in pf_ens_functions.py, can work for multiple scripts
base_dir='/home/at8471/c2_sbi_experiments/sbi_framework'
runname='sinnemahoning'
huc='02050202'
hours=840
start="2002-10-27"
end="2002-12-01"
timezone="EDT"
P=4
Q=4
ens_mems=5
scalar=2

# Set up and do baseline run
python3 setup_experiment.py "$base_dir" "$runname" "$huc" "$hours" "$start" "$end" "$timezone" "$P" "$Q" "$ens_mems" "$scalar"

