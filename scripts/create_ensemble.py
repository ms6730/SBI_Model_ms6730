import numpy as np
from pf_ens_functions import create_mannings_ensemble, backup_previous
import pickle
from sbi.utils import get_density_thresholder, RestrictedPrior

base_dir = '/home/at8471/c2_sbi_experiments/sbi_framework'
huc = '02050202'
runname = 'sinnemahoning'
ens_mems = 5
start = "2002-10-27"
end = "2002-12-01"
num_hours = 840
P = 4
Q = 4
baseline_mannings_file_name = 'mannings'
path_to_mannings_val_df = f"{base_dir}/outputs/{runname}_filtered_orig_vals.csv"
ens_num=0

# read the latest proposal
try:
    fp = open(f"{base_dir}/{runname}_posterior{ens_num}.pkl", "rb")
except FileNotFoundError:
    fp = open(f"{base_dir}/{runname}_prior.pkl", "rb")
proposal = pickle.load(fp)

# run sims with these parameters
theta = proposal.sample((num_sims,)).numpy()
# TODO: this function needs to accept theta!!!
#can it accept the proposal instead?
create_mannings_ensemble(base_path = base_dir, runname = runname, mannings_file = baseline_mannings_file_name,orig_vals_path=path_to_mannings_val_df, proposal=proposal,num_sims=ens_mems, P=P, Q=Q, ens_num = ens_num)

# save theta for sbi
filename = f"{base_dir}/{runname}_parameters.npy"
backup_previous(filename)
numpy.save(filename, theta)