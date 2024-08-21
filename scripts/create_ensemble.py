from pf_ens_functions import create_mannings_ensemble, backup_previous
import pickle
from sbi.utils import get_density_thresholder, RestrictedPrior
import shutil

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
    fp = open(f"{base_dir}/{runname}_posterior.pkl", "rb")
except FileNotFoundError:
    fp = open(f"{base_dir}/{runname}_prior.pkl", "rb")
proposal = pickle.load(fp)

#don't really need a create_mannings_ensemble function, can just run the code here
create_mannings_ensemble(proposal=proposal,base_path = base_dir, runname = runname, mannings_file = baseline_mannings_file_name,orig_vals_path=path_to_mannings_val_df,num_sims=ens_mems, P=P, Q=Q, ens_num = ens_num)

#save current inference and posterior
try:
    src_file = f"{base_dir}/{runname}_posterior.pkl"
    dest_file=f"{base_dir}/{runname}_posterior{ens_num-1}.pkl"
    shutil.copyfile(src_file, dest_file)
    
    src_file = f"{base_dir}/{runname}_inference.pkl"
    dest_file=f"{base_dir}/{runname}_inference{ens_num-1}.pkl"
    shutil.copyfile(src_file, dest_file)
except:
    pass