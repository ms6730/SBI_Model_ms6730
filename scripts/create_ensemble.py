from pf_ens_functions import create_mannings_ensemble
import pickle
from sbi.utils import get_density_thresholder, RestrictedPrior

base_path = '/home/at8471/c2_sbi_experiments/sbi_framework'
huc = '02050202'
runname = 'sinnemahoning'
num_mems = 5
start = "2002-10-27"
end = "2002-12-01"
num_hours = 840
P = 4
Q = 4
orig_mannings_file_name = 'mannings'
ens_num=0

# read the latest proposal
try:
    fp = open("posterior.pkl", "rb")
except FileNotFoundError:
    fp = open("prior.pkl", "rb")
proposal = inference = pickle.load(fp)

# run sims with these parameters
theta = proposal.sample((num_sims,)).numpy()
# TODO: this function needs to accept theta!!!
create_mannings_ensemble(theta, base_path = base_path, baseline_runname = runname, mannings_file = orig_mannings_file_name, num_ens_mem=num_mems, P=P, Q=Q, ens_num = ens_num)

# TODO: save theta for sbi
new_ens_df = pd.DataFrame(theta, columns=sample_df.columns)
new_ens_df.to_csv(f"{base_dir}/{runname}_mannings_ens{next_ens}.csv", index=False)

