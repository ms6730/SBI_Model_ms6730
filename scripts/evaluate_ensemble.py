import numpy as np
import pandas as pd
import subsettools
from sbi.inference import SNPE
from sbi.utils import get_density_thresholder, RestrictedPrior

model_eval_path = os.path.abspath('/home/at8471/c2_sbi_experiments/model_evaluation')
sys.path.append(model_eval_path)
from model_evaluation import get_observations, get_parflow_output, calculate_metrics, explore_available_observations, get_parflow_output_nc
from plots import plot_obs_locations, plot_time_series, plot_compare_scatter, plot_metric_map

# Define inputs to workflow
base_dir = "/home/at8471/c2_sbi_experiments/sbi_framework"
grid = "conus2"
huc_list = ["02050202"]
start_date = "2002-10-27"
end_date = "2002-12-01"
temporal_resolution = "daily"
runname="sinnemahoning"
variable_list = ["streamflow"]
num_sims = 5
ens_num=0
metadata_path=f'{base_dir}/outputs/{runname}/streamflow_daily_metadf.csv'
obsv_path=f'{base_dir}/outputs/{runname}/streamflow_daily_df.csv'

ij_bounds, mask = subsettools.define_huc_domain(huc_list, grid)

# Evaluate
for sim in range(0,num_sims):
    
    parflow_output_dir=f"{base_dir}/outputs/{runname}_{ens_num}_{sim}"
    nc_path = f"{parflow_output_dir}/{runname}_{sim}.nc"
    write_path=f"{parflow_output_dir}/{variable}_{temporal_resolution}_pfsim.csv"
    
    for variable in variable_list:  
        # Get ParFlow outputs matching site locations
        parflow_data_df = get_parflow_output_nc(nc_path, metadata_path, variable, write_path)
        print("created pf df")

##### SBI #####

# try loading existing inference structure
# if not there, create new one from prior
try:
    fp = open(f"{base_dir}/{runname}_proposal.pkl", "rb")
except FileNotFoundError:
    with open(f"{base_dir}/{runname}_prior.pkl", "rb") as fp:
        prior = pickle.load(fp)
    inference = SNPE(prior=prior)
else:
    with fp:
        prior = pickle.load(fp)
    fi = open(f"{base_dir}/{runname}_inference.pkl", "rb")
    inference=pickle.load(fi)
    
        
# get parameters for last ensemble run
theta_sim = np.load(f"{base_dir}/{runname}_parameters_ens{ens_num}.npy")

# create 1D torch tensors for observed and simulated outputs
sim_data = []

for i in range(ens_mems):
    sim_df = pd.read_csv(f'{base_dir}/outputs/{runname}_{ens_num}_{i}/streamflow_daily_pfsim.csv').drop('date', axis=1)
    if i == 0:
        obsv_df = pd.read_csv(obsv_path).drop('date', axis=1)
        common_columns = sim_df.columns.intersection(obsv_df.columns)
        obsv_df = obsv_df[common_columns]
        obsv_tensor = torch.tensor(obsv_df.values, dtype=torch.float)
        obsv_flat = torch.flatten(obsv_tensor)
        x_obsv = torch.reshape(obsv_flat, (1, obsv_flat.numel()))

    sim_df = sim_df[common_columns]
    sim_tensor = torch.tensor(sim_df.values, dtype=torch.float)
    sim_flat = torch.flatten(sim_tensor)
    sim_data.append(sim_flat)

x_sim = torch.stack(sim_data, dim=0)

# update posterior with new simulations
_ = inference.append_simulations(theta_sim, x_sim).train(force_first_round_loss=True)
posterior = inference.build_posterior().set_default_x(x_obs)

# update proposal for next round
proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
accept_reject_fn = get_density_thresholder(posterior, quantile=quantile, num_samples_to_estimate_support=num_samples)

# save updated results
filename = f"{base_dir}/{runname}_inference.pkl"
with open(filename, "wb") as fp:
    pickle.dump(inference, fp)

filename = f"{base_dir}/{runname}_posterior.pkl"
with open(filename, "wb") as fp:
    pickle.dump(posterior, fp)

filename = f"{base_dir}/{runname}_proposal.pkl"
with open(filename, "wb") as fp:
    pickle.dump(proposal, fp)





