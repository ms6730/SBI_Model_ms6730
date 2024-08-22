import numpy as np
import pandas as pd
import subsettools
from sbi.inference import SNPE

model_eval_path = os.path.abspath('/home/at8471/c2_sbi_experiments/model_evaluation')
sys.path.append(model_eval_path)
from model_evaluation import get_observations, get_parflow_output, calculate_metrics, explore_available_observations, get_parflow_output_nc
from plots import plot_obs_locations, plot_time_series, plot_compare_scatter, plot_metric_map

# Define inputs to workflow
base_dir = "/home/at8471/c2_sbi_experiments/sbi_framework"
grid = "conus2"
huc_list = ["02050202"]
ij_bounds, mask = subsettools.define_huc_domain(huc_list, grid)

start_date = "2002-10-27"
end_date = "2002-12-01"
temporal_resolution = "daily"

runname="sinnemahoning"

variable_list = ["streamflow"]
ens_mems = 5

# Evaluate
for mem in range(0,ens_mems):
    parflow_output_dir=f"{base_dir}/outputs/{runname}_{mem}"
    
    for variable in variable_list:
    
        # Get observation data for sites in domain
        obs_metadata_df, obs_data_df = get_observations(mask, ij_bounds, grid, start_date, end_date,
                                                        variable, temporal_resolution)
        print("created obsv df")
        
        obs_metadata_df.to_csv(f"{parflow_output_dir}/{variable}_{temporal_resolution}_metadf.csv", index=False)
        obs_data_df.to_csv(f"{parflow_output_dir}/{variable}_{temporal_resolution}_df.csv", index=False)
    
        # Get ParFlow outputs matching site locations
        parflow_data_df = get_parflow_output_nc(f"{parflow_output_dir}/{runname}_{mem}.nc", f'{parflow_output_dir}/{variable}_{temporal_resolution}_metadf.csv',var_name = variable, write_path = f"{parflow_output_dir}/{variable}_{temporal_resolution}_pfsim.csv")
    
        print("created pf df")
        
        # Calculate metrics comparing ParFlow vs. observations
        obs_data_df = obs_data_df.dropna(axis=1)
        common_columns = obs_data_df.columns.intersection(parflow_data_df.columns)
        obs_data_df = obs_data_df[common_columns]
        parflow_data_df = parflow_data_df[common_columns]
        obs_metadata_df= obs_metadata_df[obs_metadata_df['site_id'].isin(common_columns)]
        
        metrics_df = calculate_metrics(obs_data_df, parflow_data_df, obs_metadata_df,
                                       write_csv=True, csv_path=f"{parflow_output_dir}/{variable}_metrics.csv")
        print("calculated metrics")


##### SBI #####

# try loading existing inference structure
# if not there, create new one from prior
try:
    fp = open(f"{base_dir}/{runname}_inference.pkl", "rb")
except FileNotFoundError:
    with open(f"{base_dir}/{runname}_prior.pkl", "rb") as fp:
        prior = pickle.load(fp)
    inference = SNPE(prior=prior)
else:
    with fp:
        inference = pickle.load(fp)

# get parameters for last ensemble run
theta = np.load(f"{base_dir}/{runname}_parameters.npy")

#create 1D torch tensors for observed and simulated outputs
for i in range(0, ens_mems):
    sim_df = pd.read_csv(f'{base_dir}/outputs/{runname}_{i}/streamflow_daily_pfsim.csv').drop('date', axis=1)
    if i == 0:
        obsv_df = pd.read_csv(f'{base_dir}/outputs/{runname}_{i}/streamflow_daily_df.csv').drop('date', axis=1)
        common_columns = sim_df.columns.intersection(obsv_df.columns)
        obsv_df = obsv_df[common_columns]
        obsv_tensor = torch.tensor(obsv_df.values, dtype=torch.float)
        obsv_flat = torch.flatten(obsv_tensor)
        reshaped_obsv = torch.reshape(obsv_flat, (1, obsv_flat.numel()))

    sim_df = sim_df[common_columns]
    sim_tensor = torch.tensor(sim_df.values, dtype=torch.float)
    sim_flat = torch.flatten(sim_tensor)
    reshaped_sim = torch.reshape(sim_flat, (1, sim_flat.numel()))

    if i == 0: 
        sim_tensor_arr = reshaped_sim
    else:
        sim_tensor_arr = torch.cat((sim_tensor_arr, reshaped_sim), dim=0)

# update posterior with new simulations
_ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
posterior = inference.build_posterior().set_default_x(obs)

#this section wasn't originally here, i assume we still want to do restricted proposal?
accept_reject_fn = get_density_thresholder(posterior, quantile=quantile, num_samples_to_estimate_support=num_samples)

# update prior for next round
proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")

# save results, backup existing ones
filename = f"{base_dir}/{runname}_inference.pkl"
with open(filename, "wb") as fp:
    pickle.dump(inference, fp)

filename = f"{base_dir}/{runname}_posterior.pkl"
with open(filename, "wb") as fp:
    pickle.dump(posterior, fp)

filename = f"{base_dir}/{runname}_proposal.pkl"
with open(filename, "wb") as fp:
    pickle.dump(proposal, fp)





