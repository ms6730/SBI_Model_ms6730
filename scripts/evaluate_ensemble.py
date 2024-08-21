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
ens_mems = 10

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

# TODO: need to get parameters results x for these sims, move over concatenated data vector stuff

# update posterior with new simulations
_ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
posterior = inference.build_posterior().set_default_x(obs)

# save results, backup existing ones
filename = f"{base_dir}/{runname}_inference.pkl"
with open(filename, "wb") as fp:
    pickle.dump(inference, fp)

filename = f"{base_dir}/{runname}_posterior.pkl"
with open(filename, "wb") as fp:
    pickle.dump(posterior, fp)