import torch
from torch.distributions import Uniform
from typing import Dict, List, Optional, Callable
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils import get_density_thresholder, RestrictedPrior
import pandas as pd
import matplotlib.pyplot as plt

base_dir = '/home/at8471/c2_sbi_experiments/sbi_framework'
runname = 'sinnemahoning'
scalar = 10
ens_mems = 10
next_ens = 1
num_samples = 1000
quantile = 1e-4
ens_value_file_name = f'sinnemahoning_mannings_ens{next_ens-1}.csv'

#read in sample matrix
sample_df = pd.read_csv(f'{base_dir}/{ens_value_file_name}')

#define the original prior 
orig_mannings = torch.tensor(sample_df.iloc[0].to_numpy())
mins = orig_mannings/scalar
maxs = orig_mannings*scalar
prior = Uniform(mins, maxs)

#convert the sample matrix from a df to a tensor
sample_df = sample_df.drop(0) #dropping baseline value row
theta_samples = torch.tensor(sample_df.values, dtype=torch.float)

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

#initialize SNPE
inference = SNPE(prior=prior)

#train the neural posterior density estimator
inference.append_simulations(theta_samples, sim_tensor_arr)
density_estimator = inference.train(force_first_round_loss=True)

#create the posterior
posterior = inference.build_posterior(density_estimator).set_default_x(reshaped_obsv)

#restrict and update the prior
accept_reject_fn = get_density_thresholder(posterior, quantile=quantile, num_samples_to_estimate_support=num_samples)

# update prior for next round
proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
#save proposal with number (so we can see how evolves) 
with open(f'{base_dir}/outputs/prior{next_ens}.pkl', 'wb') as f:
    pickle.dump(proposal, f)
    
# add convergence criteria
#draw new samples from the updated prior for the next round of simulation

#new samples goes into create mannings ens
new_sample = proposal.sample((ens_mems,))
new_sample = new_sample.numpy()

#create and write a new dataframe from the proposal samples from which to create the next ensemble
new_ens_df = pd.DataFrame(new_sample, columns=sample_df.columns)
new_ens_df.to_csv(f"{base_dir}/outputs/{runname}_mannings_ens{next_ens}.csv", index=False)



