import sys
import os
os.environ['PARFLOW_DIR'] = '/home/SHARED/software/parflow/3.10.0'
#path to gpu build on della 
#os.environ['PARFLOW_DIR'] = '/home/ga6/parflow_mgsemi_new/parflow'
from parflow import Run
from parflow.tools.settings import set_working_directory
from pathlib import Path
from glob import glob
from parflow.tools.io import read_pfb, write_pfb, read_clm, read_pfb_sequence
from pf_ens_functions import setup_baseline_run, calculate_water_table_depth, calculate_flow, get_parflow_output_nc
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import parflow as pf
model_eval_path = os.path.abspath('/home/at8471/c2_sbi_experiments/model_evaluation')
sys.path.append(model_eval_path)
from model_evaluation import get_observations
import subsettools as st
import hf_hydrodata as hf
import torch
from torch.distributions import Uniform
import pickle

#read in variables from the job script
base_dir = sys.argv[1]
runname = sys.argv[2]
huc = sys.argv[3]
num_hours = float(sys.argv[4])
start = sys.argv[5]
end = sys.argv[6]
timezone = sys.argv[7]
P = int(sys.argv[8])
Q = int(sys.argv[9])
ens_mems = int(sys.argv[10])
scalar = int(sys.argv[11])

#set up the baseline run for the target HUC for this experiment
grid = "conus2"
temporal_resolution = "daily"
variable_list = ["streamflow"]

setup_baseline_run(base_dir = base_dir, runname = runname, hucs = [huc], start=start, end = end, P=P, Q=Q, hours = num_hours)

#run the baseline
out_dir = f"{base_dir}/outputs/{runname}"
set_working_directory(out_dir)

run = Run.from_definition(f'{out_dir}/{runname}.yaml')
run.TimingInfo.StopTime = num_hours
run.run(working_directory=f'{out_dir}')

# Create a daily mean .nc output file and delete hourly pfbs
data = run.data_accessor
slope_x_file = f'{out_dir}/slope_x.pfb'
slope_x = pf.read_pfb(slope_x_file)

slope_y_file = f'{out_dir}/slope_y.pfb'
slope_y = pf.read_pfb(slope_y_file)
mannings = pf.read_pfb(f'{out_dir}/mannings.pfb')

dz = data.dz
dx = 1000.0
dy = 1000.0
et_idx = 4
swe_idx = 10
start=start

pressure_files = sorted(glob(f'{out_dir}/{runname}.out.press.*.pfb')[1:])
saturation_files = sorted(glob(f'{out_dir}/{runname}.out.satur.*.pfb')[1:])
clm_files = sorted(glob(f'{out_dir}/{runname}.out.clm_output.*.pfb'))

timesteps = pd.date_range(start, periods=len(pressure_files), freq='1H')
ds = xr.Dataset()
ds['pressure'] = xr.DataArray(
    read_pfb_sequence(pressure_files),
    coords={'time': timesteps}, 
    dims=('time', 'z', 'y', 'x')
)
mask = ds['pressure'].isel(time=0).values > -9999
ds['saturation'] = xr.DataArray(
    read_pfb_sequence(saturation_files),
    coords={'time': timesteps}, 
    dims=('time', 'z', 'y', 'x')
)
clm = xr.DataArray(
    read_pfb_sequence(clm_files),
    coords={'time': timesteps}, 
    dims=('time', 'feature', 'y', 'x')
)
ds['wtd'] = calculate_water_table_depth(ds, dz)
ds['streamflow'] = calculate_flow(
    ds, slope_x, slope_y, mannings, dx, dy, mask
)
ds['swe'] = clm.isel(feature=swe_idx)
ds['et'] = clm.isel(feature=et_idx)

ds = ds.resample(time='1D').mean()

ds['mannings'] = xr.DataArray(
    read_pfb(f'{out_dir}/mannings.pfb')[0,:,:],
    dims=('y','x')
)
ds['porosity'] = xr.DataArray(
    read_pfb(f'{out_dir}/{runname}.out.porosity.pfb'),
    dims=('z','y','x')
)
ds['permeability'] = xr.DataArray(
    read_pfb(f'{out_dir}/{runname}.out.perm_x.pfb'),
    dims=('z','y','x')
)
ds['van_genuchten_alpha'] = xr.DataArray(
    read_pfb(f'{out_dir}/{runname}.out.alpha.pfb'),
    dims=('z','y','x')
)
ds['van_genuchten_n'] = xr.DataArray(
    read_pfb(f'{out_dir}/{runname}.out.n.pfb'),
    dims=('z','y','x')
)
lat = pf.tools.io.read_clm(f'{out_dir}/drv_vegm.dat', type='vegm')[:, :, 0]
lon = pf.tools.io.read_clm(f'{out_dir}/drv_vegm.dat', type='vegm')[:, :, 1]
ds = ds.assign_coords({
    'lat': xr.DataArray(lat, dims=['y', 'x']),
    'lon': xr.DataArray(lon, dims=['y', 'x']),
})

ds = ds.astype(np.float32)

ds.to_netcdf(f'{out_dir}/{runname}.nc', mode='w')

# Clean up
del ds
del clm
_ = [os.remove(os.path.abspath(f)) for f in pressure_files]
_ = [os.remove(os.path.abspath(f)+'.dist') for f in pressure_files]
_ = [os.remove(os.path.abspath(f)) for f in saturation_files]
_ = [os.remove(os.path.abspath(f)+'.dist') for f in saturation_files]
_ = [os.remove(os.path.abspath(f)) for f in clm_files]
_ = [os.remove(os.path.abspath(f)+'.dist') for f in clm_files]

#Write out csvs of observations from hydrodata of target variables within the subset domain 
ij_bounds, mask = st.define_huc_domain([huc], grid)

for variable in variable_list:
    # Get observation data for sites in domain
    obs_metadata_df, obs_data_df = get_observations(mask, ij_bounds, grid, start, end,
                                                    variable, temporal_resolution)
    print("created obsv df")
    
    obs_metadata_df.to_csv(f"{out_dir}/{variable}_{temporal_resolution}_metadf.csv", index=False)
    obs_data_df.to_csv(f"{out_dir}/{variable}_{temporal_resolution}_df.csv", index=False)

    # Get ParFlow outputs matching site locations
    parflow_data_df = get_parflow_output_nc(f"{out_dir}/{runname}.nc", f'{out_dir}/{variable}_{temporal_resolution}_metadf.csv',var_name = variable, write_path = f"{out_dir}/{variable}_{temporal_resolution}_pfsim.csv")

    print("created pf df")

#filter the conus2 mannings values by the subset mannings
subset_mannings = read_pfb(f"{out_dir}/mannings.pfb")
filters = {"dataset":"conus2_domain", "variable":"mannings"}
mannings_map = hf.get_gridded_data(filters)
all_vals = np.unique(mannings_map)
subset_vals = np.unique(subset_mannings)
mannings_dict = {}
    
for i in range(len(all_vals)):
    mannings_dict[f"m{i}"]=[all_vals[i]]
    
filtered_dict = {k: v for k, v in mannings_dict.items() if v in subset_vals}
filtered_df = pd.DataFrame(filtered_dict)

#create prior for subset mannings map
orig_mannings = torch.tensor(filtered_df.iloc[0].to_numpy())
mins = orig_mannings/scalar
maxs = orig_mannings*scalar
prior = Uniform(mins, maxs)

#save the prior
with open(f'{base_dir}/outputs/{runname}_prior.pkl', 'wb') as f:
    pickle.dump(prior, f)


    
    






