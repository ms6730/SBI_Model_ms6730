import sys
import os
os.environ['PARFLOW_DIR'] = '/home/SHARED/software/parflow/3.10.0'
#path to gpu build on della 
#os.environ['PARFLOW_DIR'] = '/home/ga6/parflow_mgsemi_new/parflow'
from parflow import Run
from parflow.tools.settings import set_working_directory
from pathlib import Path
from glob import glob
from parflow.tools.io import read_pfb, read_clm, read_pfb_sequence
from pf_ens_functions import calculate_water_table_depth, calculate_flow
import xarray as xr
import numpy as np
import pandas as pd
import parflow as pf


#read in variables from the json file
i = sys.argv[1]
json_path = sys.argv[2]

with open(json_path, 'r') as file:
    settings = json.load(file)
    
base_dir = settings['base_dir']
runname = settings['runname']
start = settings['start']
num_hours = settings['hours']
ens_num = settings['ens_num']

out_dir=f"{base_dir}/outputs/{runname}_{ens_num}_{i}"
set_working_directory(f'{out_dir}')

run = Run.from_definition(f'{out_dir}/{runname}.yaml')
run.TimingInfo.StopTime = num_hours
run.run(working_directory=f'{out_dir}')

# Postprocessing
data = run.data_accessor
slope_x_file = f'{out_dir}/slope_x.pfb'
slope_x = pf.read_pfb(slope_x_file)

slope_y_file = f'{out_dir}/slope_y.pfb'
slope_y = pf.read_pfb(slope_y_file)
mannings = pf.read_pfb(f'{out_dir}/mannings_{i}.pfb')

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
    read_pfb(f'{out_dir}/mannings_{i}.pfb')[0,:,:],
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

ds.to_netcdf(f'{out_dir}/{runname}_{i}.nc', mode='w')

# Clean up
del ds
del clm
_ = [os.remove(os.path.abspath(f)) for f in pressure_files]
_ = [os.remove(os.path.abspath(f)+'.dist') for f in pressure_files]
_ = [os.remove(os.path.abspath(f)) for f in saturation_files]
_ = [os.remove(os.path.abspath(f)+'.dist') for f in saturation_files]
_ = [os.remove(os.path.abspath(f)) for f in clm_files]
_ = [os.remove(os.path.abspath(f)+'.dist') for f in clm_files]