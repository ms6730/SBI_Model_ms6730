import numpy as np
import os
import glob
import parflow
from parflow import Run
from parflow.tools.io import read_pfb, write_pfb, read_clm
from parflow.tools.fs import mkdir
from parflow.tools.settings import set_working_directory
import subsettools as st
import hf_hydrodata as hf
import random
import shutil
import pandas as pd
import xarray as xr
import parflow as pf
import subprocess
        

def create_mannings_ensemble(base_path, runname, mannings_file, orig_vals_path, proposal, num_sims=5, P=1, Q=1, ens_num=0):
    subset_mannings = read_pfb(f"{base_path}/outputs/{runname}/{mannings_file}.pfb")
    filtered_df=pd.read_csv(orig_vals_path)
    
    theta = proposal.sample((num_sims,)).numpy()
    theta_df = pd.DataFrame(theta, columns=filtered_df.columns)
    theta_df.to_csv(f"{base_path}/outputs/{runname}_mannings_ens{ens_num}.csv", index=False)
    
    for row in range(len(theta_df)):
        run_dir = f"{base_path}/outputs/{runname}_{row}/"
        mkdir(run_dir)
        new_mannings = subset_mannings.copy()
        
        for col in filtered_df.columns():
            orig_val = filtered_df[col][0]
            new_val = theta_df.iloc[row][col]
    
            new_mannings[new_mannings == orig_val] = new_val

        write_pfb(f"{run_dir}/{mannings_file}_{i}.pfb", new_mannings, p=P, q=Q, dist=True)

        st.copy_files(read_dir=f"{base_path}/inputs/{runname}/static", write_dir=run_dir)

        runscript_path = f"{run_dir}/{runname}.yaml"
        
        shutil.copy(f"{base_path}/outputs/{runname}/{runname}.yaml", runscript_path)
        
        runscript_path = st.change_filename_values(
            runscript_path=runscript_path,
            mannings = f"{mannings_file}_{i}.pfb"
        )
        
        st.dist_run(P, Q, runscript_path, working_dir=run_dir, dist_clim_forcing=False)

def setup_baseline_run(base_dir, runname, hucs, start, end, grid="conus2", var_ds="conus2_domain", forcing_ds="CW3E", P=1, Q=1, hours = 96, tz="UTC"):
    #make directories
    input_dir = os.path.join(base_dir, "inputs", f"{runname}")
    output_dir = os.path.join(base_dir, "outputs")
    static_write_dir = os.path.join(input_dir, "static")
    mkdir(static_write_dir)
    forcing_dir = os.path.join(input_dir, "forcing")
    mkdir(forcing_dir)
    pf_out_dir = os.path.join(output_dir, f"{runname}")
    mkdir(pf_out_dir)

    reference_run = st.get_template_runscript(grid, "transient", "solid", pf_out_dir)

    ij_bounds, mask = st.define_huc_domain(hucs=hucs, grid=grid)

    mask_solid_paths = st.write_mask_solid(mask=mask, grid=grid, write_dir=static_write_dir)
    
    static_paths = st.subset_static(ij_bounds, dataset=var_ds, write_dir=static_write_dir)
    
    clm_paths = st.config_clm(ij_bounds, start=start, end=end, dataset=var_ds, write_dir=static_write_dir, time_zone=tz)
    
    forcing_paths = st.subset_forcing(
        ij_bounds,
        grid=grid,
        start=start,
        end=end,
        time_zone=tz,
        dataset=forcing_ds,
        write_dir=forcing_dir,
    )
    
    runscript_path = st.edit_runscript_for_subset(
        ij_bounds,
        runscript_path=reference_run,
        runname=runname,
        forcing_dir=forcing_dir,
    )
    
    st.copy_files(read_dir=static_write_dir, write_dir=pf_out_dir)
    
    init_press_path = os.path.basename(static_paths["ss_pressure_head"])
    depth_to_bedrock_path = os.path.basename(static_paths["pf_flowbarrier"])
    
    runscript_path = st.change_filename_values(
        runscript_path=runscript_path,
        init_press=init_press_path,
        depth_to_bedrock = depth_to_bedrock_path
    )
    
    runscript_path = st.dist_run(
        topo_p=P,
        topo_q=Q,
        runscript_path=runscript_path,
        dist_clim_forcing=True,
    )

    run = Run.from_definition(runscript_path)
    run.TimingInfo.StopTime = hours 
    run.Solver.CLM.MetFileName = 'CW3E'
    #turn on for running on a gpu parflow build
    #run.Solver.Linear.Preconditioner = 'MGSemi'
    #run.Solver.Nonlinear.UseJacobian = True
    #run.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'
    run.write(working_directory=pf_out_dir,file_format='yaml')

def get_parflow_output_nc(
    pf_run_nc_path, 
    obsv_metadata_path,
    var_name, 
    write_path):


    ds = xr.open_dataset(pf_run_nc_path)
    
    obs_metadata_df = pd.read_csv(obsv_metadata_path)
    num_sites = len(obs_metadata_df)

    for row in range(num_sites):
        site_id = str(obs_metadata_df.loc[row,'site_id'])
        site_id = f"0{site_id}"
        j = obs_metadata_df.loc[row,'domain_j']
        i = obs_metadata_df.loc[row,'domain_i']
        
        time_series = ds.sel(y = j, x = i)[var_name]
        ts_df = time_series.to_dataframe().reset_index()
        ts_df = ts_df[['time',var_name]]
        ts_df.rename(columns={var_name: site_id}, inplace=True)
        ts_df.rename(columns={'time': 'date'}, inplace=True)
        
        if row == 0: 
            sim_df = ts_df
        else:
            sim_df = pd.merge(sim_df, ts_df, on='date')

    sim_df.loc[:, sim_df.columns != 'date'] = sim_df.loc[:, sim_df.columns != 'date'] / 3600
    sim_df.to_csv(write_path, index=False)

    return sim_df

def calculate_water_table_depth(ds, dz):
    wtd_list = []
    for t in range(len(ds['time'])):
        wtd_list.append(parflow.tools.hydrology.calculate_water_table_depth(
            ds['pressure'].values[t], 
            ds['saturation'].values[t], 
            dz=dz
        ))
    wtd = xr.DataArray(
        np.stack(wtd_list),
        coords={'time': ds['time']},
        dims=('time', 'y', 'x')
    )
    return wtd


def calculate_flow(ds, slope_x, slope_y, mannings, dx, dy, mask):
    flow_list = []
    for t in range(len(ds['time'])):
        flow_list.append(parflow.tools.hydrology.calculate_overland_flow_grid(
            ds['pressure'].values[t], 
            slope_x, slope_y, mannings, dx, dy, mask=mask
        ))
    flow = xr.DataArray(
        np.stack(flow_list),
        coords={'time': ds['time']},
        dims=('time', 'y', 'x')
    )
    return flow

def backup_previous(filename):
    files = glob.glob(filename+"*")
    if len(files) == 0:
        return
    num = len(files) - 1
    new_name = f"{filename}.{num}"   
    os.rename(filename, new_name)
