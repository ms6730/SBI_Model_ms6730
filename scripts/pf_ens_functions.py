import os
import parflow
from parflow import Run
from parflow.tools.fs import mkdir
import subsettools as st
import hf_hydrodata as hf
import numpy as np
import pandas as pd
import xarray as xr
import shutil
        
#make functions have keyworded arguments, 
def setup_baseline_run(base_dir, runname, hucs, start, end, P, Q, hours, grid="conus2", var_ds="conus2_domain", forcing_ds="CW3E", tz="UTC", init_press_file_path=None):
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

    if init_press_file_path is not None:
        shutil.copy(init_press_file_path, static_write_dir)
        init_press_path = os.path.basename(init_press_file_path)
    else:
        init_press_path = os.path.basename(static_paths["ss_pressure_head"])
        
    depth_to_bedrock_path = os.path.basename(static_paths["pf_flowbarrier"])

    st.copy_files(read_dir=static_write_dir, write_dir=pf_out_dir)
    
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
    run.Solver.CLM.MetFileName = forcing_ds
    
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
