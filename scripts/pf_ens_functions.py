import numpy as np
import os
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
        

def create_mannings_ensemble(base_path, baseline_runname, mannings_file, num_ens_mem=5, P=1, Q=1, scalar = 10, ens_num=0, sample_file_path=None):
    subset_mannings = read_pfb(f"{base_path}/outputs/{baseline_runname}/{mannings_file}.pfb")
    filters = {"dataset":"conus2_domain", "variable":"mannings"}
    mannings_map = hf.get_gridded_data(filters)
    all_vals = np.unique(mannings_map)
    subset_vals = np.unique(subset_mannings)
    mannings_dict = {}
        
    for i in range(len(all_vals)):
        mannings_dict[f"m{i}"]=[all_vals[i]]
        
    filtered_dict = {k: v for k, v in mannings_dict.items() if v in subset_vals}
    
    if sample_file_path is not None:
        sample_df = pd.read_csv(sample_file_path)
        new_ens_dict = sample_df.to_dict(orient='columns')
        print(new_ens_dict)
        
    for i in range(0, num_ens_mem):
        run_dir = f"{base_path}/outputs/{baseline_runname}_{i}/"
        mkdir(run_dir)
        
        new_mannings = subset_mannings.copy()
        
        for key in filtered_dict.keys():
            orig_val = filtered_dict[key][0]
            if sample_file_path is None:
                low_bound = orig_val / scalar
                high_bound = orig_val * scalar
                new_val = np.random.uniform(low_bound, high_bound)
    
                filtered_dict[key].append(new_val)
            else:
                new_val = new_ens_dict[key][i]

            new_mannings[new_mannings == orig_val] = new_val

        write_pfb(f"{run_dir}/{mannings_file}_{i}.pfb", new_mannings, p=P, q=Q, dist=True)

        st.copy_files(read_dir=f"{base_path}/outputs/{baseline_runname}", write_dir=run_dir)

        runscript_path = f"{run_dir}/{baseline_runname}.yaml"
        
        runscript_path = st.change_filename_values(
            runscript_path=runscript_path,
            mannings = f"mannings_{i}.pfb"
        )

    if sample_file_path is None:
        df = pd.DataFrame(filtered_dict)
        df.to_csv(f"{base_path}/{baseline_runname}_mannings_ens{ens_num}.csv", index=False)

def setup_baseline_run(base_dir, runname, hucs, start, end, grid="conus2", var_ds="conus2_domain", forcing_ds="CW3E", P=1, Q=1, hours = 96):
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
    
    clm_paths = st.config_clm(ij_bounds, start=start, end=end, dataset=var_ds, write_dir=static_write_dir)
    
    forcing_paths = st.subset_forcing(
        ij_bounds,
        grid=grid,
        start=start,
        end=end,
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
