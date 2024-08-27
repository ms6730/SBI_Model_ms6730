import pickle
import cloudpickle
import shutil
import parflow
from parflow import Run
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.fs import mkdir
import subsettools as st
import pandas as pd
import json 

#read in variables from the json file
json_path = '/home/at8471/c2_sbi_experiments/hydrogen-sbi/scripts/settings.json' #probably need a better way to do this step
with open(json_path, 'r') as file:
    settings = json.load(file)
    
ens_num=settings['ens_num']
base_dir = settings['base_dir']
runname = settings['runname']
huc = settings['huc']
num_hours = settings['hours']
start = settings['start']
end = settings['end']
mannings_file = settings['base_mannings_file']
num_sims = settings['num_sims']
P = settings['P']
Q = settings['Q']
orig_vals_path = f"{base_dir}/{runname}_filtered_orig_vals.csv"
filtered_df=pd.read_csv(orig_vals_path)

# read the latest proposal
try:
    with open(f"{base_dir}/{runname}_posterior.pkl", "rb") as fp:
        prior = pickle.load(fp)
except FileNotFoundError:
    with open(f"{base_dir}/{runname}_prior.pkl", "rb") as fp:
        prior = pickle.load(fp)
    
theta = prior.sample((num_sims,)).numpy()
theta_df = pd.DataFrame(theta, columns=filtered_df.columns)
theta_df.to_csv(f"{base_dir}/{runname}_parameters_ens{ens_num}.csv", index=False)

subset_mannings = read_pfb(f"{base_dir}/outputs/{runname}/{mannings_file}.pfb")

for row in range(len(theta_df)):
    run_dir = f"{base_dir}/outputs/{runname}_{ens_num}_{row}/"
    mkdir(run_dir)
    new_mannings = subset_mannings.copy()
    
    for col in filtered_df.columns:
        print(col)
        orig_val = filtered_df[col][0]
        print(orig_val)
        new_val = theta_df.iloc[row][col]

        new_mannings[new_mannings == orig_val] = new_val

    write_pfb(f"{run_dir}/{mannings_file}_{row}.pfb", new_mannings, p=P, q=Q, dist=True)

    st.copy_files(read_dir=f"{base_dir}/inputs/{runname}/static", write_dir=run_dir)

    runscript_path = f"{run_dir}/{runname}.yaml"
    
    shutil.copy(f"{base_dir}/outputs/{runname}/{runname}.yaml", runscript_path)
    
    runscript_path = st.change_filename_values(
        runscript_path=runscript_path,
        mannings = f"{mannings_file}_{row}.pfb"
    )
    
    st.dist_run(P, Q, runscript_path, working_dir=run_dir, dist_clim_forcing=False)

# save current inference, proposal and posterior
try:
    src_file = f"{base_dir}/{runname}_posterior.pkl"
    dest_file=f"{base_dir}/{runname}_posterior{ens_num-1}.pkl"
    shutil.copyfile(src_file, dest_file)

    src_file = f"{base_dir}/{runname}_proposal.pkl"
    dest_file=f"{base_dir}/{runname}_proposal{ens_num-1}.pkl"
    shutil.copyfile(src_file, dest_file)
    
    src_file = f"{base_dir}/{runname}_inference.pkl"
    dest_file=f"{base_dir}/{runname}_inference{ens_num-1}.pkl"
    shutil.copyfile(src_file, dest_file)
except:
    pass