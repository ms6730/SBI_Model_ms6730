import pickle
import shutil
import parflow
from parflow import Run
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.fs import mkdir
import subsettools as st
import pandas as pd

ens_num=0
runname = 'sinnemahoning'
base_dir = '/home/at8471/c2_sbi_experiments/sbi_framework'
mannings_file = 'mannings'
orig_vals_path = f"{base_dir}/{runname}_filtered_orig_vals.csv"
huc = '02050202'
num_sims = 5
start = "2002-10-27"
end = "2002-12-01"
num_hours = 840
P = 4
Q = 4

# read the latest proposal
try:
    fp = open(f"{base_dir}/{runname}_posterior.pkl", "rb")
except FileNotFoundError:
    fp = open(f"{base_dir}/{runname}_prior.pkl", "rb")
proposal = pickle.load(fp)

subset_mannings = read_pfb(f"{base_dir}/outputs/{runname}/{mannings_file}.pfb")
filtered_df=pd.read_csv(orig_vals_path)

theta = proposal.sample((num_sims,)).numpy()
theta_df = pd.DataFrame(theta, columns=filtered_df.columns)
theta_df.to_csv(f"{base_dir}/{runname}_parameters_ens{ens_num}.csv", index=False)

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

#save current inference and posterior
try:
    src_file = f"{base_dir}/{runname}_posterior.pkl"
    dest_file=f"{base_dir}/{runname}_posterior{ens_num-1}.pkl"
    shutil.copyfile(src_file, dest_file)
    
    src_file = f"{base_dir}/{runname}_inference.pkl"
    dest_file=f"{base_dir}/{runname}_inference{ens_num-1}.pkl"
    shutil.copyfile(src_file, dest_file)
except:
    pass