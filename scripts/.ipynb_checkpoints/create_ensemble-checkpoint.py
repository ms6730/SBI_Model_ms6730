from pf_ens_functions import create_mannings_ensemble

base_path = '/home/at8471/c2_sbi_experiments/sbi_framework'
huc = '02050202'
runname = 'sinnemahoning'
num_mems = 5
start = "2002-10-01"
end = "2002-12-31"
num_hours = 2208
P = 4
Q = 4
orig_mannings_file_name = 'mannings'
ens_num=0

create_mannings_ensemble(base_path = base_path, baseline_runname = runname, mannings_file = orig_mannings_file_name, num_ens_mem=num_mems, P=P, Q=Q, ens_num = ens_num)


