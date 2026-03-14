# making sure the notebook is run from the right directory, as lots of paths are relative to the root directory
from pathlib import Path
PROJECT_ROOT = Path.cwd().parents[0]

if not (PROJECT_ROOT / 'holodeck_output_lib_gp_mcmc').exists():
    raise RuntimeError("Please run Jupyter from the notebooks_and_files_for_final_results directory")


# We want a GP trained on 8k points to predict for test libraries (like 4k, e.g.)

import h5py
import numpy as np
from holodeck.gps import gp_utils
import pickle, h5py, glob, os
from pathlib import Path
NFREQS = 5


## Phenom ##
# GP predictions for training set
N_train = 2000 
npz_path = f'{PROJECT_ROOT}/holodeck_output_lib_gp_mcmc/Phenom_Uniform_n{N_train}_r2000_f5/library/gp_predictions_for_{N_train}_train_points.npz'
hdf5_file_path = f'{PROJECT_ROOT}/holodeck_output_lib_gp_mcmc/Phenom_Uniform_n{N_train}_r2000_f5/library/sam-library.hdf5'
gp_dir = Path(f'{PROJECT_ROOT}/holodeck_output_lib_gp_mcmc/Phenom_Uniform_n{N_train}_r2000_f5/gp_training_output')
##############

# hdf5 file path for library

with h5py.File(hdf5_file_path, 'r') as f:
    param_space_test = f['sample_params'][:]    


# for trained med GPs    
# Look for a file ending with 'med.pkl' in the specified directory
med_pkl_files = list(gp_dir.glob('*med.pkl'))
if med_pkl_files:
    # If such a file exists, use its path
    trainedGPdir = str(med_pkl_files[-1])
    print(f"No. of med.pkl files present: {len(med_pkl_files)}")
    print(f"File name/names:\n" + "\n".join(pkl.name for pkl in med_pkl_files))
    print(f"The chosen GP file is : {trainedGPdir} \n")
else:
    # Handle the case where no matching file is found
    raise FileNotFoundError(f"No file ending with 'med.pkl' found in {gp_dir}")

## for trained std GPs
# Look for a file ending with 'std.pkl' in the specified directory
std_pkl_files = list(gp_dir.glob('*std.pkl'))
if std_pkl_files:
    # If such a file exists, use its path
    trained_varGPdir = str(std_pkl_files[-1])
    print(f"No. of std.pkl files present: {len(std_pkl_files)}")
    print(f"File name/names:\n" + "\n".join(pkl.name for pkl in std_pkl_files))
    print(f"The chosen GP file is : {trained_varGPdir} \n")
else:
    # Handle the case where no matching file is found
    raise FileNotFoundError(f"No file ending with 'std.pkl' found in {gp_dir}")
# GP for 8000 points
GW_var_gp_george_pkl = [trained_varGPdir]
GW_gp_george_pkl = [trainedGPdir]


# open gp_george objects
GW_gp_george = pickle.load(open(GW_gp_george_pkl[0], 'rb'))
GW_var_gp_george = pickle.load(open(GW_var_gp_george_pkl[0], 'rb'))
# make gaussprob objects
spectra = None
GW_gp = gp_utils.set_up_predictions(spectra, GW_gp_george)
GW_var_gp = gp_utils.set_up_predictions(spectra, GW_var_gp_george)

# ## commenting out because the result of this is saved in an npz file

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import numpy as np

N_cores = 64 # rch

def run_gp(i):
    _, rho, rho_pred = gp_utils.hc_from_gp(
        GW_gp_george[:NFREQS], GW_gp[:NFREQS],
        GW_var_gp_george[:NFREQS], GW_var_gp[:NFREQS],
        param_space_train[i, :]
    )
    return (
        rho, # predicted median values of log10(strain^2)
        rho_pred[:, 1] # total predicted uncertainty of log10(strain^2) 
        # (this is equal to np.sqrt(std_pred**2 + std_pred_unc**2 + mean_pred_unc**2))- holodeck/gps/gp_utils.py line 677
    )

with tqdm_joblib(tqdm(total=param_space_train.shape[0])) as progress_bar:
    results = Parallel(n_jobs=N_cores)(
        delayed(run_gp)(i) for i in range(param_space_train.shape[0])
    )

pred_med_log10_hc2_array = np.zeros((param_space_train.shape[0], NFREQS))
pred_std_log10_hc2_array = np.zeros((param_space_train.shape[0], NFREQS))

# Unpack results into arrays
for i, (pred_med_log10_hc2, pred_std_log10_hc2) in enumerate(results):
    pred_med_log10_hc2_array[i] = pred_med_log10_hc2
    pred_std_log10_hc2_array[i] = pred_std_log10_hc2

# saving this in npz files


np.savez(npz_path, gp_prediction_median_log10_hc2 = pred_med_log10_hc2_array, gp_prediction_std_log10_hc2 = pred_std_log10_hc2_array)

