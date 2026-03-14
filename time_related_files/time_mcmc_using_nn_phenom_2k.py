# making sure the notebook is run from the right directory, as lots of paths are relative to the root directory
from pathlib import Path
PROJECT_ROOT = Path.cwd().parents[0]

if not (PROJECT_ROOT / 'holodeck_output_lib_gp_mcmc').exists():
    raise RuntimeError("Please run Jupyter from the notebooks_and_files_for_final_results directory")

# copied from /home/sti50/neural_network/mcmc_gen_using_nn_8k.py

# not using external multiprocessing, as the internal one works well

from pathlib import Path
import pickle
import numpy as np
from enterprise.signals import parameter
from ceffyl.ceffyl_gp import ceffylGPSampler
import os


# chain_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
# print(f"Running independent chain {chain_id}")
# np.random.seed(12345 + chain_id)

#################################
Nsamples_mcmc = int(1e4)

n_training = 2000
LIB_DIR = f'{PROJECT_ROOT}/time_related_files/Phenom_Uniform_n2000_r2000_f5/library'
#################################


# Load trained GPs and NNs
gp_dir = Path(LIB_DIR).parent / 'gp_training_output'
nn_dir = Path(LIB_DIR).parent / 'nn_training_output'
trainedGPdir = str(list(gp_dir.glob('*med.pkl'))[-1])
trained_varGPdir = str(list(gp_dir.glob('*std.pkl'))[-1])
spectradir = str(Path(LIB_DIR) / "sam-library.hdf5")

ceffyldir = f'{PROJECT_ROOT}/Data/ceffyl_data/30f_fs{hd+mp+dp+cp}_ceffyl_hd-only'

outdir = str(
    Path(LIB_DIR).parent
    # Path('/home/sti50/neural_network/notebooks_and_files_for_final_results/time_related_files/SIDM_Astro_Extended_Version2_n8000_r2000_f5')
    / 'nn_mcmc_log_likelihood_all_freqs'
    # / 'ceffyl_output_parallel'
    # / f'chain_{chain_id:03d}'
    / 'ceffyl_output/'
)

Path(outdir).mkdir(parents=True, exist_ok=True)

## ceffyl MCMC training
import sys
# sys.path.append('../../ng15yr_astro_interp/')
with open(trainedGPdir, 'rb') as f:
    gp = pickle.load(f)  # open directory of trained GPs

# automated parameter set-up
hp_names = list(gp[0].par_dict.keys())
hyperparams = []
# for hp in hp_names:
#     h = parameter.Uniform(gp[0].par_dict[hp]['min'],
#                             gp[0].par_dict[hp]['max'])(hp)
#     hyperparams.append(h)
##################################################################
# added on 06.02.2026 to apply astro priors correctly 
for hp in hp_names:
    if (hp=='hard_time') or (hp=='hard_gamma_inner'):
        h = parameter.Uniform(gp[0].par_dict[hp]['min'], gp[0].par_dict[hp]['max'])(hp)
    # uniform priors
    elif hp=='gsmf_phi0_log10':
        h = parameter.Uniform(gp[0].par_dict[hp]['min'], gp[0].par_dict[hp]['max'])(hp)
    elif hp=='gsmf_mchar0_log10':
        h = parameter.Uniform(gp[0].par_dict[hp]['min'], gp[0].par_dict[hp]['max'])(hp)
    elif hp=='mmb_mamp_log10':
        h = parameter.Uniform(gp[0].par_dict[hp]['min'], gp[0].par_dict[hp]['max'])(hp)
    elif hp=='mmb_scatter_dex':
        h = parameter.Uniform(gp[0].par_dict[hp]['min'], gp[0].par_dict[hp]['max'])(hp)
    
    # astro priors
    # elif hp=='gsmf_phi0_log10':
    #     h = parameter.Normal(mu=-2.56, sigma=0.4)(hp)
    # elif hp=='gsmf_mchar0_log10':
    #     h = parameter.Normal(mu=10.9, sigma=0.4)(hp)
    # elif hp=='mmb_mamp_log10':
    #     h = parameter.Normal(mu=8.5, sigma=0.2)(hp)
    # elif hp=='mmb_scatter_dex':
    #     h = parameter.Normal(mu=0.32, sigma=0.15)(hp)

    hyperparams.append(h)
###################################################################

test_constant = False  # test if parameter.constant function works
if test_constant:
    hyperparams[-1] = parameter.Constant(1.5)

Nfreqs = 5  # number of frequencies to fit (same as no. of frequencies used to train GPs)
freq_idxs=np.arange(0, Nfreqs) # added by me

# set up sampler!
print('initializing sampler')
sampler = ceffylGPSampler(trainedGP=trainedGPdir, spectrafile=spectradir,
                            trained_varGP=trained_varGPdir, # added by me
                            ceffyldir=ceffyldir, hyperparams=hyperparams,
                            Nfreqs=Nfreqs, outdir=outdir,
                            freq_idxs=freq_idxs, # added by me
                            nn_dir=nn_dir, # added by me
                            analysis_type='nn', jump=True)

print('Here are your parameters...\n')
print(sampler.ceffyl_gp.param_names)
x0 = sampler.ceffyl_gp.initial_samples()

import time
start = time.perf_counter()

sampler.sampler.sample(x0, Nsamples_mcmc)

end = time.perf_counter()
total_wall_time = end - start
print(f"Total MCMC wall time: {total_wall_time:.3f} seconds")
print(f"Total MCMC wall time: {total_wall_time/60:.3f} minutes")