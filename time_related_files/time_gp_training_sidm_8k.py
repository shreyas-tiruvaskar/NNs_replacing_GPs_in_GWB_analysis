# making sure the notebook is run from the right directory, as lots of paths are relative to the root directory
from pathlib import Path
PROJECT_ROOT = Path.cwd().parents[0]

if not (PROJECT_ROOT / 'holodeck_output_lib_gp_mcmc').exists():
    raise RuntimeError("Please run Jupyter from the notebooks_and_files_for_final_results directory")

# this is the file that created a GP that is trained enough
import platform
import multiprocessing
import os
import time

import numpy as np
import holodeck
from pathlib import Path
import h5py
from holodeck.constants import YR, NWTG, SPLC, GYR, PC, MSOL
import holodeck.gps.gp_utils as gu
from datetime import datetime
import pickle
import os

PARAM_SPACE = 'SIDM_Astro_Uniform_Extended_Version2'
NSAMPLES = 8000
NREALS = 2000
NFREQS = 5

OUTPUT = f'{PROJECT_ROOT}/time_related_files/{PARAM_SPACE}_n{NSAMPLES}_r{NREALS}_f{NFREQS}/library/'

os.chdir('/home/sti50/holodeck_repository_rch_shreyas/holodeck')

N_CORES = 128 # rch


print("Processor:", platform.processor())
print("Machine:", platform.machine())
print("CPU count (logical):", multiprocessing.cpu_count())
print("Environment cores requested:", N_CORES)
print("Python version:", platform.python_version())

##########################################
# spectra_file = Path(OUTPUT + 'sam-library.hdf5')
# spectra = h5py.File(spectra_file, 'r+') # this is to be able to edit the key name
# spectra['fobs'] = spectra.pop('fobs_cents') # renames fobs_cents to fobs (got it from chatgpt)
# spectra.close()
#####################################

SPECTRA_FILE_NAME = (
    OUTPUT +
    "/sam-library.hdf5"
)


# based on https://github.com/nanograv/holodeck/blob/dev/scripts/gp_config_TEMPLATE.ini
NFREQS_TRAINING = 5
TEST_FRAC = 0.0
BURN_FRAC = 0.25
# NWALKERS = 30
NWALKERS = N_CORES
NSAMPLES = 500
########################################################################################
MPI = False
CENTER_MEASURE = "median"
# added by me 
# because even though the input expected in train_gp() is a string like "ExpSquaredKernel"
# but when train_gp() passes it to create_gp_kernels(), it expects a dictionary instead
spectra_file = Path(SPECTRA_FILE_NAME)
spectra = h5py.File(spectra_file, 'r')
pars = list(spectra.attrs["param_names"].astype(str))
KERNEL = {par: "ExpSquaredKernel" for par in pars}

yisvariance = [False, True]
trainedGPdir, trained_varGPdir = '', ''
trained_gp_address = [trainedGPdir, trained_varGPdir]

start_time = time.perf_counter()

for i in range(0, 2):
    Y_IS_VARIANCE = yisvariance[i]
    # trying to perform GP training directly similar to the code on https://github.com/nanograv/holodeck/blob/dev/scripts/gp_trainer.py
    trained_gps = gu.train_gp(spectra_file=spectra_file,
                                nfreqs=NFREQS_TRAINING,
                                nwalkers=NWALKERS,
                                nsamples=NSAMPLES,
                                burn_frac=BURN_FRAC,
                                test_frac=TEST_FRAC,
                                center_measure=CENTER_MEASURE,
                                y_is_variance=Y_IS_VARIANCE,
                                mpi=MPI,
                                kernel=KERNEL)

    # Add datestring to ensure unique name
    datestr = datetime.now().strftime('%Y%m%d_%H%M%S')

    # # added by me
    if(Y_IS_VARIANCE):
        y_is_variance = "std"
    else: y_is_variance = "med"

    # Save the trained GP as a pickle to be used with PTA data!
    # gp_file = Path("trained_gp_on_" + f"{NFREQS_TRAINING}f_n{NSAMPLES}_" + spectra_file.parent.parent.name + "_" + datestr + "_" + y_is_variance +
    #                 ".pkl")
    gp_file = Path(y_is_variance + ".pkl")
    loc_gp_file = Path(OUTPUT).parent / "gp_training_output" / gp_file
    trained_gp_address[i] = loc_gp_file # to save the gp address

    # Create the directory if it doesn't exist
    loc_gp_file.parent.mkdir(parents=True, exist_ok=True) # asked chatgpt

    with open(loc_gp_file, "wb") as gpf:
        pickle.dump(trained_gps, gpf)
    print(f"GPs are saved at {loc_gp_file}")

end_time = time.perf_counter()

elapsed_minutes = (end_time - start_time) / 60
print(f"Training time: {elapsed_minutes:.2f} minutes")
