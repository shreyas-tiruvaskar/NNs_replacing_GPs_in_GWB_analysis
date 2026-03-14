# making sure the notebook is run from the right directory, as lots of paths are relative to the root directory
from pathlib import Path
PROJECT_ROOT = Path.cwd().parents[0]

if not (PROJECT_ROOT / 'holodeck_output_lib_gp_mcmc').exists():
    raise RuntimeError("Please run Jupyter from the notebooks_to_generate_plots directory")

## change directory to where you have your holodeck directory installed

import platform
import multiprocessing
import subprocess
import time
import datetime

N_CORES = 128  # requested cores

print("Processor:", platform.processor())
print("Machine:", platform.machine())
print("CPU count (logical):", multiprocessing.cpu_count())
print("Environment cores requested:", N_CORES)
print("Python version:", platform.python_version())

start_time = time.perf_counter()
start_readable = datetime.datetime.now()
print("Start time:", start_readable)

# Command as a list (safer than string)
cmd = [
    # "srun",
    "mpirun", "-np", "128",
    "python", "-m", "holodeck.librarian.gen_lib",
    "-n", "2000",
    "-r", "2000",
    "-f", "5",
    "--gwb",
    "--no-ss",
    "--no-params",
    "PS_Classic_Phenom_Uniform",
    "/home/sti50/neural_network/notebooks_and_files_for_final_results/time_related_files/Phenom_Uniform_n2000_r2000_f5/library/" # or wherever you want to save it
]

# Run MPI command
subprocess.run(cmd, check=True)

end_time = time.perf_counter()
end_readable = datetime.datetime.now()
print("End time:", end_readable)

elapsed_minutes = (end_time - start_time) / 60
print(f"Total wall-clock time: {elapsed_minutes:.2f} minutes")
print(f"Total wall-clock time: {elapsed_minutes/60:.2f} hours")