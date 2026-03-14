libraries (library and library2_gp_nn_accuracy) are generated using the following commands: (make sure to change directory to the holodeck directory before running the following commands)

mpirun -np 128 python -m holodeck.librarian.gen_lib -n 4000 -r 2000 -f 5 --gwb --no-ss --no-params 'PS_Classic_SIDM_Astro_Uniform_Extended_Version2' \
     '/home/sti50/neural_network/notebooks_and_files_for_final_results/holodeck_output_lib_gp_mcmc/SIDM_Astro_Uniform_Extended_Version2_n4000_r2000_f5/library/'


mpirun -np 128 python -m holodeck.librarian.gen_lib -n 4000 -r 2000 -f 5 --gwb --no-ss --no-params 'PS_Classic_SIDM_Astro_Uniform_Extended_Version2' \
     '/home/sti50/neural_network/notebooks_and_files_for_final_results/holodeck_output_lib_gp_mcmc/SIDM_Astro_Uniform_Extended_Version2_n4000_r2000_f5/library2_gp_nn_accuracy/'

