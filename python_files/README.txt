In the directory ./holodeck_output_lib_gp_mcmc there are 4 folders (3 for SIDM and 1 for Phenom)

The gp_trainer and mcmc python files in this directory can be used to 
- train GPs on the libraries from those 4 folders
- generate MCMC chains
respectively.

In the SIDM related python files, the variable n_training or NSAMPLES can be changed to 2000, 4000, or 8000 for the corresponding use.

gp_predictions python files are used to generate predictions of GPs for test and training set, which helps generate Figure 8  and Figure 9 (GP vs NN errors)

The sigma_over_m_using_chains_fixed_mtot_for_2k_4k_gp_nn.py can be used to create the MCMC chains with sigma/m using maximum contributing Mtot for GP/NN MCMCs for 8k training sets
The sigma_over_m_using_chains_fixed_mtot_for_2k_4k_gp_nn.py can be run to create the MCMC chains with sigma/m for GP/NN MCMCs using the same maximum contributing Mtot as 8k case for 2k and 4k training points