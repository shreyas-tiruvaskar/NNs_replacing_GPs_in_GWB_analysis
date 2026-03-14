In the directory ./holodeck_output_lib_gp_mcmc there are 4 folders (3 for SIDM and 1 for Phenom)

The gp_trainer and mcmc python files in this directory can be used to 
- train GPs on the libraries from those 4 folders
- generate MCMC chains
respectively.

In the SIDM related python files, the variable n_training or NSAMPLES can be changed to 2000, 4000, or 8000 for the corresponding use.

gp_predictions python files are used to generate predictions of GPs for test and training set, which helps generate Figure 8  and Figure 9 (GP vs NN errors)
