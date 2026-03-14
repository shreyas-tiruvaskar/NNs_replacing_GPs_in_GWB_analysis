For SIDM_Astro_Uniform cases, applying UNIFORM prior (also for astro params) while generating the library

Applying ASTRO (Normal) prior while generating MCMC

For this, easiest way was- create the library and train GP using param space which has uniform prior on astro params

Only while generating MCMC, in the MCMC generation file, apply the Normal prior to parameters

The file sidm_sigma_over_m_using_chains.py is used for SIDM models, to generate mcmc chains for maximum contributing M, q, z combinations for each frequency bin.
sidm_lambda0_yvalues_for_cvalues_1_to_5_for_500_points.npz- this file is being used in the above mentioned calculation


The SIDM and Phenom files were originally generated in the location:
/home/sti50/project_shreyas_sidm/holodeck_output/post_holodeck_redshift_minus1_fix/correct_prior_implementation/