# making sure the notebook is run from the right directory, as lots of paths are relative to the root directory
from pathlib import Path
PROJECT_ROOT = Path.cwd().parents[0]

if not (PROJECT_ROOT / 'holodeck_output_lib_gp_mcmc').exists():
    raise RuntimeError("Please run Jupyter from the notebooks_to_generate_plots directory")


import holodeck as holo
import holodeck
from holodeck.constants import YR, NWTG, SPLC, GYR, PC, MSOL
from holodeck.galaxy_profiles import NFW
from holodeck import utils
from holodeck.librarian import DEF_NUM_FBINS, DEF_NUM_LOUDEST, DEF_PTA_DUR
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp, quad
import numpy as np
import kalepy as kale
from tqdm import tqdm
import matplotlib.pyplot as plt

from pathlib import Path
import os
import la_forge.core as co
from ceffyl.chain_utils import chain_utils
import la_forge.diagnostics as dg
from chainconsumer.chainconsumer import ChainConsumer
from emcee.autocorr import integrated_time

import pandas as pd
from multiprocessing import Pool

#####################################################
##### CHECK THIS BEFORE RUNNING #####################
num_cores = 128
#####################################################

# To find the max contributing M,q,z
# PARAM_SPACE = 'SIDM_Astro_Extended_Version2'
PARAM_SPACE = 'SIDM_Astro_Uniform_Extended_Version2'
NSAMPLES = 2000
NREALS = 2000
NFREQS = 5

chain_path = Path(f'{PROJECT_ROOT}/holodeck_output_lib_gp_mcmc/{PARAM_SPACE}_n{NSAMPLES}_r{NREALS}_f{NFREQS}/nn_mcmc_log_likelihood_all_freqs/ceffyl_output_hd_mp_dp_curn/chain_1.txt')

data = np.loadtxt(chain_path)
print('chain shape without mask', data.shape)
maxLidx = np.argmax(data[:, -4]) # 4th to last column is log posterior
maxL = data[maxLidx, :6] # last 4 columns in mcmc chains here are log posterior, and other stuff

print("Max Likelihood Array For SIDM:", ", ".join("{:.2f}".format(ml) for ml in maxL))  # respects your .format preference

print('chain shape with mask', data.shape)

# strain calculation using the maxL params
nfreqs = 5
param_space = 'PS_Classic_SIDM_Astro_Extended_Version2'
space_name = param_space.split(".")
space_class = holo.librarian.param_spaces_dict[space_name[0]] # /home/users/sti50/holodeck/holodeck/librarian/gen_lib.py (line 498)
print(space_class)
# for edges
pta_dur = holo.librarian.DEF_PTA_DUR # from /home/users/sti50/holodeck/holodeck/librarian/gen_lib.py (line 363)
fobs_cents, fobs_edges = utils.pta_freqs(dur=pta_dur*YR, num=nfreqs) # from line 787 from /home/users/sti50/.conda/envs/holodeck_ceffyl_ptmcmc/lib/python3.9/site-packages/holodeck/librarian/lib_tools.py
fobs_orb_cents = fobs_cents / 2.0
fobs_orb_edges = fobs_edges / 2.0
# if don't want to use resume
space = space_class() # inspired from /home/users/sti50/holodeck/holodeck/librarian/gen_lib.py line 516
# if don't want to use resume and use own params
maxL_param_values = maxL.tolist()
param_names = ['gsmf_phi0_log10', 'gsmf_mchar0_log10', 'mmb_mamp_log10', 'mmb_scatter_dex', 'vt', 'sigma0_over_m_times_t_age_per_1Gyr']
params_maxL = dict(zip(param_names, maxL_param_values))
params = params_maxL
sam, hard = space.model_for_params(params) # from /home/users/sti50/holodeck/holodeck/librarian/gen_lib.py (line 302)

from holodeck.sams import sam_cyutils
from holodeck import utils, cosmo
redz_final, diff_num = sam_cyutils.dynamic_binary_number_at_fobs(fobs_orb_cents, sam, hard, cosmo) # from librarian/lib_tools.py
print('redz_final, diff_num done')
use_redz = redz_final # from librarian/lib_tools.py
edges = [sam.mtot, sam.mrat, sam.redz, fobs_orb_edges] # from librarian/lib_tools.py
number = sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num) # from librarian/lib_tools.py
print('4d number done')
hc2 = holodeck.gravwaves.char_strain_sq_from_bin_edges_redz(edges, use_redz) # from gravwaves.py
print('hc2 4d done')
# with realizations (from gravwaves.py)
realize = 2000
shape = number.shape + (realize,)
hc2 = hc2[..., np.newaxis] * holodeck.gravwaves.poisson_as_needed(number[..., np.newaxis] * np.ones(shape))
hc2_median = np.median(hc2, axis=-1)
print('gwb done')

nfreqs = 5
param_space = 'PS_Classic_GWOnly_Uniform'
space_name = param_space.split(".")
space_class = holo.librarian.param_spaces_dict[space_name[0]] # /home/users/sti50/holodeck/holodeck/librarian/gen_lib.py (line 498)
print(space_class)
# for edges
pta_dur = holo.librarian.DEF_PTA_DUR # from /home/users/sti50/holodeck/holodeck/librarian/gen_lib.py (line 363)
fobs_cents, fobs_edges = utils.pta_freqs(dur=pta_dur*YR, num=nfreqs) # from line 787 from /home/users/sti50/.conda/envs/holodeck_ceffyl_ptmcmc/lib/python3.9/site-packages/holodeck/librarian/lib_tools.py
fobs_orb_cents = fobs_cents / 2.0
fobs_orb_edges = fobs_edges / 2.0
# if don't want to use resume
space = space_class() # inspired from /home/users/sti50/holodeck/holodeck/librarian/gen_lib.py line 516
# if don't want to use resume and use own params
maxL_param_values = maxL[:-2].tolist() # for gw only
param_names = ['gsmf_phi0_log10', 'gsmf_mchar0_log10', 'mmb_mamp_log10', 'mmb_scatter_dex']
params_maxL = dict(zip(param_names, maxL_param_values))
params = params_maxL
sam_gw, hard_gw = space.model_for_params(params) # from /home/users/sti50/holodeck/holodeck/librarian/gen_lib.py (line 302)
from holodeck.sams import sam_cyutils
from holodeck import utils, cosmo
redz_final_gw, diff_num_gw = sam_cyutils.dynamic_binary_number_at_fobs(fobs_orb_cents, sam_gw, hard_gw, cosmo) # from librarian/lib_tools.py
use_redz_gw = redz_final_gw # from librarian/lib_tools.py
edges_gw = [sam_gw.mtot, sam_gw.mrat, sam_gw.redz, fobs_orb_edges] # from librarian/lib_tools.py
number_gw = sam_cyutils.integrate_differential_number_3dx1d(edges_gw, diff_num_gw) # from librarian/lib_tools.py
hc2_gw = holodeck.gravwaves.char_strain_sq_from_bin_edges_redz(edges_gw, use_redz_gw) # from gravwaves.py
# with realizations (from gravwaves.py)
realize = 2000
shape = number_gw.shape + (realize,)
hc2_gw = hc2_gw[..., np.newaxis] * holodeck.gravwaves.poisson_as_needed(number_gw[..., np.newaxis] * np.ones(shape))
hc2_median_gw = np.median(hc2_gw, axis=-1)

hc2_median_effective_sidm = np.abs(hc2_median_gw- hc2_median)
m_idx, q_idx, z_idx = np.unravel_index(np.argmax(hc2_median_effective_sidm[:, :, :, 0]), hc2_median_effective_sidm[:, :, :, 0].shape)

hc2_median_effective_sidm = np.abs(hc2_median_gw- hc2_median)

mtot_array = np.zeros(5)
mrat_array = np.zeros(5)
redz_array = np.zeros(5)

for f in range(fobs_cents.shape[0]):
    m_idx, q_idx, z_idx = np.unravel_index(np.argmax(hc2_median_effective_sidm[:, :, :, f]), hc2_median_effective_sidm[:, :, :, f].shape)
    m_max_contri = (sam.mtot[m_idx + 1] + sam.mtot[m_idx]) / (2 * MSOL) # in solar masses
    q_max_contri = (sam.mrat[q_idx + 1] + sam.mrat[q_idx]) / 2
    z_max_contri = (sam.redz[z_idx + 1] + sam.redz[z_idx]) / 2
    # print(np.unravel_index(np.argmax(np.abs(hc2_median_gw[:, :, :, f]- hc2_median[:, :, :, f])), hc2_median[:, :, :, f].shape))
    print(f'frequency bin no.{f}: max strain contribution from M = {m_max_contri:.2e}, q = {q_max_contri:.2f}, z = {z_max_contri:.2f}')
    mtot_array[f] = m_max_contri
    mrat_array[f] = q_max_contri
    redz_array[f] = z_max_contri

max_contri_mqz_5_freqs_dict = {"mtot_array": mtot_array,
                               "mrat_array": mrat_array,
                               "redz_array": redz_array}
                               
max_contri_mqz_5_freqs_dict_path = chain_path.parent / 'stats_analysis' / 'max_contri_mqz_5_freqs_dict.npz'
os.makedirs(max_contri_mqz_5_freqs_dict_path.parent, exist_ok=True)
np.savez_compressed(max_contri_mqz_5_freqs_dict_path, **max_contri_mqz_5_freqs_dict)

max_contri_mqz_5_freqs_dict = np.load(max_contri_mqz_5_freqs_dict_path)
mtot_array = max_contri_mqz_5_freqs_dict["mtot_array"]
mrat_array = max_contri_mqz_5_freqs_dict["mrat_array"]
redz_array = max_contri_mqz_5_freqs_dict["redz_array"]

# for mtot in mtot_array:
for i in range(len(mtot_array)):
    mtot = mtot_array[i] * MSOL
    mrat = mrat_array[i]
    redz = redz_array[i]
    
    print("{:.1e}".format(mtot/ MSOL))
    new_path_to_updated_txt_file = chain_path.parent / f'stats_analysis/freq_bin{i}_mtot_{(mtot/MSOL):.2e}_mrat_{mrat:.2f}_redz_{redz:.2f}/updated_chain_1_with_sigma_over_m.txt'

    load_path = "f'{PROJECT_ROOT}/holodeck_output_lib_gp_mcmc/sidm_lambda0_yvalues_for_cvalues_1_to_5_for_500_points.npz"
    data = np.load(load_path)
    C_values, Lambda0_higher_values, Lambda0_lower_values, y_higher_values, y_lower_values = data["a1"], data["a2"], data["a3"], data["a4"], data["a5"]
    interval_high = np.argmax(y_higher_values), np.where(y_higher_values>0)[0][-1] # Indices of the highest positive elements, and the last positive element

    C_of_y_higher_values = interp1d(y_higher_values[interval_high[0] + 30: interval_high[1] + 1], C_values[interval_high[0] + 30 : interval_high[1] + 1], fill_value="extrapolate")
    subintervals = np.linspace(1e-10, 6, 100)

    interval_rt = (1e-11 * PC, 1e10 * PC) # r values chosen by observation of Fig.2 (seeing approximately between what values does r_t occur)
    subintervals_rt = np.linspace(interval_rt[0], interval_rt[1], 500)


    # Define the differential equation system
    def ode(w, vector, C):
        Lambda, dLambda_dw = vector
        d2Lambda_dw2 = -C * np.exp(Lambda) - 2 * dLambda_dw / w
        return [dLambda_dw, d2Lambda_dw2]

    rgw = 0.1 * PC # above eq.6 ACD
    rstar = 10 * PC # below eq.6 ACD
    from scipy.special import erf
    u_1 = 11/4 * mrat**(3/2) * (1 + mrat)**(-3/2) # ACD eq.C3
    N1 = erf(u_1 / np.sqrt(2)) - np.sqrt(2 / np.pi) * u_1 * np.exp(-u_1**2 / 2) # ACD eq.C2
    u_2 = 11/4 * (1 + mrat)**(-3/2) # ACD eq.C3
    N2 = erf(u_2 / np.sqrt(2)) - np.sqrt(2 / np.pi) * u_2 * np.exp(-u_2**2 / 2) # ACD eq.C2


    def compute_sigma0_over_m_and_t_df(mmb_mamp_log10, sigma0_over_m_times_t_age_by_1Gyr, vt_kms):
        
        Chen_2019_instance = holo.host_relations.MMBulge_Chen2019(mamp_log10=mmb_mamp_log10, mplaw=1.1, scatter_dex=None)
        mstar = Chen_2019_instance.mstar_from_mbh(mtot, redz)
        Girelli_2020_instance = holo.host_relations.Girelli_2020()
        mhalo = Girelli_2020_instance.halo_mass(np.array([mstar]), np.array([redz]))[0][0]
        rho_s, rs = NFW._nfw_rho_rad(mhalo, redz)
        def v0_y_higher_values(y): # using eq. B4# v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_lower_values(y)) * NWTG * rho_s * r_s**2 # in cgs (cm/s)
            v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_higher_values(y)) * NWTG * rho_s * rs**2 # in cgs (cm/s)
            return np.sqrt(v0_squared)
        
        vt = vt_kms * 1e5 # cm/s
        # v0, rsp are calculated for a=0 case
        ###################################################################
        ### sigma, vt dependent parts below ###############################
        def eqn2(y):
            term_1 = sigma0_over_m_times_t_age_by_1Gyr * GYR * v0_y_higher_values(y) * rho_s / (y * (1 + y)**2) # t_age is expressed in Gyrs
            return term_1 - 1
        count = 0 # To see how many times the equation becomes zero
        for k in range(len(subintervals) - 1):
            lower, upper = subintervals[k], subintervals[k + 1]
            if eqn2(lower) * eqn2(upper) < 0:  # Sign change indicates a root in (a, b)
                result = root_scalar(eqn2, bracket=(lower, upper), method='bisect')
                if result.converged:
                    y = result.root
                    r1 = y * rs
                    v0 = v0_y_higher_values(y)
                count = count + 1
                break
        if count == 0:
            y = np.max(y_higher_values)
            r1 = y * rs
            v0 = v0_y_higher_values(y)

        if(v0<vt):
            # explanation for changing the interpolation indices by 30 is given above
            Lambda0_of_y_higher_values = interp1d(y_higher_values[interval_high[0] + 30: interval_high[1] + 1], Lambda0_higher_values[interval_high[0] + 30 : interval_high[1] + 1], fill_value="extrapolate")
            Lambda0_higher = Lambda0_of_y_higher_values(r1 / rs)
            rho_c = rho_s / ((r1 / rs) * (1 + r1 / rs)**2)
            C = 4 * np.pi * NWTG * rho_c * r1**2 / v0**2

            solution = solve_ivp(ode, [1e-8, 1], [Lambda0_higher, 0], args=(C,), t_eval=np.linspace(1e-8, 1.0, 100))
            Lambda_values = solution.y[0]
            rho_core_values = rho_c * np.exp(Lambda_values) # cgs

            rhosp = rho_core_values[0] # cgs
            rsp = NWTG * mtot / v0**2
            m1 = mtot / (1 + mrat)
            tsp = (rsp**3 / (NWTG * m1))**0.5
            gamma = 3.0 / 4.0
            f_q_gamma = 96 * np.pi * mrat * ((1 + mrat) / 2)**(gamma + 1/2) * (N2 + N1*mrat**(-3-gamma))
            B = f_q_gamma * rhosp * rsp**3 / m1
            # for rt
            def f(r): # This is what we want to become zero at r=r_t
                return v_r(r) - vt
            def v_r(r): # Equation B6
                return v0 * (7/11 + 4/11 * (rsp / r)**0.5) # v_0 in cm/s , r_sp in cm, so we also must input r in cm
            count = 0
            for k in range(len(subintervals_rt) - 1):
                a, b = subintervals_rt[k], subintervals_rt[k + 1]
                # print(f(a), f(b))
                if f(a) * f(b) < 0:  # Sign change indicates a root in (a, b)
                    result = root_scalar(f, bracket=(a, b), method='bisect')
                    if result.converged:
                        rt = result.root
                        count = count + 1
            if(count==0):
                print(f'v0={v0/1e5}, vt={vt/1e5}, rsp={rsp/PC}')
                rt = 1e4 * PC
            xt = rt / (2*rsp)
            xgw = rgw / (2*rsp)
            xstar = rstar / (2*rsp)
            if(rt>rstar):
                tdf = 2 * tsp/(B*xt) * (xstar**(1/4) - xgw**(1/4)) # ACD eq.B8
            elif(rt<rgw):
                tdf = 4*tsp / (3*B) * (xgw**(-3/4) - xstar**(1/4)) # ACD eq.B7
            else:
                tdf = 2 * tsp/B * (5/3*xt**(-3/4) - xgw**(1/4)/xt - 2/3*xstar**(-3/4)) # ACD eq.B9
            
            sigma0_over_m = sigma0_over_m_times_t_age_by_1Gyr * GYR / tdf # cgs
            sigma_over_m = sigma0_over_m # ACD eq.B5
        
        else:
            def eqn2(y):
                # sigma * v = sigma0 / (1 + (v0/vt)**4) * v0

                # term_1 = (vt / v0_y_higher_values(y))**4 * sigma0_over_m_times_t_age_by_1Gyr * GYR * v0_y_higher_values(y) * rho_s / (y * (1 + y)**2) # t_age is expressed in Gyrs
                # same as above, just rewriting so that v0_y_higher_values isn't computed multiple times
                term_1 = vt * (vt / v0_y_higher_values(y))**3 * sigma0_over_m_times_t_age_by_1Gyr * GYR * rho_s / (y * (1 + y)**2) # t_age is expressed in Gyrs
                
                return term_1 - 1
            count = 0 # To see how many times the equation becomes zero
            for k in range(len(subintervals) - 1):
                lower, upper = subintervals[k], subintervals[k + 1]
                if eqn2(lower) * eqn2(upper) < 0:  # Sign change indicates a root in (a, b)
                    result = root_scalar(eqn2, bracket=(lower, upper), method='bisect')
                    if result.converged:
                        y = result.root
                        r1 = y * rs
                        v0 = v0_y_higher_values(y)
                    count = count + 1
                    break
            if count == 0:
                # print('count zero')
                y = np.max(y_higher_values)
                r1 = y * rs
                v0 = v0_y_higher_values(y)
            # explanation for changing the interpolation indices by 30 is given above
            Lambda0_of_y_higher_values = interp1d(y_higher_values[interval_high[0] + 30: interval_high[1] + 1], Lambda0_higher_values[interval_high[0] + 30 : interval_high[1] + 1], fill_value="extrapolate")
            Lambda0_higher = Lambda0_of_y_higher_values(r1 / rs)
            rho_c = rho_s / ((r1 / rs) * (1 + r1 / rs)**2)
            C = 4 * np.pi * NWTG * rho_c * r1**2 / v0**2
            solution = solve_ivp(ode, [1e-8, 1], [Lambda0_higher, 0], args=(C,), t_eval=np.linspace(1e-8, 1.0, 100))
            Lambda_values = solution.y[0]
            rho_core_values = rho_c * np.exp(Lambda_values) # cgs
            rhosp = rho_core_values[0] # cgs
            rsp = NWTG * mtot / v0**2
            m1 = mtot / (1 + mrat)
            tsp = (rsp**3 / (NWTG * m1))**0.5
            gamma = 7.0 / 4.0
            f_q_gamma = 96 * np.pi * mrat * ((1 + mrat) / 2)**(gamma + 1/2) * (N2 + N1*mrat**(-3-gamma))
            B = f_q_gamma * rhosp * rsp**3 / m1
            p = 5/2 - gamma
            xgw = rgw / (2*rsp)
            xstar = rstar / (2*rsp)
            tdf = tsp/B * (xgw**(-p+1) - xstar**(-p+1)) / (p-1) # inspired from ACD eq.B7
            sigma0_over_m = sigma0_over_m_times_t_age_by_1Gyr * GYR / tdf # cgs
            sigma_over_m = sigma0_over_m * (vt / v0)**4 # ACD eq.B5
        return sigma0_over_m, tdf, sigma_over_m, y, v0 # in seconds



    # to use pandas

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from multiprocessing import Pool


    # Load data into a Pandas DataFrame


    df = pd.read_csv(chain_path, delim_whitespace=True, header=None)

    # Assign column names
    df.columns = ["psi0", "gsmf_mchar0_log10", "mmb_mamp_log10", "mmb_scatter_dex", "vt", "sigma0_over_m_times_t_age_by_1Gyr",
                  "log_likelihood_freq_bin0", "log_likelihood_freq_bin1", "log_likelihood_freq_bin2", "log_likelihood_freq_bin3", "log_likelihood_freq_bin4",
                  "log_posterior", "log_likelihood", "last_2_columns_1", "last_2_columns_2"]

    # Function wrapper for parallel execution
    def compute_parallel(args):
        mmb_mamp_log10, sigma0_over_m_times_t_age_by_1Gyr, vt = args
        try:
            return compute_sigma0_over_m_and_t_df(mmb_mamp_log10, sigma0_over_m_times_t_age_by_1Gyr, vt)
        except Exception as e:
            print(f"Failed for inputs: mmb_mamp_log10={mmb_mamp_log10}, sigma0_over_m_times_t_age_by_1Gyr={sigma0_over_m_times_t_age_by_1Gyr}, vt={vt}")
            raise e

    # Prepare arguments
    data_tuples = list(zip(df["mmb_mamp_log10"], df["sigma0_over_m_times_t_age_by_1Gyr"], df["vt"]))


    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(compute_parallel, data_tuples), total=len(data_tuples), desc="Computing tdf"))

    # Convert results back to DataFrame
    df[["sigma0_over_m", "tdf", "sigma_over_m", "y", "v0"]] = pd.DataFrame(results, index=df.index)



    column_order = ["psi0", "gsmf_mchar0_log10", "mmb_mamp_log10", "mmb_scatter_dex", "vt", "sigma0_over_m_times_t_age_by_1Gyr",
                    "sigma0_over_m", "tdf", "sigma_over_m", "y", "v0",
                    "log_likelihood_freq_bin0", "log_likelihood_freq_bin1", "log_likelihood_freq_bin2", "log_likelihood_freq_bin3", "log_likelihood_freq_bin4",
                    "log_posterior", "log_likelihood", "last_2_columns_1", "last_2_columns_2"]

    new_chain_data = df[column_order].to_numpy()  # Ensure correct order


    # Create the directory if it doesn't exist
    Path(new_path_to_updated_txt_file).parent.mkdir(parents=True, exist_ok=True) # asked chatgpt

    np.savetxt(new_path_to_updated_txt_file, new_chain_data, delimiter=" ", fmt="%.18e")




