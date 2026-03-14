"""

$ python setup.py build_ext -i

$ python setup.py develop

"""

import cython
import numpy as np
cimport numpy as np
np.import_array()

from scipy.optimize.cython_optimize cimport brentq

from libc.stdio cimport printf, fflush, stdout
from libc.stdlib cimport malloc, free
# make sure to use c-native math functions instead of python/numpy
from libc.math cimport pow, sqrt, M_PI, NAN, log10, sin, cos

import holodeck as holo
from holodeck.cyutils cimport interp_at_index, _interp_between_vals


# ---- Define Parameters


# ---- Define Constants

cdef double MY_NWTG = 6.6742999e-08
cdef double MY_SPLC = 29979245800.0
cdef double MY_MPC = 3.08567758e+24
cdef double MY_MSOL = 1.988409870698051e+33
cdef double MY_YR = 31557600.0
cdef double MY_SCHW = 1.4852320538237328e-28     #: Schwarzschild Constant  2*G/c^2  [cm]
cdef double GW_DADT_SEP_CONST = - 64.0 * pow(MY_NWTG, 3) / 5.0 / pow(MY_SPLC, 5)

cdef double MY_PC = MY_MPC / 1.0e6
cdef double MY_GYR = MY_YR * 1.0e9
cdef double KEPLER_CONST_FREQ = (1.0 / (2.0*M_PI)) * sqrt(MY_NWTG)
cdef double KEPLER_CONST_SEPA = pow(MY_NWTG, 1.0/3.0) / pow(2.0*M_PI, 2.0/3.0)
cdef double FOUR_PI_SPLC_OVER_MPC = 4 * M_PI * MY_SPLC / MY_MPC


@cython.cdivision(True)
cpdef double hard_gw(double mtot, double mrat, double sepa):
# cdef double hard_gw(double mtot, double mrat, double sepa):
    cdef double dadt = GW_DADT_SEP_CONST * pow(mtot, 3) * mrat / pow(sepa, 3) / pow(1 + mrat, 2)
    return dadt


@cython.cdivision(True)
cdef double kepler_freq_from_sepa(double mtot, double sepa):
    cdef double freq = KEPLER_CONST_FREQ * sqrt(mtot) / pow(sepa, 1.5)
    return freq


@cython.cdivision(True)
cdef double kepler_sepa_from_freq(double mtot, double freq):
    cdef double sepa = KEPLER_CONST_SEPA * pow(mtot, 1.0/3.0) / pow(freq, 2.0/3.0)
    return sepa


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int while_while_increasing(int start, int size, double val, double[:] edges):
    """Step through an INCREASING array of `edges`, first forward, then backward, to find edges bounding `val`.

    Use this function when `start` is already a close guess, and we just need to update a bit.

    """

    cdef int index = start    #: index corresponding to the LEFT side of the edges bounding `val`

    # `index < size-2` so that the result is always within the edges array
    # `edges[index+1] < val` to get the RIGHT-edge to be MORE than `val`
    while (index < size - 2) and (edges[index+1] < val):
        index += 1

    # `edges[index] > val` to get the LEFT-edge to be LESS than `val`
    while (index > 0) and (edges[index] > val):
        index -= 1

    return index


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int while_while_decreasing(int start, int size, double val, double[:] edges):
    """Step through a DECREASING array of `edges`, first forward, then backward, to find edges bounding `val`.

    Use this function when `start` is already a close guess, and we just need to update a bit.

    """

    cdef int index = start    #: index corresponding to the LEFT side of the edges bounding `val`

    # `index < size-1` so that the result is always within the edges array
    # `edges[index+1] > val` to get the RIGHT-edge to be LESS than `val`
    while (index < size - 1) and (edges[index+1] > val):
        index += 1

    # `edges[index-1] < val` to get the LEFT-edge to be MORE than `val`
    while (index > 0) and (edges[index-1] < val):
        index -= 1

    return index


# ==================================================================================================
# ====    Integrate Bins from differential-parameter-volume to total numbers   ====
# ==================================================================================================


def integrate_differential_number_3dx1d(edges, dnum):
    """Integrate the differential number of binaries over each grid bin into total numbers of binaries.

    Trapezoid used over first 3 dims (mtot, mrat, redz), and Riemann over 4th (freq).
    (Riemann seemed empirically to be more accurate for freq, but this should be revisited.)
    mtot is integrated over `log10(mtot)` and frequency is integrated over `ln(f)`.

    Note on array shapes:

    * input  `dnum` is shaped (M, Q, Z, F)
    * input  `edges` must be (4,) of array_like of lengths:  M, Q, Z, F+1
    * output `numb` is shaped (M-1, Q-1, Z-1, F)

    Arguments
    ---------
    edges : (4,) array_like  w/ lengths M, Q, Z, F+1
        Grid edges of `mtot`, `mrat`, `redz`, and `freq`.  NOTE:

        * `mtot` should be passed as regular `mtot`, NOT log10(mtot)
        * `freq` should be passed as regular `freq`, NOT    ln(freq)

    dnum : (M, Q, Z, F)
        Differential number of binaries, dN/[dlog10M dq qz dlnf] where 'N' is in units of dimensionless number.

    Returns
    -------
    numb : (M-1, Q-1, Z-1, F)

    """

    # each edge should have the same length as the corresponding dimension of `dnum`
    shape = [len(ee) for ee in edges]
    err = f"Shape of edges={shape} does not match dnum={np.shape(dnum)}"
    # except the last edge (freq), where `dnum` should be 1-shorter
    shape[-1] -= 1
    assert np.shape(dnum) == tuple(shape), err
    # the number will be shaped as one-less the size of each dimension of `dnum`
    new_shape = [sh-1 for sh in dnum.shape]
    # except for the last dimension (freq) which is the same shape
    new_shape[-1] = dnum.shape[-1]

    # prepare output array
    cdef np.ndarray[np.double_t, ndim=4] numb = np.zeros(new_shape)
    # Convert from  mtot => log10(mtot)  and  freq ==> ln(freq)
    ee = [np.log10(edges[0]), edges[1], edges[2], np.diff(np.log(edges[3]))]
    # integrate
    _integrate_differential_number_3dx1d(ee[0], ee[1], ee[2], ee[3], dnum, numb)

    return numb


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _integrate_differential_number_3dx1d(
    double[:] log10_mtot,
    double[:] mrat,
    double[:] redz,
    double[:] dln_freq,    # actually ln(freq)
    double[:, :, :, :] dnum,
    # output
    double[:, :, :, :] numb
):
    """Integrate the differential number of binaries over each grid bin into total numbers of binaries.

    Trapezoid used over first 3 dims (mtot, mrat, redz), and Riemann over 4th (freq).
    See docstrings in `integrate_differential_number_3dx1d`

    """

    cdef int n_mtot = log10_mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = dln_freq.size

    cdef int mm, qq, zz, ff, ii, jj, kk
    cdef double dm, dmdq, dmdqdz

    for mm in range(n_mtot-1):                              # iterate over output-shape of mass-grid
        dm = log10_mtot[mm+1] - log10_mtot[mm]              # get the bin-length

        for qq in range(n_mrat-1):                          # iterate over output-shape of mass-ratio-grid
            dmdq = dm * (mrat[qq+1] - mrat[qq])             # get the bin-area

            for zz in range(n_redz-1):                      # iterate over output-shape of redsz-grid
                dmdqdz = dmdq * (redz[zz+1] - redz[zz])     # get the bin-volume

                # iterate over output-shape of frequency
                # note that this is a Riemann sum, so input and output dimensions are the same size
                for ff in range(n_freq):
                    temp = 0.0

                    # iterate over each vertex of this cube, averaging the contributions
                    for ii in range(2):                     # mass vertices
                        for jj in range(2):                 # mass-ratio vertices
                            for kk in range(2):             # redshift vertices
                                temp += dnum[mm+ii, qq+jj, zz+kk, ff]

                    numb[mm, qq, zz, ff] = temp * dmdqdz * dln_freq[ff] / 8.0

    return


# ==================================================================================================
# ====    Fixed_Time_2pwl_SAM - Hardening Model    ====
# ==================================================================================================


# @cython.cdivision(True)
# cpdef double _hard_func_2pwl(double norm, double xx, double gamma_inner, double gamma_outer):
#     cdef double dadt = - norm * pow(1.0 + xx, -gamma_outer+gamma_inner) / pow(xx, gamma_inner-1)
#     return dadt


# @cython.cdivision(True)
# cpdef double hard_func_2pwl_gw(
#     double mtot, double mrat, double sepa,
#     double norm, double rchar, double gamma_inner, double gamma_outer
# ):
#     cdef double dadt = _hard_func_2pwl(norm, sepa/rchar, gamma_inner, gamma_outer)
#     dadt += hard_gw(mtot, mrat, sepa)
#     return dadt


@cython.cdivision(True)
cdef double _hard_func_2pwl(double norm, double xx, double gamma_inner, double gamma_outer):
    cdef double dadt = - norm * pow(1.0 + xx, -gamma_outer+gamma_inner) / pow(xx, gamma_inner-1)
    return dadt


@cython.cdivision(True)
cdef double _hard_func_2pwl_gw(
    double mtot, double mrat, double sepa,
    double norm, double rchar, double gamma_inner, double gamma_outer
):
    cdef double dadt = _hard_func_2pwl(norm, sepa/rchar, gamma_inner, gamma_outer)
    dadt += hard_gw(mtot, mrat, sepa)
    return dadt


@cython.cdivision(True)
cdef double[:] _hard_func_2pwl_gw_1darray(
    double[:] mtot, double[:] mrat, double[:] sepa,
    double[:] norm, double[:] rchar, double[:] gamma_inner, double[:] gamma_outer
):
    cdef int ii
    cdef int size = mtot.size
    cdef np.ndarray[np.double_t, ndim=1] dadt = np.zeros(size)
    for ii in range(size):
        dadt[ii] = _hard_func_2pwl(norm[ii], sepa[ii]/rchar[ii], gamma_inner[ii], gamma_outer[ii])
        dadt[ii] += hard_gw(mtot[ii], mrat[ii], sepa[ii])

    return dadt

#### dadt for CDM ###

@cython.cdivision(True)
cdef double density_nfw(double sepa, double rho_s, double rs):
    cdef double dens_nfw
    dens_nfw = rho_s / (sepa / rs) / pow(1 + sepa / rs, 2)
    return dens_nfw

@cython.cdivision(True)
cdef double density_cdm(double mtot, double sepa, double rho_s, double rs, double gamma_sp):
    cdef double r_sp, rho_sp0, dens
    r_sp = 0.2 * sqrt(mtot / (M_PI * rho_s * rs))
    if (sepa / r_sp < 1):
        rho_sp0 = density_nfw(r_sp, rho_s, rs)
        dens = rho_sp0 * pow(r_sp / sepa, gamma_sp)
    else:
        dens = density_nfw(sepa, rho_s, rs)
    return dens

@cython.cdivision(True)
cdef double _hard_func_cdm(double mtot, double mrat, double sepa, double rho_s, double rs, double gamma_sp):
    cdef double sepa1 = mrat * sepa / (1 + mrat)
    cdef double sepa2 = sepa / (1 + mrat)
    cdef double p_gw = 32.0/5.0 * pow(MY_NWTG / (1 + mrat), 4) * pow(mrat, 2) * pow(mtot / MY_SPLC / sepa, 5)
    cdef double p_df = (12.0 * M_PI * pow(mrat, 2) / (1 + mrat) 
            * pow(MY_NWTG * mtot, 3.0/2.0) * sqrt(sepa)
            * (density_cdm(mtot, sepa1, rho_s, rs, gamma_sp) / pow(mrat, 3) + density_cdm(mtot, sepa2, rho_s, rs, gamma_sp))
            )
    cdef double dadt = - 2.0 / MY_NWTG / mrat * pow(sepa * (1 + mrat) / mtot, 2) * (p_gw + p_df)
    """

    NOTE: Negative sign, because that's how holodeck does it in phenom case too.
    This dadt is later on multiplied by -2/3 to get a positive value

    """
    return dadt


#### dadt for SIDM ###

@cython.cdivision(True)
cdef double density_core(double sepa, int n_rvals, double[:] r_values, double[:] rho_values):
    cdef double dens
    cdef int interp_idx
    interp_idx = while_while_decreasing(0, n_rvals, sepa, r_values)
    dens = interp_at_index(interp_idx, sepa, r_values, rho_values)
    return dens

@cython.cdivision(True)
cdef double density_spike(double sepa, double rt, double r_sp, double gamma1, double gamma2, double rho0):
    cdef double dens
    if (sepa > rt):
        dens = rho0 * pow((r_sp / sepa), gamma1)
    else:
        dens = rho0 * pow((r_sp / sepa), gamma2)
    return dens

@cython.cdivision(True)
cdef double density_sidm(double sepa, int n_rvals, double[:] r_values, double[:] rho_values, double rt, double r_sp, double gamma1, double gamma2):
    cdef double dens
    dens = density_core(sepa, n_rvals, r_values, rho_values) + density_spike(sepa, rt, r_sp, gamma1, gamma2, rho_values[0])
    return dens

@cython.cdivision(True)
cdef double _hard_func_sidm(double mtot, double mrat, double sepa, int n_rvals, 
                            double[:] r_values_a0, double[:] rho_values_a0, double[:] r_values_a4, double[:] rho_values_a4,
                            double v0, double vt, double rt, double r_sp_a4, double n1, double n2,
                            double gamma1, double gamma2,
):  
    cdef double p_df
    cdef double sepa1 = mrat * sepa / (1 + mrat)
    cdef double sepa2 = sepa / (1 + mrat)
    cdef double p_gw = 32.0/5.0 * pow(MY_NWTG / (1 + mrat), 4) * pow(mrat, 2) * pow(mtot / MY_SPLC / sepa, 5)
    if (v0 > vt): # only a=4 spike
        p_df = (12.0 * M_PI * pow(mrat, 2) / (1 + mrat) 
                * pow(MY_NWTG * mtot, 3.0/2.0) * sqrt(sepa)
                * (n1 * density_sidm(sepa1, n_rvals, r_values_a4, rho_values_a4, 0, r_sp_a4, gamma2, gamma2) / pow(mrat, 3) + n2 * density_sidm(sepa2, n_rvals, r_values_a4, rho_values_a4, 0, r_sp_a4, gamma2, gamma2))
                ) # setting rt to zero because we don't want any transition happening in this case
    else: # a0 to a4 transition at rt
        p_df = (12.0 * M_PI * pow(mrat, 2) / (1 + mrat) 
                * pow(MY_NWTG * mtot, 3.0/2.0) * sqrt(sepa)
                * (n1 * density_sidm(sepa1, n_rvals, r_values_a0, rho_values_a0, rt, rt, gamma1, gamma2) / pow(mrat, 3) + n2 * density_sidm(sepa2, n_rvals, r_values_a0, rho_values_a0, rt, rt, gamma1, gamma2))
                )
    cdef double dadt = - 2.0 / MY_NWTG / mrat * pow(sepa * (1 + mrat) / mtot, 2) * (p_gw + p_df)
    """

    NOTE: Negative sign, because that's how holodeck does it in phenom case too.
    This dadt is later on multiplied by -2/3 to get a positive value

    """
    return dadt

@cython.cdivision(True)
cdef double density_sidm_Version2(double sepa, int n_rvals, double[:] r_values, double[:] density_values):
    cdef double dens
    cdef int interp_idx
    # interp_idx = while_while_decreasing(0, n_rvals, sepa, r_values)
    ####### checking ##########
    interp_idx = while_while_increasing(0, n_rvals, sepa, r_values)
    ##########################
    dens = interp_at_index(interp_idx, sepa, r_values, density_values)
    return dens

@cython.cdivision(True)
cdef double _hard_func_sidm_gw_Version2(double mtot, double mrat, double sepa, int n_rvals, 
                            double[:] r_values, double[:] density_values,
                            double n1, double n2,
):  
    cdef double p_df
    cdef double sepa1 = mrat * sepa / (1 + mrat)
    cdef double sepa2 = sepa / (1 + mrat)
    cdef double p_gw = 32.0/5.0 * pow(MY_NWTG / (1 + mrat), 4) * pow(mrat, 2) * pow(mtot / MY_SPLC / sepa, 5)
    p_df = (12.0 * M_PI * pow(mrat, 2) / (1 + mrat) 
                * pow(MY_NWTG * mtot, 3.0/2.0) * sqrt(sepa)
                * (n1 * density_sidm_Version2(sepa1, n_rvals, r_values, density_values) / pow(mrat, 3) + n2 * density_sidm_Version2(sepa2, n_rvals, r_values, density_values))
                )
    cdef double dadt = - 2.0 / MY_NWTG / mrat * pow(sepa * (1 + mrat) / mtot, 2) * (p_gw + p_df)
    ################################################
    # dadt += hard_gw(mtot, mrat, sepa) # for testing
    ################################################
    """

    NOTE: Negative sign, because that's how holodeck does it in phenom case too.
    This dadt is later on multiplied by -2/3 to get a positive value

    Also, like other classes, dadt_gw is not added separately because it is included in p_gw

    """
    return dadt

#### dadt for 3 Body Scattering ###

@cython.cdivision(True)
cdef double _hard_func_3bs(double rho_i, double sigma_i, double sepa, double H,
):  
    
    cdef double dadt = - H * MY_NWTG * rho_i / sigma_i * pow(sepa, 2)
    """

    NOTE: Chen2024 eq.4
    """
    return dadt

@cython.cdivision(True)
cdef double _hard_func_cdm_gw(
    double mtot, double mrat, double sepa,
    double rho_s, double rs, double gamma_sp,
):
    cdef double dadt = _hard_func_cdm(mtot, mrat, sepa, rho_s, rs, gamma_sp)
    dadt += hard_gw(mtot, mrat, sepa)
    return dadt

@cython.cdivision(True)
cdef double _hard_func_sidm_gw(
                            double mtot, double mrat, double sepa, int n_rvals, 
                            double[:] r_values_a0, double[:] rho_values_a0, double[:] r_values_a4, double[:] rho_values_a4,
                            double v0, double vt, double rt, double r_sp_a4, double n1, double n2,
                            double gamma1, double gamma2,
):
    cdef double dadt = _hard_func_sidm(mtot, mrat, sepa, n_rvals, r_values_a0, rho_values_a0, r_values_a4, rho_values_a4, v0, vt, rt, r_sp_a4, n1, n2, gamma1, gamma2)
    dadt += hard_gw(mtot, mrat, sepa)
    return dadt

# cython.cdivision(True) # commented out becaue was giving an error while cythonizing otherwise
cdef double _hard_func_3bs_gw(
                            double mtot, double mrat, double sepa, 
                            double rho_i, double sigma_i, double H,
):
    cdef double dadt = _hard_func_3bs(rho_i, sigma_i, sepa, H)
    dadt += hard_gw(mtot, mrat, sepa)
    return dadt

def hard_func_2pwl_gw(
    mtot, mrat, sepa,
    norm, rchar, gamma_inner, gamma_outer
):
    """

    NOTE: this function will be somewhat slow, because of the explicit broadcasting!

    """
    args = mtot, mrat, sepa, norm, rchar, gamma_inner, gamma_outer
    args = np.broadcast_arrays(*args)
    shape = args[0].shape
    mtot, mrat, sepa, norm, rchar, gamma_inner, gamma_outer = [aa.flatten() for aa in args]
    dadt = _hard_func_2pwl_gw_1darray(mtot, mrat, sepa, norm, rchar, gamma_inner, gamma_outer)
    dadt = np.array(dadt).reshape(shape)
    return dadt


def find_2pwl_hardening_norm(time, mtot, mrat, sepa_init, rchar, gamma_inner, gamma_outer, nsteps):
    assert np.ndim(time) == 0
    assert np.ndim(mtot) == 1
    assert np.shape(mtot) == np.shape(mrat)

    cdef np.ndarray[np.double_t, ndim=1] norm_log10 = np.zeros(mtot.size)

    cdef lifetime_2pwl_params args
    args.target_time = time
    args.sepa_init = sepa_init
    args.rchar = rchar
    args.gamma_inner = gamma_inner
    args.gamma_outer = gamma_outer
    args.nsteps = nsteps

    _get_hardening_norm_2pwl(mtot, mrat, args, norm_log10)

    return norm_log10


ctypedef struct lifetime_2pwl_params:
    double target_time
    double mt
    double mr
    double sepa_init
    double rchar
    double gamma_inner
    double gamma_outer
    int nsteps


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _get_hardening_norm_2pwl(
    double[:] mtot,
    double[:] mrat,
    lifetime_2pwl_params args,
    # output
    double[:] norm_log10,
):

    cdef double XTOL = 1e-3
    cdef double RTOL = 1e-5
    cdef int MITR = 100    # note: the function doesn't return an error on failure, it still returns last try
    cdef double NORM_LOG10_LO = -20.0
    cdef double NORM_LOG10_HI = +20.0

    cdef int num = mtot.size
    assert mtot.size == mrat.size
    cdef double time

    cdef int ii
    for ii in range(num):
        args.mt = mtot[ii]
        args.mr = mrat[ii]
        norm_log10[ii] = brentq(
            get_binary_lifetime_2pwl, NORM_LOG10_LO, NORM_LOG10_HI,
            <lifetime_2pwl_params *> &args, XTOL, RTOL, MITR, NULL
        )
        # time = get_binary_lifetime_2pwl(norm_log10[ii], <lifetime_2pwl_params *> &args)
        # total_time[ii] = time + args.target_time

    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double get_binary_lifetime_2pwl(double norm_log10, void *args) noexcept:
    cdef lifetime_2pwl_params *pars = <lifetime_2pwl_params *> args

    cdef double risco_log10 = log10(3.0 * MY_SCHW * pars.mt)
    cdef double sepa_log10 = log10(pars.sepa_init)
    cdef double norm = pow(10.0, norm_log10)

    # step-size, in log10-space, to go from sepa_init to ISCO
    cdef double dx = (sepa_log10 - risco_log10) / pars.nsteps
    cdef double time = 0.0

    cdef int ii
    cdef double sepa_right, dadt_right, dt

    cdef double sepa_left = pow(10.0, sepa_log10)
    cdef double dadt_left = _hard_func_2pwl_gw(
        pars.mt, pars.mr, sepa_left,
        norm, pars.rchar, pars.gamma_inner, pars.gamma_outer
    )

    for ii in range(pars.nsteps):
        sepa_log10 -= dx
        sepa_right = pow(10.0, sepa_log10)

        # Get total hardening rate at k+1 edge
        dadt_right = _hard_func_2pwl_gw(
            pars.mt, pars.mr, sepa_right,
            norm, pars.rchar, pars.gamma_inner, pars.gamma_outer
        )

        # Find time to move from left to right
        dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
        time += dt

        sepa_left = sepa_right
        dadt_left = dadt_right

    time = time - pars.target_time
    return time


def integrate_binary_evolution_2pwl(norm_log10, mtot, mrat, sepa_init, rchar, gamma_inner, gamma_outer, nsteps):
    cdef lifetime_2pwl_params args
    args.mt = mtot
    args.mr = mrat
    args.target_time = 0.0
    args.sepa_init = sepa_init
    args.rchar = rchar
    args.gamma_inner = gamma_inner
    args.gamma_outer = gamma_outer
    args.nsteps = nsteps

    time = get_binary_lifetime_2pwl(norm_log10, <lifetime_2pwl_params *> &args)
    return time


# ==================================================================================================
# ====    Dynamic Binary Number - calculate number of binaries at each frequency    ====
# ==================================================================================================


def dynamic_binary_number_at_fobs(fobs_orb, sam, hard, cosmo):
    """Calculate the differential number of binaries at the given frequencies.

    This function converts from differential binary volume-density to differential number of
    binaries.  The differential binary volume-density is:

    .. math::
        d^3 n / [d \log_{10} M  d q  d z]

    Where the number density is $n = d N/d V_c$ for a comoving volume $V_c$.  The differential
    binary number is:

    .. math::
        d^4 N / [d \log_{10} M  d q  d z  d \ln f]

    Arguments
    ---------
    fobs_orb : (F,) array of float, [1/s]
        The observer-frame orbital frequencies of interest, in units of inverse seconds.
    sam : :py:class:`holodeck.sams.sam.Semi_Analytic_Model` instance
        The semi-analytic model population.
    hard : :py:class:`holodeck.hardening._Hardening` subclass instance,
        The binary evolution model to evolve binaries from galaxy merger to the given frequencies.
    cosmo : :py:class:`astropy.cosmology.core.Cosmology` instance
        Cosmology object used for calculating cosmological measurements.

    Returns
    -------
    redz_final : (M, Q, Z, F) array of float, []
        The redshifts at which binaries at each grid point reach the frequencies of interest.
        Unitless.
    diff_num : (M, Q, Z, F) array of float, []
        The differential number of binaries at each grid point.  Unitless.

    """

    nden = sam.static_binary_density

    shape = sam.shape + (fobs_orb.size,)
    cdef np.ndarray[np.double_t, ndim=4] diff_num = np.zeros(shape)
    cdef np.ndarray[np.double_t, ndim=4] redz_final = -1.0 * np.ones(shape)

    # ---- Fixed_Time_2pwl_SAM

    if isinstance(hard, holo.hardening.Fixed_Time_2PL_SAM):
        gmt_time = sam._gmt_time
        # if `sam` is using galaxy merger rate (GMR), then `gmt_time` will be `None`
        if gmt_time is None:
            sam._log.info("`gmt_time` not calculated in SAM.  Setting to zeros.")
            gmt_time = np.zeros(sam.shape)

        _dynamic_binary_number_at_fobs_2pwl(
            fobs_orb, hard._sepa_init, hard._num_steps,
            hard._norm, hard._rchar, hard._gamma_inner, hard._gamma_outer,
            nden, sam.mtot, sam.mrat, sam.redz, gmt_time,
            cosmo._grid_z, cosmo._grid_dcom, cosmo._grid_age,
            # output:
            redz_final, diff_num
        )

    # ---- Hard_GW

    # elif isinstance(hard, holo.hardening.Hard_GW) or issubclass(hard, holo.hardening.Hard_GW):
    # commented out because issubclass() was giving an error
    elif isinstance(hard, holo.hardening.Hard_GW):
        redz_prime = sam._redz_prime
        # if `sam` doesn't use a galaxy merger time (GMT), then `redz_prime` will be `None`,
        # set to initial redshift values instead
        if redz_prime is None:
            sam._log.info("`redz_prime` not calculated in SAM.  Setting to `redz` (initial) values.")
            redz_prime = sam.redz[np.newaxis, np.newaxis, :] * np.ones(sam.shape)

        _dynamic_binary_number_at_fobs_gw(
            fobs_orb,
            nden, sam.mtot, sam.mrat, sam.redz, redz_prime,
            cosmo._grid_z, cosmo._grid_dcom,
            # output:
            redz_final, diff_num
        )

    # ---- Hard_CDM

    elif isinstance(hard, holo.hardening.Hard_CDM):
        gmt_time = sam._gmt_time
        # if `sam` is using galaxy merger rate (GMR), then `gmt_time` will be `None`
        if gmt_time is None:
            sam._log.info("`gmt_time` not calculated in SAM.  Setting to zeros.")
            gmt_time = np.zeros(sam.shape)

        _dynamic_binary_number_at_fobs_cdm(
            fobs_orb, hard._sepa_init, hard._num_steps,
            hard._gamma_sp,
            hard._rho_s_3d_array, hard._rs_3d_array,
            nden, sam.mtot, sam.mrat, sam.redz, gmt_time,
            cosmo._grid_z, cosmo._grid_dcom, cosmo._grid_age,
            # output:
            redz_final, diff_num
        )

    # ---- Hard_SIDM

    elif isinstance(hard, holo.hardening.Hard_SIDM):
        gmt_time = sam._gmt_time
        # if `sam` is using galaxy merger rate (GMR), then `gmt_time` will be `None`
        if gmt_time is None:
            sam._log.info("`gmt_time` not calculated in SAM.  Setting to zeros.")
            gmt_time = np.zeros(sam.shape)

        _dynamic_binary_number_at_fobs_sidm(
            fobs_orb, hard._sepa_init, hard._num_steps,
            nden, sam.mtot, sam.mrat, sam.redz, gmt_time,
            cosmo._grid_z, cosmo._grid_dcom, cosmo._grid_age,
            hard._r_values_a0, hard._rho_core_values_a0,
            hard._r_values_a4, hard._rho_core_values_a4,
            hard._v0_a0, hard._vt, hard._rt, hard._r_sp_a4,
            hard._N1_array, hard._N2_array,
            3.0/4.0, 7.0/4.0,
            # output:
            redz_final, diff_num
        )

    elif isinstance(hard, holo.hardening.Hard_SIDM_Version2):
        gmt_time = sam._gmt_time
        # if `sam` is using galaxy merger rate (GMR), then `gmt_time` will be `None`
        if gmt_time is None:
            sam._log.info("`gmt_time` not calculated in SAM.  Setting to zeros.")
            gmt_time = np.zeros(sam.shape)

        _dynamic_binary_number_at_fobs_sidm_Version2(
            fobs_orb, hard._sepa_init, hard._num_steps,
            nden, sam.mtot, sam.mrat, sam.redz, gmt_time,
            cosmo._grid_z, cosmo._grid_dcom, cosmo._grid_age,
            hard._r_values, hard._density_3d_array,
            hard._N1_array, hard._N2_array,
            # output:
            redz_final, diff_num
        )

    # ---- Hard_3BS

    elif isinstance(hard, holo.hardening.Hard_3BS):
        gmt_time = sam._gmt_time
        # if `sam` is using galaxy merger rate (GMR), then `gmt_time` will be `None`
        if gmt_time is None:
            sam._log.info("`gmt_time` not calculated in SAM.  Setting to zeros.")
            gmt_time = np.zeros(sam.shape)

        _dynamic_binary_number_at_fobs_3bs(
            fobs_orb, hard._sepa_init, hard._num_steps,
            hard._rho_i_1d_array, hard._sigma_i_1d_array, hard._H,
            nden, sam.mtot, sam.mrat, sam.redz, gmt_time,
            cosmo._grid_z, cosmo._grid_dcom, cosmo._grid_age,
            # output:
            redz_final, diff_num
        )
    

    # ---- OTHER

    else:
        raise ValueError(f"Unexpected `hard` value {hard}!")

    return redz_final, diff_num

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _dynamic_binary_number_at_fobs_2pwl(
    double[:] target_fobs_orb,
    double sepa_init,
    int num_steps,

    double[:, :] hard_norm,
    double hard_rchar,
    double hard_gamma_inner,
    double hard_gamma_outer,

    double[:, :, :] nden,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] gmt_time,

    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,
    double[:] tage_interp_grid,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
) except -1:
    """Calculate differential binary number at the given frequencies, with phenom 2pl evolution.

    This function converts from differential binary volume-density to differential binary number.
    Binary evolution follows the 'phenomenological' double power-law model implemented in the
    :py:func:`_hard_func_2pwl_gw`, which matches the implementation in
    :py:class:`Fixed_Time_2PL_SAM`.

    See :py:func:`dynamic_binary_number_at_fobs` for more information.

    Arguments
    ---------
    target_fobs_orb : (F,) array of float [1/s]
        The observer-frame orbital frequencies of interest, in units of inverse seconds.
    nden : (M, Q, Z) array of float [Mpc^{-3}]
        The differential binary volume-density in units of inverse-cubic comoving-Mpc.
    mtot : (M,) array of float [g]
        The edges of the total-mass grid dimension in units of grams.
    mrat : (Q,) array of float []
        The edges of the mass-ratio grid dimension.  Unitless.
    redz : (Z,) array of float []
        The edges of the redshift grid dimension.  Unitless.
    redz_prime : (M, Q, Z) array of float []
        The redshifts of binaries after galaxy merger, but before binary evolution to the
        frequencies of interest.  Unitless.
    redz_interp_grid : (Zi,) array of float, []
        The redshift values at which comoving distances are calculated; used for interpolation.
    dcom_interp_grid : (Zi,) array of float, [cm]
        The comoving-distance values at the ``redz_interp_grid`` redshifts, in units of centimeters,
        used for interpolation.

    Returns
    -------
    redz_final : (M, Q, Z, F) array of float, []
        The redshifts at which binaries at each grid point reach the frequencies of interest.
        Unitless.
    diff_num : (M, Q, Z, F) array of float, []
        The differential number of binaries at each grid point.  Unitless.

    """

    cdef int n_mtot = mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = target_fobs_orb.size
    cdef int n_interp = redz_interp_grid.size
    cdef double age_universe = tage_interp_grid[n_interp - 1]
    cdef double sepa_init_log10 = log10(sepa_init)

    cdef int ii, jj, kk, ff, step, interp_left_idx, interp_right_idx, new_interp_idx
    cdef double mt, mr, norm, risco, dx, new_redz, gmt, ftarget, target_frst_orb
    cdef double sepa_log10, sepa, sepa_left, sepa_right, dadt_left, dadt_right
    cdef double time_evo, redz_left, redz_right, time_left, time_right, new_time
    cdef double frst_orb_left, fobs_orb_left, frst_orb_right, fobs_orb_right

    # ---- Calculate ages corresponding to SAM `redz` grid

    cdef double *redz_age = <double *>malloc(n_redz * sizeof(double))     # (Z,) age of the universe in [sec]
    ii = 0
    cdef int rev
    for kk in range(n_redz):
        # iterate in reverse order to match with `redz_interp_grid` which is decreasing
        rev = n_redz - 1 - kk
        # get to the right index of the interpolation-grid
        while (redz_interp_grid[ii+1] > redz[rev]) and (ii < n_interp - 1):
            ii += 1

        # interpolate
        redz_age[rev] = interp_at_index(ii, redz[rev], redz_interp_grid, tage_interp_grid)

    # ---- calculate dynamic binary numbers for all SAM grid bins

    for ii in range(n_mtot):
        mt = mtot[ii]

        # Determine separation step-size, in log10-space, to integrate from sepa_init to ISCO
        risco = 3.0 * MY_SCHW * mt     # ISCO is 3x combined schwarzschild radius
        dx = (sepa_init_log10 - log10(risco)) / num_steps

        for jj in range(n_mrat):
            mr = mrat[jj]

            # Binary evolution is determined by M and q only
            # so integration is started for each of these bins
            sepa_log10 = sepa_init_log10                # set initial separation to initial value
            norm = hard_norm[ii, jj]                    # get hardening-rate normalization for this bin

            # Get total hardening rate at left-most edge
            sepa_left = pow(10.0, sepa_log10)
            dadt_left = _hard_func_2pwl_gw(
                mt, mr, sepa_left,
                norm, hard_rchar, hard_gamma_inner, hard_gamma_outer
            )

            # get rest-frame orbital frequency of binary at left edge
            frst_orb_left = kepler_freq_from_sepa(mt, sepa_left)

            # ---- Integrate of `num_steps` discrete intervals in binary separation from large to small

            time_evo = 0.0                  # track total binary evolution time
            interp_left_idx = 0                 # interpolation index, will be updated in each step
            for step in range(num_steps):
                # Increment the current separation
                sepa_log10 -= dx
                sepa_right = pow(10.0, sepa_log10)
                frst_orb_right = kepler_freq_from_sepa(mt, sepa_right)

                # Get total hardening rate at the right-edge of this step (left-edge already obtained)
                dadt_right = _hard_func_2pwl_gw(
                    mt, mr, sepa_right,
                    norm, hard_rchar, hard_gamma_inner, hard_gamma_outer
                )

                # Find time to move from left- to right- edges:  dt = da / (da/dt)
                # average da/dt on the left- and right- edges of the bin (i.e. trapezoid rule)
                dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
                # if ii == 8 and jj == 0:
                #     printf("cy %03d : %.2e ==> %.2e  ==  %.2e\n", step, sepa_left, sepa_right, dt)

                time_evo += dt

                # ---- Iterate over starting redshift bins

                for kk in range(n_redz-1, -1, -1):
                    # get the total time from each starting redshift, plus GMT time, plus evolution time to this step
                    gmt = gmt_time[ii, jj, kk]
                    time_right = time_evo + gmt + redz_age[kk]
                    # also get the evolution-time to the left edge
                    time_left = time_right - dt

                    # if we pass the age of the universe, this binary has stalled, no further redshifts will work
                    # NOTE: if `gmt_time` decreases faster than redshift bins increase the universe age,
                    #       then systems in later `redz` bins may no longer stall, so we still need to calculate them.
                    #       i.e. we can NOT use a `break` statement here, must use `continue` statement.
                    if time_left > age_universe:
                        continue

                    # find the redshift bins corresponding to left- and right- side of step
                    # left edge
                    interp_left_idx = while_while_increasing(interp_left_idx, n_interp, time_left, tage_interp_grid)

                    redz_left = interp_at_index(interp_left_idx, time_left, tage_interp_grid, redz_interp_grid)

                    # double check that left-edge is within age of Universe (should rarely if ever be a problem
                    # but possible due to rounding/interpolation errors
                    if redz_left < 0.0:
                        continue

                    # find right-edge starting from left edge, i.e. `interp_left_idx` (`interp_left_idx` is not a typo!)
                    interp_right_idx = while_while_increasing(interp_left_idx, n_interp, time_right, tage_interp_grid)
                    # NOTE: because `time_right` can be larger than age of universe, it can exceed `tage_interp_grid`
                    #       in this case `interp_right_idx=n_interp-2`, and the `interp_at_index` function can still
                    #       be used to extrapolate to further out values, which will likely be negative

                    redz_right = interp_at_index(interp_right_idx, time_right, tage_interp_grid, redz_interp_grid)
                    # NOTE: at this point `redz_right` could be negative, even though `redz_left` is definitely not
                    if redz_right < 0.0:
                        redz_right = 0.0

                    # if ii == 8 and jj == 0 and kk == 11:
                    #     printf("cy %03d : t=%.2e z=%.2e\n", step, time_right, redz_right)

                    # convert to frequencies
                    fobs_orb_left = frst_orb_left / (1.0 + redz_left)
                    fobs_orb_right = frst_orb_right / (1.0 + redz_right)

                    # ---- Iterate over all target frequencies

                    # NOTE: there should be a more efficient way to do this.
                    #       Tried a different implementation in `_dynamic_binary_number_at_fobs_1`, but not working
                    #       some of the frequency bins seem to be getting skipped in that version.

                    for ff in range(n_freq):
                        ftarget = target_fobs_orb[ff]

                        # If the integration-step does NOT bracket the target frequency, continue to next frequency
                        if (ftarget < fobs_orb_left) or (fobs_orb_right < ftarget):
                            continue

                        # ------------------------------------------------------
                        # ---- TARGET FOUND ----

                        # At this point in the code, this target frequency is inbetween the left- and right- edges
                        # of the integration step, so we can interpolate the evolution to exactly this frequency,
                        # and perform the actual dynamic_binary_number calculation

                        new_time = _interp_between_vals(ftarget, fobs_orb_left, fobs_orb_right, time_left, time_right)

                        # `time_right` can be after age of Universe, make sure interpolated value is not
                        #    if it is, then all higher-frequencies will also, so break out of target-frequency loop
                        if new_time > tage_interp_grid[n_interp - 1]:
                            break

                        # find index in interpolation grid for this exact time
                        new_interp_idx = interp_left_idx      # start from left-step edge
                        new_interp_idx = while_while_increasing(new_interp_idx, n_interp, new_time, tage_interp_grid)

                        # get redshift
                        new_redz = interp_at_index(new_interp_idx, new_time, tage_interp_grid, redz_interp_grid)
                        # get comoving distance
                        dcom = interp_at_index(new_interp_idx, new_time, tage_interp_grid, dcom_interp_grid)

                        # if (ii == 0) and (jj == 0) and (kk == 0):
                        #     printf("cy f=%03d (step=%03d)\n", ff, step)
                        #     printf(
                        #         "fl=%.6e, f=%.6e, fr=%.6e ==> tl=%.6e, t=%.6e, tr=%.6e\n",
                        #         fobs_orb_left, ftarget, fobs_orb_right,
                        #         time_left, new_time, time_right
                        #     )
                        #     printf(
                        #         "interp (%d) time: %.6e, %.6e, %.6e ==> z: %.6e, %.6e, %.6e\n",
                        #         new_interp_idx,
                        #         tage_interp_grid[new_interp_idx], new_time, tage_interp_grid[new_interp_idx+1],
                        #         redz_interp_grid[new_interp_idx], new_redz, redz_interp_grid[new_interp_idx+1],
                        #     )
                        #     printf("======> z=%.6e\n", new_redz)

                        # Store redshift
                        redz_final[ii, jj, kk, ff] = new_redz

                        # find rest-frame orbital frequency and binary separation
                        target_frst_orb = ftarget * (1.0 + new_redz)
                        sepa = kepler_sepa_from_freq(mt, target_frst_orb)

                        # calculate total hardening rate at this exact separation
                        dadt = _hard_func_2pwl_gw(
                            mt, mr, sepa,
                            norm, hard_rchar, hard_gamma_inner, hard_gamma_outer
                        )

                        # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                        tres = - (2.0/3.0) * sepa / dadt

                        # calculate number of binaries
                        cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + new_redz) * pow(dcom / MY_MPC, 2)
                        diff_num[ii, jj, kk, ff] = nden[ii, jj, kk] * tres * cosmo_fact

                        # ----------------------
                        # ------------------------------------------------------

                # update new left edge
                dadt_left = dadt_right
                sepa_left = sepa_right
                frst_orb_left = frst_orb_right
                # note that we _cannot_ do this for redz or freqs because the redshift _bin_ is changing

    free(redz_age)

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _dynamic_binary_number_at_fobs_gw(
    double[:] target_fobs_orb,

    double[:, :, :] nden,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] redz_prime,

    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
) except -1:
    """Calculate differential binary number at the given frequencies, under GW-only evolution.

    This function converts from differential binary volume-density to differential binary number.

    See :py:func:`dynamic_binary_number_at_fobs` for more information.

    Arguments
    ---------
    target_fobs_orb : (F,) array of float [1/s]
        The observer-frame orbital frequencies of interest, in units of inverse seconds.
    nden : (M, Q, Z) array of float [Mpc^{-3}]
        The differential binary volume-density in units of inverse-cubic comoving-Mpc.
    mtot : (M,) array of float [g]
        The edges of the total-mass grid dimension in units of grams.
    mrat : (Q,) array of float []
        The edges of the mass-ratio grid dimension.  Unitless.
    redz : (Z,) array of float []
        The edges of the redshift grid dimension.  Unitless.
    redz_prime : (M, Q, Z) array of float []
        The redshifts of binaries after galaxy merger, but before binary evolution to the
        frequencies of interest.  Unitless.
    redz_interp_grid : (Zi,) array of float, []
        The redshift values at which comoving distances are calculated; used for interpolation.
    dcom_interp_grid : (Zi,) array of float, [cm]
        The comoving-distance values at the ``redz_interp_grid`` redshifts, in units of centimeters,
        used for interpolation.

    Returns
    -------
    redz_final : (M, Q, Z, F) array of float, []
        The redshifts at which binaries at each grid point reach the frequencies of interest.
        Unitless.
    diff_num : (M, Q, Z, F) array of float, []
        The differential number of binaries at each grid point.  Unitless.

    """

    cdef int n_mtot = mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = target_fobs_orb.size
    cdef int n_interp = redz_interp_grid.size

    cdef int ii, jj, kk, ff, interp_idx, _kk
    cdef double mt, mr, ftarget, target_frst_orb, sepa, rad_isco, frst_orb_isco, rzp


    # ---- calculate dynamic binary numbers for all SAM grid bins

    for ii in range(n_mtot):
        mt = mtot[ii]
        rad_isco = 3.0 * MY_SCHW * mt
        frst_orb_isco = kepler_freq_from_sepa(mt, rad_isco)

        for jj in range(n_mrat):
            mr = mrat[jj]

            interp_idx = 0
            for _kk in range(n_redz):
                kk = n_redz - 1 - _kk

                # redz_prime is -1 for systems past age of Universe
                rzp = <double>redz_prime[ii, jj, kk]
                if rzp <= 0.0:
                    continue

                for ff in range(n_freq):
                    redz_final[ii, jj, kk, ff] = rzp

                    ftarget = target_fobs_orb[ff]
                    # find rest-frame orbital frequency and binary separation
                    target_frst_orb = ftarget * (1.0 + rzp)
                    # if target frequency is above ISCO freq, then all future ones will be also, so: break
                    if target_frst_orb > frst_orb_isco:
                        # but still fill in the final redshifts (redz_final)
                        for fp in range(ff+1, n_freq):
                            redz_final[ii, jj, kk, fp] = rzp

                        break

                    # get comoving distance
                    interp_idx = while_while_decreasing(interp_idx, n_interp, rzp, redz_interp_grid)
                    dcom = interp_at_index(interp_idx, rzp, redz_interp_grid, dcom_interp_grid)

                    # calculate total hardening rate at this exact separation
                    sepa = kepler_sepa_from_freq(mt, target_frst_orb)
                    dadt = hard_gw(mt, mr, sepa)

                    # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                    tres = - (2.0/3.0) * sepa / dadt

                    # calculate number of binaries
                    cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + rzp) * pow(dcom / MY_MPC, 2)
                    diff_num[ii, jj, kk, ff] = nden[ii, jj, kk] * tres * cosmo_fact

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _dynamic_binary_number_at_fobs_cdm(
    double[:] target_fobs_orb,
    double sepa_init,
    int num_steps,

    double gamma_sp,
    double[:, :, :] rho_s_3d_array,
    double[:, :, :] rs_3d_array,

    double[:, :, :] nden,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] gmt_time,

    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,
    double[:] tage_interp_grid,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
) except -1:
    """Calculate differential binary number at the given frequencies, with phenom 2pl evolution.

    This function converts from differential binary volume-density to differential binary number.
    Binary evolution follows the 'phenomenological' double power-law model implemented in the
    :py:func:`_hard_func_2pwl_gw`, which matches the implementation in
    :py:class:`Fixed_Time_2PL_SAM`.

    See :py:func:`dynamic_binary_number_at_fobs` for more information.

    Arguments
    ---------
    target_fobs_orb : (F,) array of float [1/s]
        The observer-frame orbital frequencies of interest, in units of inverse seconds.
    nden : (M, Q, Z) array of float [Mpc^{-3}]
        The differential binary volume-density in units of inverse-cubic comoving-Mpc.
    mtot : (M,) array of float [g]
        The edges of the total-mass grid dimension in units of grams.
    mrat : (Q,) array of float []
        The edges of the mass-ratio grid dimension.  Unitless.
    redz : (Z,) array of float []
        The edges of the redshift grid dimension.  Unitless.
    redz_prime : (M, Q, Z) array of float []
        The redshifts of binaries after galaxy merger, but before binary evolution to the
        frequencies of interest.  Unitless.
    redz_interp_grid : (Zi,) array of float, []
        The redshift values at which comoving distances are calculated; used for interpolation.
    dcom_interp_grid : (Zi,) array of float, [cm]
        The comoving-distance values at the ``redz_interp_grid`` redshifts, in units of centimeters,
        used for interpolation.

    Returns
    -------
    redz_final : (M, Q, Z, F) array of float, []
        The redshifts at which binaries at each grid point reach the frequencies of interest.
        Unitless.
    diff_num : (M, Q, Z, F) array of float, []
        The differential number of binaries at each grid point.  Unitless.

    """

    cdef int n_mtot = mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = target_fobs_orb.size
    cdef int n_interp = redz_interp_grid.size
    cdef double age_universe = tage_interp_grid[n_interp - 1]
    cdef double sepa_init_log10 = log10(sepa_init)

    cdef int ii, jj, kk, ff, step, interp_left_idx, interp_right_idx, new_interp_idx
    cdef double mt, mr, norm, risco, dx, new_redz, gmt, ftarget, target_frst_orb
    cdef double sepa_log10, sepa, sepa_left, sepa_right, dadt_left, dadt_right
    cdef double time_evo, redz_left, redz_right, time_left, time_right, new_time
    cdef double frst_orb_left, fobs_orb_left, frst_orb_right, fobs_orb_right

    # ---- Calculate ages corresponding to SAM `redz` grid

    cdef double *redz_age = <double *>malloc(n_redz * sizeof(double))     # (Z,) age of the universe in [sec]
    ii = 0
    cdef int rev
    for kk in range(n_redz):
        # iterate in reverse order to match with `redz_interp_grid` which is decreasing
        rev = n_redz - 1 - kk
        # get to the right index of the interpolation-grid
        while (redz_interp_grid[ii+1] > redz[rev]) and (ii < n_interp - 1):
            ii += 1

        # interpolate
        redz_age[rev] = interp_at_index(ii, redz[rev], redz_interp_grid, tage_interp_grid)

    # ---- calculate dynamic binary numbers for all SAM grid bins

    for ii in range(n_mtot):
        mt = mtot[ii]

        # Determine separation step-size, in log10-space, to integrate from sepa_init to ISCO
        risco = 3.0 * MY_SCHW * mt     # ISCO is 3x combined schwarzschild radius
        dx = (sepa_init_log10 - log10(risco)) / num_steps

        for jj in range(n_mrat):
            mr = mrat[jj]         
            
            for kk in range(n_redz-1, -1, -1):
                rho_s = rho_s_3d_array[ii, jj, kk]
                rs = rs_3d_array[ii, jj, kk]
                # Get total hardening rate at left-most edge
                sepa_log10 = sepa_init_log10                # set initial separation to initial value
                sepa_left = pow(10.0, sepa_log10)
                dadt_left = _hard_func_cdm_gw(mt, mr, sepa_left, rho_s, rs, gamma_sp)

                # get rest-frame orbital frequency of binary at left edge
                frst_orb_left = kepler_freq_from_sepa(mt, sepa_left)

                # Binary evolution is determined by M, q and z
                # so integration is started for each of these bins
                # ---- Integrate of `num_steps` discrete intervals in binary separation from large to small

                time_evo = 0.0                  # track total binary evolution time
                interp_left_idx = 0                 # interpolation index, will be updated in each step
                for step in range(num_steps):
                    # Increment the current separation
                    sepa_log10 -= dx
                    sepa_right = pow(10.0, sepa_log10)
                    frst_orb_right = kepler_freq_from_sepa(mt, sepa_right)

                    # Get total hardening rate at the right-edge of this step (left-edge already obtained)
                    dadt_right = _hard_func_cdm_gw(mt, mr, sepa_right, rho_s, rs, gamma_sp)

                    # Find time to move from left- to right- edges:  dt = da / (da/dt)
                    # average da/dt on the left- and right- edges of the bin (i.e. trapezoid rule)
                    dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
                    # if ii == 8 and jj == 0:
                    #     printf("cy %03d : %.2e ==> %.2e  ==  %.2e\n", step, sepa_left, sepa_right, dt)

                    time_evo += dt
                
                    # get the total time from each starting redshift, plus GMT time, plus evolution time to this step
                    gmt = gmt_time[ii, jj, kk]
                    time_right = time_evo + gmt + redz_age[kk]
                    # also get the evolution-time to the left edge
                    time_left = time_right - dt

                    # if we pass the age of the universe, this binary has stalled, no further time-evolutions will work
                    if time_left > age_universe:
                        continue

                    # find the redshift bins corresponding to left- and right- side of step
                    # left edge
                    interp_left_idx = while_while_increasing(interp_left_idx, n_interp, time_left, tage_interp_grid)
                    redz_left = interp_at_index(interp_left_idx, time_left, tage_interp_grid, redz_interp_grid)
                    # double check that left-edge is within age of Universe (should rarely if ever be a problem)
                    # but possible due to rounding/interpolation errors
                    if redz_left < 0.0:
                        continue

                    # find right-edge starting from left edge, i.e. `interp_left_idx` (`interp_left_idx` is not a typo!)
                    interp_right_idx = while_while_increasing(interp_left_idx, n_interp, time_right, tage_interp_grid)
                    # NOTE: because `time_right` can be larger than age of universe, it can exceed `tage_interp_grid`
                    #       in this case `interp_right_idx=n_interp-2`, and the `interp_at_index` function can still
                    #       be used to extrapolate to further out values, which will likely be negative

                    redz_right = interp_at_index(interp_right_idx, time_right, tage_interp_grid, redz_interp_grid)
                    # NOTE: at this point `redz_right` could be negative, even though `redz_left` is definitely not
                    if redz_right < 0.0:
                        redz_right = 0.0

                    # convert to frequencies
                    fobs_orb_left = frst_orb_left / (1.0 + redz_left)
                    fobs_orb_right = frst_orb_right / (1.0 + redz_right)

                    # ---- Iterate over all target frequencies

                    # NOTE: there should be a more efficient way to do this.
                    #       Tried a different implementation in `_dynamic_binary_number_at_fobs_1`, but not working
                    #       some of the frequency bins seem to be getting skipped in that version.

                    for ff in range(n_freq):
                        ftarget = target_fobs_orb[ff]

                        # If the integration-step does NOT bracket the target frequency, continue to next frequency
                        if (ftarget < fobs_orb_left) or (fobs_orb_right < ftarget):
                            continue

                        # ------------------------------------------------------
                        # ---- TARGET FOUND ----

                        # At this point in the code, this target frequency is inbetween the left- and right- edges
                        # of the integration step, so we can interpolate the evolution to exactly this frequency,
                        # and perform the actual dynamic_binary_number calculation

                        new_time = _interp_between_vals(ftarget, fobs_orb_left, fobs_orb_right, time_left, time_right)

                        # `time_right` can be after age of Universe, make sure interpolated value is not
                        #    if it is, then all higher-frequencies will also, so break out of target-frequency loop
                        if new_time > tage_interp_grid[n_interp - 1]:
                            break

                        # find index in interpolation grid for this exact time
                        new_interp_idx = interp_left_idx      # start from left-step edge
                        new_interp_idx = while_while_increasing(new_interp_idx, n_interp, new_time, tage_interp_grid)

                        # get redshift
                        new_redz = interp_at_index(new_interp_idx, new_time, tage_interp_grid, redz_interp_grid)
                        # get comoving distance
                        dcom = interp_at_index(new_interp_idx, new_time, tage_interp_grid, dcom_interp_grid)

                        # Store redshift
                        redz_final[ii, jj, kk, ff] = new_redz

                        # find rest-frame orbital frequency and binary separation
                        target_frst_orb = ftarget * (1.0 + new_redz)
                        sepa = kepler_sepa_from_freq(mt, target_frst_orb)

                        # calculate total hardening rate at this exact separation
                        dadt = _hard_func_cdm_gw(mt, mr, sepa, rho_s, rs, gamma_sp)

                        # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                        tres = - (2.0/3.0) * sepa / dadt

                        # calculate number of binaries
                        cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + new_redz) * pow(dcom / MY_MPC, 2)
                        diff_num[ii, jj, kk, ff] = nden[ii, jj, kk] * tres * cosmo_fact

                        # ----------------------
                        # ------------------------------------------------------

                    # update new left edge
                    dadt_left = dadt_right
                    sepa_left = sepa_right
                    frst_orb_left = frst_orb_right
                    # note that we _cannot_ do this for redz or freqs because the redshift _bin_ is changing

    free(redz_age)

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _dynamic_binary_number_at_fobs_sidm(
    double[:] target_fobs_orb,
    double sepa_init,
    int num_steps,

    double[:, :, :] nden,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] gmt_time,

    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,
    double[:] tage_interp_grid,

    double[:, :, :] r_values_a0_3d,
    double[:, :, :] rho_values_a0_3d,
    double[:, :, :] r_values_a4_3d,
    double[:, :, :] rho_values_a4_3d,
    double[:, :] v0_2d_array,
    double vt,
    double[:, :] rt_2d_array,
    double[:, :] r_sp_a4_2d_array,
    double[:] N1_array,
    double[:] N2_array,
    double gamma1,
    double gamma2,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
) except -1:
    """Calculate differential binary number at the given frequencies.

    This function converts from differential binary volume-density to differential binary number.

    See :py:func:`dynamic_binary_number_at_fobs` for more information.

    Arguments
    ---------
    target_fobs_orb : (F,) array of float [1/s]
        The observer-frame orbital frequencies of interest, in units of inverse seconds.
    nden : (M, Q, Z) array of float [Mpc^{-3}]
        The differential binary volume-density in units of inverse-cubic comoving-Mpc.
    mtot : (M,) array of float [g]
        The edges of the total-mass grid dimension in units of grams.
    mrat : (Q,) array of float []
        The edges of the mass-ratio grid dimension.  Unitless.
    redz : (Z,) array of float []
        The edges of the redshift grid dimension.  Unitless.
    redz_prime : (M, Q, Z) array of float []
        The redshifts of binaries after galaxy merger, but before binary evolution to the
        frequencies of interest.  Unitless.
    redz_interp_grid : (Zi,) array of float, []
        The redshift values at which comoving distances are calculated; used for interpolation.
    dcom_interp_grid : (Zi,) array of float, [cm]
        The comoving-distance values at the ``redz_interp_grid`` redshifts, in units of centimeters,
        used for interpolation.
    r_values_a0_3d : (M, Q, Z) 3d array of float [cm]
        The r_value array for interpolation of isothermal core denisty
    rho_values_a0_3d : (M, Q, Z) 3d array of float [g/cm^3]
        The density values array for interpolation of isothermal core denisty
    r_values_a4_3d : (M, Q, Z) 3d array of float [cm]
        The r_value array for interpolation of isothermal core denisty
    rho_values_a4_3d : (M, Q, Z) 3d array of float [g/cm^3]
        The density values array for interpolation of isothermal core denisty
    v0_2d_array : (M, Z)
        The SIDM velocity 
    vt : Transition velocity [cm/s]
    rt_2d_array : (M, Z) [cm]
    r_sp_a4_2d_array : (M, Z) [cm]
    N1_array : (Q,)
    N2_array : (Q,)
    gamma1
    gamma2

    Returns
    -------
    redz_final : (M, Q, Z, F) array of float, []
        The redshifts at which binaries at each grid point reach the frequencies of interest.
        Unitless.
    diff_num : (M, Q, Z, F) array of float, []
        The differential number of binaries at each grid point.  Unitless.

    """

    cdef int n_rvals = r_values_a0_3d[0, 0, :].size
    cdef int n_mtot = mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = target_fobs_orb.size
    cdef int n_interp = redz_interp_grid.size
    cdef double age_universe = tage_interp_grid[n_interp - 1]
    cdef double sepa_init_log10 = log10(sepa_init)

    cdef int ii, jj, kk, ff, step, interp_left_idx, interp_right_idx, new_interp_idx
    cdef double mt, mr, norm, risco, dx, new_redz, gmt, ftarget, target_frst_orb
    cdef double sepa_log10, sepa, sepa_left, sepa_right, dadt_left, dadt_right
    cdef double time_evo, redz_left, redz_right, time_left, time_right, new_time
    cdef double frst_orb_left, fobs_orb_left, frst_orb_right, fobs_orb_right

    # ---- Calculate ages corresponding to SAM `redz` grid

    cdef double *redz_age = <double *>malloc(n_redz * sizeof(double))     # (Z,) age of the universe in [sec]
    ii = 0
    cdef int rev
    for kk in range(n_redz):
        # iterate in reverse order to match with `redz_interp_grid` which is decreasing
        rev = n_redz - 1 - kk
        # get to the right index of the interpolation-grid
        while (redz_interp_grid[ii+1] > redz[rev]) and (ii < n_interp - 1):
            ii += 1

        # interpolate
        redz_age[rev] = interp_at_index(ii, redz[rev], redz_interp_grid, tage_interp_grid)

    # ---- calculate dynamic binary numbers for all SAM grid bins

    for ii in range(n_mtot):
        mt = mtot[ii]

        # Determine separation step-size, in log10-space, to integrate from sepa_init to ISCO
        risco = 3.0 * MY_SCHW * mt     # ISCO is 3x combined schwarzschild radius
        dx = (sepa_init_log10 - log10(risco)) / num_steps

        for jj in range(n_mrat):
            mr = mrat[jj]         
            n1 = N1_array[jj]
            n2 = N2_array[jj]
            for kk in range(n_redz-1, -1, -1):
                v0 = v0_2d_array[ii, kk]
                rt = rt_2d_array[ii, kk]
                r_sp_a4 = r_sp_a4_2d_array[ii, kk]
                r_values_a0 = r_values_a0_3d[ii, kk, :]
                rho_values_a0 = rho_values_a0_3d[ii, kk, :]
                r_values_a4 = r_values_a4_3d[ii, kk, :]
                rho_values_a4 = rho_values_a4_3d[ii, kk, :]
                
                # Get total hardening rate at left-most edge
                sepa_log10 = sepa_init_log10                # set initial separation to initial value
                sepa_left = pow(10.0, sepa_log10)
                dadt_left = _hard_func_sidm_gw(mt, mr, sepa_left, n_rvals, r_values_a0, rho_values_a0, r_values_a4, rho_values_a4, v0, vt, rt, r_sp_a4, n1, n2, gamma1, gamma2)

                # get rest-frame orbital frequency of binary at left edge
                frst_orb_left = kepler_freq_from_sepa(mt, sepa_left)

                # Binary evolution is determined by M, q and z
                # so integration is started for each of these bins
                # ---- Integrate of `num_steps` discrete intervals in binary separation from large to small

                time_evo = 0.0                  # track total binary evolution time
                interp_left_idx = 0                 # interpolation index, will be updated in each step
                for step in range(num_steps):
                    # Increment the current separation
                    sepa_log10 -= dx
                    sepa_right = pow(10.0, sepa_log10)
                    frst_orb_right = kepler_freq_from_sepa(mt, sepa_right)

                    # Get total hardening rate at the right-edge of this step (left-edge already obtained)
                    dadt_right = _hard_func_sidm_gw(mt, mr, sepa_right, n_rvals, r_values_a0, rho_values_a0, r_values_a4, rho_values_a4, v0, vt, rt, r_sp_a4, n1, n2, gamma1, gamma2)

                    # Find time to move from left- to right- edges:  dt = da / (da/dt)
                    # average da/dt on the left- and right- edges of the bin (i.e. trapezoid rule)
                    dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
                    # if ii == 8 and jj == 0:
                    #     printf("cy %03d : %.2e ==> %.2e  ==  %.2e\n", step, sepa_left, sepa_right, dt)

                    time_evo += dt
                
                    # get the total time from each starting redshift, plus GMT time, plus evolution time to this step
                    gmt = gmt_time[ii, jj, kk]
                    time_right = time_evo + gmt + redz_age[kk]
                    # also get the evolution-time to the left edge
                    time_left = time_right - dt

                    # if we pass the age of the universe, this binary has stalled, no further time-evolutions will work
                    if time_left > age_universe:
                        continue

                    # find the redshift bins corresponding to left- and right- side of step
                    # left edge
                    interp_left_idx = while_while_increasing(interp_left_idx, n_interp, time_left, tage_interp_grid)
                    redz_left = interp_at_index(interp_left_idx, time_left, tage_interp_grid, redz_interp_grid)
                    # double check that left-edge is within age of Universe (should rarely if ever be a problem)
                    # but possible due to rounding/interpolation errors
                    if redz_left < 0.0:
                        continue

                    # find right-edge starting from left edge, i.e. `interp_left_idx` (`interp_left_idx` is not a typo!)
                    interp_right_idx = while_while_increasing(interp_left_idx, n_interp, time_right, tage_interp_grid)
                    # NOTE: because `time_right` can be larger than age of universe, it can exceed `tage_interp_grid`
                    #       in this case `interp_right_idx=n_interp-2`, and the `interp_at_index` function can still
                    #       be used to extrapolate to further out values, which will likely be negative

                    redz_right = interp_at_index(interp_right_idx, time_right, tage_interp_grid, redz_interp_grid)
                    # NOTE: at this point `redz_right` could be negative, even though `redz_left` is definitely not
                    if redz_right < 0.0:
                        redz_right = 0.0

                    # convert to frequencies
                    fobs_orb_left = frst_orb_left / (1.0 + redz_left)
                    fobs_orb_right = frst_orb_right / (1.0 + redz_right)

                    # ---- Iterate over all target frequencies

                    # NOTE: there should be a more efficient way to do this.
                    #       Tried a different implementation in `_dynamic_binary_number_at_fobs_1`, but not working
                    #       some of the frequency bins seem to be getting skipped in that version.

                    for ff in range(n_freq):
                        ftarget = target_fobs_orb[ff]

                        # If the integration-step does NOT bracket the target frequency, continue to next frequency
                        if (ftarget < fobs_orb_left) or (fobs_orb_right < ftarget):
                            continue

                        # ------------------------------------------------------
                        # ---- TARGET FOUND ----

                        # At this point in the code, this target frequency is inbetween the left- and right- edges
                        # of the integration step, so we can interpolate the evolution to exactly this frequency,
                        # and perform the actual dynamic_binary_number calculation

                        new_time = _interp_between_vals(ftarget, fobs_orb_left, fobs_orb_right, time_left, time_right)

                        # `time_right` can be after age of Universe, make sure interpolated value is not
                        #    if it is, then all higher-frequencies will also, so break out of target-frequency loop
                        if new_time > tage_interp_grid[n_interp - 1]:
                            break

                        # find index in interpolation grid for this exact time
                        new_interp_idx = interp_left_idx      # start from left-step edge
                        new_interp_idx = while_while_increasing(new_interp_idx, n_interp, new_time, tage_interp_grid)

                        # get redshift
                        new_redz = interp_at_index(new_interp_idx, new_time, tage_interp_grid, redz_interp_grid)
                        # get comoving distance
                        dcom = interp_at_index(new_interp_idx, new_time, tage_interp_grid, dcom_interp_grid)

                        # Store redshift
                        redz_final[ii, jj, kk, ff] = new_redz

                        # find rest-frame orbital frequency and binary separation
                        target_frst_orb = ftarget * (1.0 + new_redz)
                        sepa = kepler_sepa_from_freq(mt, target_frst_orb)

                        # calculate total hardening rate at this exact separation
                        dadt = _hard_func_sidm_gw(mt, mr, sepa, n_rvals, r_values_a0, rho_values_a0, r_values_a4, rho_values_a4, v0, vt, rt, r_sp_a4, n1, n2, gamma1, gamma2)

                        # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                        tres = - (2.0/3.0) * sepa / dadt

                        # calculate number of binaries
                        cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + new_redz) * pow(dcom / MY_MPC, 2)
                        diff_num[ii, jj, kk, ff] = nden[ii, jj, kk] * tres * cosmo_fact

                        # ----------------------
                        # ------------------------------------------------------

                    # update new left edge
                    dadt_left = dadt_right
                    sepa_left = sepa_right
                    frst_orb_left = frst_orb_right
                    # note that we _cannot_ do this for redz or freqs because the redshift _bin_ is changing

    free(redz_age)

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _dynamic_binary_number_at_fobs_sidm_Version2(
    double[:] target_fobs_orb,
    double sepa_init,
    int num_steps,

    double[:, :, :] nden,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] gmt_time,

    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,
    double[:] tage_interp_grid,

    double[:] r_values,
    double[:, :, :] density_array,
    double[:] N1_array,
    double[:] N2_array,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
) except -1:
    """Calculate differential binary number at the given frequencies.

    This function converts from differential binary volume-density to differential binary number.

    See :py:func:`dynamic_binary_number_at_fobs` for more information.

    Arguments
    ---------
    target_fobs_orb : (F,) array of float [1/s]
        The observer-frame orbital frequencies of interest, in units of inverse seconds.
    nden : (M, Q, Z) array of float [Mpc^{-3}]
        The differential binary volume-density in units of inverse-cubic comoving-Mpc.
    mtot : (M,) array of float [g]
        The edges of the total-mass grid dimension in units of grams.
    mrat : (Q,) array of float []
        The edges of the mass-ratio grid dimension.  Unitless.
    redz : (Z,) array of float []
        The edges of the redshift grid dimension.  Unitless.
    redz_prime : (M, Q, Z) array of float []
        The redshifts of binaries after galaxy merger, but before binary evolution to the
        frequencies of interest.  Unitless.
    redz_interp_grid : (Zi,) array of float, []
        The redshift values at which comoving distances are calculated; used for interpolation.
    dcom_interp_grid : (Zi,) array of float, [cm]
        The comoving-distance values at the ``redz_interp_grid`` redshifts, in units of centimeters,
        used for interpolation.
    r_values_a0_3d : (M, Q, Z) 3d array of float [cm]
        The r_value array for interpolation of isothermal core denisty
    rho_values_a0_3d : (M, Q, Z) 3d array of float [g/cm^3]
        The density values array for interpolation of isothermal core denisty
    r_values_a4_3d : (M, Q, Z) 3d array of float [cm]
        The r_value array for interpolation of isothermal core denisty
    rho_values_a4_3d : (M, Q, Z) 3d array of float [g/cm^3]
        The density values array for interpolation of isothermal core denisty
    v0_2d_array : (M, Z)
        The SIDM velocity 
    vt : Transition velocity [cm/s]
    rt_2d_array : (M, Z) [cm]
    r_sp_a4_2d_array : (M, Z) [cm]
    N1_array : (Q,)
    N2_array : (Q,)
    gamma1
    gamma2

    Returns
    -------
    redz_final : (M, Q, Z, F) array of float, []
        The redshifts at which binaries at each grid point reach the frequencies of interest.
        Unitless.
    diff_num : (M, Q, Z, F) array of float, []
        The differential number of binaries at each grid point.  Unitless.

    """

    cdef int n_rvals = r_values.size
    cdef int n_mtot = mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = target_fobs_orb.size
    cdef int n_interp = redz_interp_grid.size
    cdef double age_universe = tage_interp_grid[n_interp - 1]
    cdef double sepa_init_log10 = log10(sepa_init)

    cdef int ii, jj, kk, ff, step, interp_left_idx, interp_right_idx, new_interp_idx
    cdef double mt, mr, norm, risco, dx, new_redz, gmt, ftarget, target_frst_orb
    cdef double sepa_log10, sepa, sepa_left, sepa_right, dadt_left, dadt_right
    cdef double time_evo, redz_left, redz_right, time_left, time_right, new_time
    cdef double frst_orb_left, fobs_orb_left, frst_orb_right, fobs_orb_right

    # ---- Calculate ages corresponding to SAM `redz` grid

    cdef double *redz_age = <double *>malloc(n_redz * sizeof(double))     # (Z,) age of the universe in [sec]
    ii = 0
    cdef int rev
    for kk in range(n_redz):
        # iterate in reverse order to match with `redz_interp_grid` which is decreasing
        rev = n_redz - 1 - kk
        # get to the right index of the interpolation-grid
        while (redz_interp_grid[ii+1] > redz[rev]) and (ii < n_interp - 1):
            ii += 1

        # interpolate
        redz_age[rev] = interp_at_index(ii, redz[rev], redz_interp_grid, tage_interp_grid)

    # ---- calculate dynamic binary numbers for all SAM grid bins

    for ii in range(n_mtot):
        mt = mtot[ii]

        # Determine separation step-size, in log10-space, to integrate from sepa_init to ISCO
        risco = 3.0 * MY_SCHW * mt     # ISCO is 3x combined schwarzschild radius
        dx = (sepa_init_log10 - log10(risco)) / num_steps

        for jj in range(n_mrat):
            mr = mrat[jj]         
            n1 = N1_array[jj]
            n2 = N2_array[jj]
            for kk in range(n_redz-1, -1, -1):
                density_values = density_array[ii, kk, :]
                
                # Get total hardening rate at left-most edge
                sepa_log10 = sepa_init_log10                # set initial separation to initial value
                sepa_left = pow(10.0, sepa_log10)
                dadt_left = _hard_func_sidm_gw_Version2(mt, mr, sepa_left, n_rvals, r_values, density_values, n1, n2)

                # get rest-frame orbital frequency of binary at left edge
                frst_orb_left = kepler_freq_from_sepa(mt, sepa_left)

                # Binary evolution is determined by M, q and z
                # so integration is started for each of these bins
                # ---- Integrate of `num_steps` discrete intervals in binary separation from large to small

                time_evo = 0.0                  # track total binary evolution time
                interp_left_idx = 0                 # interpolation index, will be updated in each step
                for step in range(num_steps):
                    # Increment the current separation
                    sepa_log10 -= dx
                    sepa_right = pow(10.0, sepa_log10)
                    frst_orb_right = kepler_freq_from_sepa(mt, sepa_right)

                    # Get total hardening rate at the right-edge of this step (left-edge already obtained)
                    dadt_right = _hard_func_sidm_gw_Version2(mt, mr, sepa_right, n_rvals, r_values, density_values, n1, n2)

                    # Find time to move from left- to right- edges:  dt = da / (da/dt)
                    # average da/dt on the left- and right- edges of the bin (i.e. trapezoid rule)
                    dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
                    # if ii == 8 and jj == 0:
                    #     printf("cy %03d : %.2e ==> %.2e  ==  %.2e\n", step, sepa_left, sepa_right, dt)

                    time_evo += dt
                
                    # get the total time from each starting redshift, plus GMT time, plus evolution time to this step
                    gmt = gmt_time[ii, jj, kk]
                    time_right = time_evo + gmt + redz_age[kk]
                    # also get the evolution-time to the left edge
                    time_left = time_right - dt

                    # if we pass the age of the universe, this binary has stalled, no further time-evolutions will work
                    if time_left > age_universe:
                        continue

                    # find the redshift bins corresponding to left- and right- side of step
                    # left edge
                    interp_left_idx = while_while_increasing(interp_left_idx, n_interp, time_left, tage_interp_grid)
                    redz_left = interp_at_index(interp_left_idx, time_left, tage_interp_grid, redz_interp_grid)
                    # double check that left-edge is within age of Universe (should rarely if ever be a problem)
                    # but possible due to rounding/interpolation errors
                    if redz_left < 0.0:
                        continue

                    # find right-edge starting from left edge, i.e. `interp_left_idx` (`interp_left_idx` is not a typo!)
                    interp_right_idx = while_while_increasing(interp_left_idx, n_interp, time_right, tage_interp_grid)
                    # NOTE: because `time_right` can be larger than age of universe, it can exceed `tage_interp_grid`
                    #       in this case `interp_right_idx=n_interp-2`, and the `interp_at_index` function can still
                    #       be used to extrapolate to further out values, which will likely be negative

                    redz_right = interp_at_index(interp_right_idx, time_right, tage_interp_grid, redz_interp_grid)
                    # NOTE: at this point `redz_right` could be negative, even though `redz_left` is definitely not
                    if redz_right < 0.0:
                        redz_right = 0.0

                    # convert to frequencies
                    fobs_orb_left = frst_orb_left / (1.0 + redz_left)
                    fobs_orb_right = frst_orb_right / (1.0 + redz_right)

                    # ---- Iterate over all target frequencies

                    # NOTE: there should be a more efficient way to do this.
                    #       Tried a different implementation in `_dynamic_binary_number_at_fobs_1`, but not working
                    #       some of the frequency bins seem to be getting skipped in that version.

                    for ff in range(n_freq):
                        ftarget = target_fobs_orb[ff]

                        # If the integration-step does NOT bracket the target frequency, continue to next frequency
                        if (ftarget < fobs_orb_left) or (fobs_orb_right < ftarget):
                            continue

                        # ------------------------------------------------------
                        # ---- TARGET FOUND ----

                        # At this point in the code, this target frequency is inbetween the left- and right- edges
                        # of the integration step, so we can interpolate the evolution to exactly this frequency,
                        # and perform the actual dynamic_binary_number calculation

                        new_time = _interp_between_vals(ftarget, fobs_orb_left, fobs_orb_right, time_left, time_right)

                        # `time_right` can be after age of Universe, make sure interpolated value is not
                        #    if it is, then all higher-frequencies will also, so break out of target-frequency loop
                        if new_time > tage_interp_grid[n_interp - 1]:
                            break

                        # find index in interpolation grid for this exact time
                        new_interp_idx = interp_left_idx      # start from left-step edge
                        new_interp_idx = while_while_increasing(new_interp_idx, n_interp, new_time, tage_interp_grid)

                        # get redshift
                        new_redz = interp_at_index(new_interp_idx, new_time, tage_interp_grid, redz_interp_grid)
                        # get comoving distance
                        dcom = interp_at_index(new_interp_idx, new_time, tage_interp_grid, dcom_interp_grid)

                        # Store redshift
                        redz_final[ii, jj, kk, ff] = new_redz

                        # find rest-frame orbital frequency and binary separation
                        target_frst_orb = ftarget * (1.0 + new_redz)
                        sepa = kepler_sepa_from_freq(mt, target_frst_orb)

                        # calculate total hardening rate at this exact separation
                        dadt = _hard_func_sidm_gw_Version2(mt, mr, sepa, n_rvals, r_values, density_values, n1, n2)

                        # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                        tres = - (2.0/3.0) * sepa / dadt

                        # calculate number of binaries
                        cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + new_redz) * pow(dcom / MY_MPC, 2)
                        diff_num[ii, jj, kk, ff] = nden[ii, jj, kk] * tres * cosmo_fact

                        # ----------------------
                        # ------------------------------------------------------

                    # update new left edge
                    dadt_left = dadt_right
                    sepa_left = sepa_right
                    frst_orb_left = frst_orb_right
                    # note that we _cannot_ do this for redz or freqs because the redshift _bin_ is changing

    free(redz_age)

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _dynamic_binary_number_at_fobs_3bs(
    double[:] target_fobs_orb,
    double sepa_init,
    int num_steps,

    double[:] hard_rho_i_1d_array,
    double[:] hard_sigma_i_1d_array,
    double H,

    double[:, :, :] nden,
    double[:] mtot,
    double[:] mrat,
    double[:] redz,
    double[:, :, :] gmt_time,

    double[:] redz_interp_grid,
    double[:] dcom_interp_grid,
    double[:] tage_interp_grid,

    # output
    double[:, :, :, :] redz_final,
    double[:, :, :, :] diff_num,
) except -1:
    """Calculate differential binary number at the given frequencies, with 3 Body Scattering evolution.
    [Chen2024]


    Arguments
    ---------
    target_fobs_orb : (F,) array of float [1/s]
        The observer-frame orbital frequencies of interest, in units of inverse seconds.
    nden : (M, Q, Z) array of float [Mpc^{-3}]
        The differential binary volume-density in units of inverse-cubic comoving-Mpc.
    mtot : (M,) array of float [g]
        The edges of the total-mass grid dimension in units of grams.
    mrat : (Q,) array of float []
        The edges of the mass-ratio grid dimension.  Unitless.
    redz : (Z,) array of float []
        The edges of the redshift grid dimension.  Unitless.
    redz_prime : (M, Q, Z) array of float []
        The redshifts of binaries after galaxy merger, but before binary evolution to the
        frequencies of interest.  Unitless.
    redz_interp_grid : (Zi,) array of float, []
        The redshift values at which comoving distances are calculated; used for interpolation.
    dcom_interp_grid : (Zi,) array of float, [cm]
        The comoving-distance values at the ``redz_interp_grid`` redshifts, in units of centimeters,
        used for interpolation.

    Returns
    -------
    redz_final : (M, Q, Z, F) array of float, []
        The redshifts at which binaries at each grid point reach the frequencies of interest.
        Unitless.
    diff_num : (M, Q, Z, F) array of float, []
        The differential number of binaries at each grid point.  Unitless.

    """

    cdef int n_mtot = mtot.size
    cdef int n_mrat = mrat.size
    cdef int n_redz = redz.size
    cdef int n_freq = target_fobs_orb.size
    cdef int n_interp = redz_interp_grid.size
    cdef double age_universe = tage_interp_grid[n_interp - 1]
    cdef double sepa_init_log10 = log10(sepa_init)

    cdef int ii, jj, kk, ff, step, interp_left_idx, interp_right_idx, new_interp_idx
    cdef double mt, mr, norm, risco, dx, new_redz, gmt, ftarget, target_frst_orb
    cdef double sepa_log10, sepa, sepa_left, sepa_right, dadt_left, dadt_right
    cdef double time_evo, redz_left, redz_right, time_left, time_right, new_time
    cdef double frst_orb_left, fobs_orb_left, frst_orb_right, fobs_orb_right

    # ---- Calculate ages corresponding to SAM `redz` grid

    cdef double *redz_age = <double *>malloc(n_redz * sizeof(double))     # (Z,) age of the universe in [sec]
    ii = 0
    cdef int rev
    for kk in range(n_redz):
        # iterate in reverse order to match with `redz_interp_grid` which is decreasing
        rev = n_redz - 1 - kk
        # get to the right index of the interpolation-grid
        while (redz_interp_grid[ii+1] > redz[rev]) and (ii < n_interp - 1):
            ii += 1

        # interpolate
        redz_age[rev] = interp_at_index(ii, redz[rev], redz_interp_grid, tage_interp_grid)

    # ---- calculate dynamic binary numbers for all SAM grid bins

    for ii in range(n_mtot):
        mt = mtot[ii]
        rho_i = hard_rho_i_1d_array[ii]
        sigma_i = hard_sigma_i_1d_array[ii]

        # Determine separation step-size, in log10-space, to integrate from sepa_init to ISCO
        risco = 3.0 * MY_SCHW * mt     # ISCO is 3x combined schwarzschild radius
        dx = (sepa_init_log10 - log10(risco)) / num_steps

        for jj in range(n_mrat):
            mr = mrat[jj]

            # Binary evolution is determined by M and q only
            # so integration is started for each of these bins
            sepa_log10 = sepa_init_log10                # set initial separation to initial value
            
            # Get total hardening rate at left-most edge
            sepa_left = pow(10.0, sepa_log10)
            dadt_left = _hard_func_3bs_gw(mt, mr, sepa_left, rho_i, sigma_i, H)

            # get rest-frame orbital frequency of binary at left edge
            frst_orb_left = kepler_freq_from_sepa(mt, sepa_left)

            # ---- Integrate of `num_steps` discrete intervals in binary separation from large to small

            time_evo = 0.0                  # track total binary evolution time
            interp_left_idx = 0                 # interpolation index, will be updated in each step
            for step in range(num_steps):
                # Increment the current separation
                sepa_log10 -= dx
                sepa_right = pow(10.0, sepa_log10)
                frst_orb_right = kepler_freq_from_sepa(mt, sepa_right)

                # Get total hardening rate at the right-edge of this step (left-edge already obtained)
                dadt_right = _hard_func_3bs_gw(mt, mr, sepa_right, rho_i, sigma_i, H)

                # Find time to move from left- to right- edges:  dt = da / (da/dt)
                # average da/dt on the left- and right- edges of the bin (i.e. trapezoid rule)
                dt = 2.0 * (sepa_right - sepa_left) / (dadt_left + dadt_right)
                # if ii == 8 and jj == 0:
                #     printf("cy %03d : %.2e ==> %.2e  ==  %.2e\n", step, sepa_left, sepa_right, dt)

                time_evo += dt

                # ---- Iterate over starting redshift bins

                for kk in range(n_redz-1, -1, -1):
                    # get the total time from each starting redshift, plus GMT time, plus evolution time to this step
                    gmt = gmt_time[ii, jj, kk]
                    time_right = time_evo + gmt + redz_age[kk]
                    # also get the evolution-time to the left edge
                    time_left = time_right - dt

                    # if we pass the age of the universe, this binary has stalled, no further redshifts will work
                    # NOTE: if `gmt_time` decreases faster than redshift bins increase the universe age,
                    #       then systems in later `redz` bins may no longer stall, so we still need to calculate them.
                    #       i.e. we can NOT use a `break` statement here, must use `continue` statement.
                    if time_left > age_universe:
                        continue

                    # find the redshift bins corresponding to left- and right- side of step
                    # left edge
                    interp_left_idx = while_while_increasing(interp_left_idx, n_interp, time_left, tage_interp_grid)

                    redz_left = interp_at_index(interp_left_idx, time_left, tage_interp_grid, redz_interp_grid)

                    # double check that left-edge is within age of Universe (should rarely if ever be a problem
                    # but possible due to rounding/interpolation errors
                    if redz_left < 0.0:
                        continue

                    # find right-edge starting from left edge, i.e. `interp_left_idx` (`interp_left_idx` is not a typo!)
                    interp_right_idx = while_while_increasing(interp_left_idx, n_interp, time_right, tage_interp_grid)
                    # NOTE: because `time_right` can be larger than age of universe, it can exceed `tage_interp_grid`
                    #       in this case `interp_right_idx=n_interp-2`, and the `interp_at_index` function can still
                    #       be used to extrapolate to further out values, which will likely be negative

                    redz_right = interp_at_index(interp_right_idx, time_right, tage_interp_grid, redz_interp_grid)
                    # NOTE: at this point `redz_right` could be negative, even though `redz_left` is definitely not
                    if redz_right < 0.0:
                        redz_right = 0.0

                    # if ii == 8 and jj == 0 and kk == 11:
                    #     printf("cy %03d : t=%.2e z=%.2e\n", step, time_right, redz_right)

                    # convert to frequencies
                    fobs_orb_left = frst_orb_left / (1.0 + redz_left)
                    fobs_orb_right = frst_orb_right / (1.0 + redz_right)

                    # ---- Iterate over all target frequencies

                    # NOTE: there should be a more efficient way to do this.
                    #       Tried a different implementation in `_dynamic_binary_number_at_fobs_1`, but not working
                    #       some of the frequency bins seem to be getting skipped in that version.

                    for ff in range(n_freq):
                        ftarget = target_fobs_orb[ff]

                        # If the integration-step does NOT bracket the target frequency, continue to next frequency
                        if (ftarget < fobs_orb_left) or (fobs_orb_right < ftarget):
                            continue

                        # ------------------------------------------------------
                        # ---- TARGET FOUND ----

                        # At this point in the code, this target frequency is inbetween the left- and right- edges
                        # of the integration step, so we can interpolate the evolution to exactly this frequency,
                        # and perform the actual dynamic_binary_number calculation

                        new_time = _interp_between_vals(ftarget, fobs_orb_left, fobs_orb_right, time_left, time_right)

                        # `time_right` can be after age of Universe, make sure interpolated value is not
                        #    if it is, then all higher-frequencies will also, so break out of target-frequency loop
                        if new_time > tage_interp_grid[n_interp - 1]:
                            break

                        # find index in interpolation grid for this exact time
                        new_interp_idx = interp_left_idx      # start from left-step edge
                        new_interp_idx = while_while_increasing(new_interp_idx, n_interp, new_time, tage_interp_grid)

                        # get redshift
                        new_redz = interp_at_index(new_interp_idx, new_time, tage_interp_grid, redz_interp_grid)
                        # get comoving distance
                        dcom = interp_at_index(new_interp_idx, new_time, tage_interp_grid, dcom_interp_grid)

                        # if (ii == 0) and (jj == 0) and (kk == 0):
                        #     printf("cy f=%03d (step=%03d)\n", ff, step)
                        #     printf(
                        #         "fl=%.6e, f=%.6e, fr=%.6e ==> tl=%.6e, t=%.6e, tr=%.6e\n",
                        #         fobs_orb_left, ftarget, fobs_orb_right,
                        #         time_left, new_time, time_right
                        #     )
                        #     printf(
                        #         "interp (%d) time: %.6e, %.6e, %.6e ==> z: %.6e, %.6e, %.6e\n",
                        #         new_interp_idx,
                        #         tage_interp_grid[new_interp_idx], new_time, tage_interp_grid[new_interp_idx+1],
                        #         redz_interp_grid[new_interp_idx], new_redz, redz_interp_grid[new_interp_idx+1],
                        #     )
                        #     printf("======> z=%.6e\n", new_redz)

                        # Store redshift
                        redz_final[ii, jj, kk, ff] = new_redz

                        # find rest-frame orbital frequency and binary separation
                        target_frst_orb = ftarget * (1.0 + new_redz)
                        sepa = kepler_sepa_from_freq(mt, target_frst_orb)

                        # calculate total hardening rate at this exact separation
                        dadt = _hard_func_3bs_gw(mt, mr, sepa, rho_i, sigma_i, H)

                        # calculate residence/hardening time = f/[df/dt] = -(2/3) a/[da/dt]
                        tres = - (2.0/3.0) * sepa / dadt

                        # calculate number of binaries
                        cosmo_fact = FOUR_PI_SPLC_OVER_MPC * (1.0 + new_redz) * pow(dcom / MY_MPC, 2)
                        diff_num[ii, jj, kk, ff] = nden[ii, jj, kk] * tres * cosmo_fact

                        # ----------------------
                        # ------------------------------------------------------

                # update new left edge
                dadt_left = dadt_right
                sepa_left = sepa_right
                frst_orb_left = frst_orb_right
                # note that we _cannot_ do this for redz or freqs because the redshift _bin_ is changing

    free(redz_age)

    return 0