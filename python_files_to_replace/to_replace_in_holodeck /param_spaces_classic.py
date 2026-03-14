"""'Classic' parameter spaces used in the NANOGrav 15yr analysis.
"""

from holodeck.constants import PC, GYR
from holodeck.librarian.lib_tools import _Param_Space, PD_Uniform, PD_Normal
from holodeck import sams, hardening, host_relations


class _PS_Classic_Phenom(_Param_Space):
    """Base class for classic phenomenological parameter space used in 15yr analysis.
    """

    DEFAULTS = dict(
        hard_time=3.0,          # [Gyr]
        hard_sepa_init=1e4,     # [pc]
        hard_rchar=100.0,       # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0_log10=-2.77,
        gsmf_phiz=-0.6,
        gsmf_mchar0_log10=11.24,
        gsmf_mcharz=0.11,
        gsmf_alpha0=-1.21,
        gsmf_alphaz=-0.03,

        gpf_frac_norm_allq=0.025,
        gpf_malpha=0.0,
        gpf_qgamma=0.0,
        gpf_zbeta=1.0,
        gpf_max_frac=1.0,

        gmt_norm=0.5,           # [Gyr]
        gmt_malpha=0.0,
        gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
        gmt_zbeta=-0.5,

        mmb_mamp_log10=8.69,
        mmb_plaw=1.10,          # average MM2013 and KH2013
        mmb_scatter_dex=0.3,
    )

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm_allq=params['gpf_frac_norm_allq'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_KH2013(
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
            # scatter_dex=params['mmb_scatter'], # to match the param name from pspace
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam

    @classmethod
    def _init_hard(cls, sam, params):
        hard = hardening.Fixed_Time_2PL_SAM(
            sam,
            params['hard_time']*GYR,
            sepa_init=params['hard_sepa_init']*PC,
            rchar=params['hard_rchar']*PC,
            gamma_inner=params['hard_gamma_inner'],
            gamma_outer=params['hard_gamma_outer'],
        )
        return hard


class PS_Classic_Phenom_Uniform(_PS_Classic_Phenom):
    """Classic 5D phenomenological, uniform parameter space used in 15yr analysis.

    Previously called the `PS_Uniform_09B` parameter space, or 'phenom-uniform'.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("gsmf_phi0_log10", -3.5, -1.5),
            PD_Uniform("gsmf_mchar0_log10", 10.5, 12.5),   # [log10(Msol)]
            PD_Uniform("mmb_mamp_log10", +7.6, +9.0),      # [log10(Msol)]
            PD_Uniform("mmb_scatter_dex", +0.0, +0.9),
            PD_Uniform("hard_time", 0.1, 11.0),            # [Gyr]
            PD_Uniform("hard_gamma_inner", -1.5, +0.0),
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )


class PS_Classic_Phenom_Astro_Extended(_PS_Classic_Phenom):
    """Classic 12D phenomenological, uniform parameter space used in 15yr analysis.

    Previously called the `PS_New_Astro_02B` parameter space, or 'phenom-astro+extended'.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("hard_time", 0.1, 11.0),   # [Gyr]
            PD_Uniform("hard_gamma_inner", -1.5, +0.5),

            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            # PD_Normal("gsmf_phi0", -2.56, 0.4),
            PD_Normal("gsmf_phi0_log10", -2.56, 0.4), # because in class _Param_Space from librarian/lib_tools.py lines 112, 181
            # gsmf_phi0 is replaced by gsmf_phi0_log10. so it should be replaced here too.
            PD_Normal("gsmf_mchar0_log10", 10.9, 0.4),   # [log10(Msol)]
            PD_Normal("gsmf_alpha0", -1.2, 0.2),

            PD_Normal("gpf_zbeta", +0.8, 0.4),
            PD_Normal("gpf_qgamma", +0.5, 0.3),

            PD_Uniform("gmt_norm", 0.2, 5.0),    # [Gyr]
            PD_Uniform("gmt_zbeta", -2.0, +0.0),

            PD_Normal("mmb_mamp_log10", +8.6, 0.2),   # [log10(Msol)]
            PD_Normal("mmb_plaw", +1.2, 0.2),
            PD_Normal("mmb_scatter_dex", +0.32, 0.15),
        ]
        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )


class _PS_Classic_GWOnly(_Param_Space):
    """Base class for classic GW-Only parameter space used in 15yr analysis.
    """

    DEFAULTS = dict(
        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        # gsmf_phi0=-2.77,
        gsmf_phi0_log10=-2.77, # because in class _Param_Space from librarian/lib_tools.py lines 112, 181
        # gsmf_phi0 is replaced by gsmf_phi0_log10. so it should be replaced here too.
        gsmf_phiz=-0.6,
        gsmf_mchar0_log10=11.24,
        gsmf_mcharz=0.11,
        gsmf_alpha0=-1.21,
        gsmf_alphaz=-0.03,

        gpf_frac_norm_allq=0.025,
        gpf_malpha=0.0,
        gpf_qgamma=0.0,
        gpf_zbeta=1.0,
        gpf_max_frac=1.0,

        gmt_norm=0.5,           # [Gyr]
        gmt_malpha=0.0,
        gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
        gmt_zbeta=-0.5,

        mmb_mamp_log10=8.69,
        mmb_plaw=1.10,          # average MM2013 and KH2013
        mmb_scatter_dex=0.3,
    )

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # PD_Uniform("gsmf_phi0", -3.5, -1.5),
            PD_Uniform("gsmf_phi0_log10", -3.5, -1.5), # because in class _Param_Space from librarian/lib_tools.py lines 112, 181
            # gsmf_phi0 is replaced by gsmf_phi0_log10. so it should be replaced here too.
            PD_Uniform("gsmf_mchar0_log10", 10.5, 12.5),   # [log10(Msol)]
            # PD_Uniform("mmb_mamp_log10", +7.5, +9.5),   # [log10(Msol)]
            # PD_Uniform("mmb_scatter_dex", +0.0, +1.2), # name changed to mmb_scatter_dex to match the convention in rest of the code
            PD_Uniform("mmb_mamp_log10", +7.6, +9.0),   # [log10(Msol)] # changed to match with agazie2023 table B1
            PD_Uniform("mmb_scatter_dex", +0.0, +0.9), # changed to match with agazie2023 table B1
        ]
        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            # phi0=params['gsmf_phi0'],
            phi0=params['gsmf_phi0_log10'], # because in class _Param_Space from librarian/lib_tools.py lines 112, 181
            # gsmf_phi0 is replaced by gsmf_phi0_log10. so it should be replaced here too.
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm_allq=params['gpf_frac_norm_allq'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_KH2013(
        # mmbulge = host_relations.MMBulge_Chen2019( #added by Shreyas to implement mstar-dependant bulge frac from Chen2019, and Alonso-Alvarez 2024
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
            # scatter_dex=params['mmb_scatter'], # to match the param name from pspace
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam

    @classmethod
    def _init_hard(cls, sam, params):
        hard = hardening.Hard_GW()
        return hard


class PS_Classic_GWOnly_Uniform(_PS_Classic_GWOnly):
    """Classic 4D GW-Only, uniform parameter space used in 15yr analysis.

    Previously called the `PS_Uniform_07_GW` parameter space, or 'gw-only'.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # PD_Uniform("gsmf_phi0", -3.5, -1.5),
            PD_Uniform("gsmf_phi0_log10", -3.5, -1.5),  # because in class _Param_Space from librarian/lib_tools.py lines 112, 181
            # gsmf_phi0 is replaced by gsmf_phi0_log10. so it should be replaced here too.
            PD_Uniform("gsmf_mchar0_log10", 10.5, 12.5),   # [log10(Msol)]
            # PD_Uniform("mmb_mamp_log10", +7.5, +9.5),   # [log10(Msol)]
            # PD_Uniform("mmb_scatter_dex", +0.0, +1.2), # name changed to mmb_scatter_dex to match the convention in rest of the code
            PD_Uniform("mmb_mamp_log10", +7.6, +9.0),   # [log10(Msol)] # changed to match with agazie2023 table B1
            PD_Uniform("mmb_scatter_dex", +0.0, +0.9), # changed to match with agazie2023 table B1
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class PS_Classic_GWOnly_Astro_Extended(_PS_Classic_GWOnly):
    """Classic 10D GW-Only, uniform parameter space used in 15yr analysis.

    Previously called the `PS_New_Astro_02_GW` parameter space, or 'gw-only+extended'.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            # from `sam-parameters.ipynb` fits to [Tomczak+2014] with 4x stdev values
            # PD_Normal("gsmf_phi0", -2.56, 0.4),
            PD_Normal("gsmf_phi0_log10", -2.56, 0.4),  # because in class _Param_Space from librarian/lib_tools.py lines 112, 181
            # gsmf_phi0 is replaced by gsmf_phi0_log10. so it should be replaced here too.
            PD_Normal("gsmf_mchar0_log10", 10.9, 0.4),   # [log10(Msol)]
            PD_Normal("gsmf_alpha0", -1.2, 0.2),

            PD_Normal("gpf_zbeta", +0.8, 0.4),
            PD_Normal("gpf_qgamma", +0.5, 0.3),

            PD_Uniform("gmt_norm", 0.2, 5.0),    # [Gyr]
            PD_Uniform("gmt_zbeta", -2.0, +0.0),

            PD_Normal("mmb_mamp_log10", +8.6, 0.2),   # [log10(Msol)]
            PD_Normal("mmb_plaw", +1.2, 0.2),
            PD_Normal("mmb_scatter_dex", +0.32, 0.15),
        ]
        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class PS_Test(_PS_Classic_Phenom):
    """Simple test parameter space in 2D.
    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):

        parameters = [
            PD_Uniform("mmb_mamp_log10", +7.5, +9.5),   # [log10(Msol)]
            PD_Uniform("hard_time", 0.1, 11.0),         # [Gyr]
        ]

        _Param_Space.__init__(
            self, parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        return


class _PS_Classic_CDM(_Param_Space):
    """Base class for CDM (added by Shreyas) taking inspiration from class _PS_Classic_Phenom(_Param_Space).
    """
    DEFAULTS = dict(
        hard_time=3.0,          # [Gyr]
        hard_sepa_init=1e4,     # [pc]
        hard_rchar=100.0,       # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0_log10=-2.77,
        gsmf_phiz=-0.6,
        gsmf_mchar0_log10=11.5, # ACD2024 table 1 (different than in phenom case)
        gsmf_mcharz=0.11,
        gsmf_alpha0=-1.21,
        gsmf_alphaz=-0.03,

        gpf_frac_norm_allq=0.033, # ACD2024 table 1 (different than in phenom case)
        gpf_malpha=0.0,
        gpf_qgamma=0.0,
        gpf_zbeta=1.0,
        gpf_max_frac=1.0,

        gmt_norm=0.5,           # [Gyr]
        gmt_malpha=0.0,
        gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
        gmt_zbeta=-0.5,

        mmb_mamp_log10=8.7, # ACD2024 eq.A3 (different than in phenom case)
        mmb_plaw=1.10,          # average MM2013 and KH2013
        mmb_scatter_dex=0.0, # ACD2024 eq.A3 (different than in phenom case)

        gamma_sp = 1.5 # just some value inspired from ACD2024
    )

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm_allq=params['gpf_frac_norm_allq'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_Chen2019(
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
            # scatter_dex=params['mmb_scatter'], # to match the param name from pspace
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam


    @classmethod
    def _init_hard(cls, sam, params):
        import holodeck
        from holodeck.galaxy_profiles import NFW
        import numpy as np

        Girelli_2020_instance = holodeck.host_relations.Girelli_2020()
        mstar = sam._mmbulge.mstar_from_mbh(sam.mtot)
        mhalo = Girelli_2020_instance.halo_mass(mstar, sam.redz)
        rho_s, rs = NFW._nfw_rho_rad(mhalo, sam.redz)
        rho_s_3d_array = np.broadcast_to(rho_s[:, np.newaxis, :], (91, 81, 101)) # 3d array (shape M, Q, Z)
        rs_3d_array = np.broadcast_to(rs[:, np.newaxis, :], (91, 81, 101)) # 3d array (shape M, Q, Z)
        
        hard = hardening.Hard_CDM(
                rho_s_3d_array= rho_s_3d_array,
                rs_3d_array = rs_3d_array,
                gamma_sp=params['gamma_sp'],
                sepa_init=params['hard_sepa_init']*PC,
                )
        return hard
    
class PS_Classic_CDM_Uniform(_PS_Classic_CDM):
    """
    2D CDM case (inspired form ACD2024) with only gamma_sp and GSMF (\psi_0) carying, uniform parameter space.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("gsmf_phi0_log10", -3.5, -1.5),
            PD_Uniform("gamma_sp", 0.5, 7.0/3.0),
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )


class _PS_Classic_SIDM(_Param_Space):
    """
    Base class for CDM (added by Shreyas) taking inspiration from class _PS_Classic_Phenom(_Param_Space).
    """
    import numpy as np
    DEFAULTS = dict(
        hard_time=3.0,          # [Gyr]
        hard_sepa_init=1e4,     # [pc]
        hard_rchar=100.0,       # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0_log10=-2.77,
        gsmf_phiz=-0.6,
        gsmf_mchar0_log10=11.5, # ACD2024 table 1 (different than in phenom case)
        gsmf_mcharz=0.11,
        gsmf_alpha0=-1.21,
        gsmf_alphaz=-0.03,

        gpf_frac_norm_allq=0.033, # ACD2024 table 1 (different than in phenom case)
        gpf_malpha=0.0,
        gpf_qgamma=0.0,
        gpf_zbeta=1.0,
        gpf_max_frac=1.0,

        gmt_norm=0.5,           # [Gyr]
        gmt_malpha=0.0,
        gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
        gmt_zbeta=-0.5,

        mmb_mamp_log10=8.7, # ACD2024 eq.A3 (different than in phenom case)
        mmb_plaw=1.10,          # average MM2013 and KH2013
        mmb_scatter_dex=0.0, # ACD2024 eq.A3 (different than in phenom case)

        vt = 500, # (km/s) just some value inspired from ACD2024
        sigma0_over_m_times_t_age_per_1Gyr = 30 # (cm**2/g) just some value inspired from ACD2024
        # log10_vt = np.log10(500), # converting to log10
        # log10_sigma0_over_m_times_t_age_per_1Gyr = np.log10(30) # converting to log10
    )

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm_allq=params['gpf_frac_norm_allq'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_Chen2019(
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
            # scatter_dex=params['mmb_scatter'], # to match the param name from pspace
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam


    @classmethod
    def _init_hard(cls, sam, params):
        
        vt_kms = params['vt']
        vt = vt_kms * 1e5 # cgs
        sigma0_over_m_times_t_age_per_1Gyr = params['sigma0_over_m_times_t_age_per_1Gyr']
        import numpy as np
        # log10_vt_kms = params['log10_vt']
        # vt_kms = np.power(10, log10_vt_kms)
        # vt = vt_kms * 1e5 # cgs
        # log10_sigma0_over_m_times_t_age_per_1Gyr = params['log10_sigma0_over_m_times_t_age_per_1Gyr']
        # sigma0_over_m_times_t_age_per_1Gyr = np.power(10, log10_sigma0_over_m_times_t_age_per_1Gyr)
        
        import holodeck
        from holodeck.constants import YR, NWTG, SPLC
        from holodeck.galaxy_profiles import NFW
        from holodeck import utils
        from holodeck.librarian import DEF_NUM_FBINS
        from scipy.interpolate import interp1d
        from scipy.optimize import root_scalar
        from scipy.integrate import solve_ivp, quad
        import kalepy as kale

        Girelli_2020_instance = holodeck.host_relations.Girelli_2020()
        mstar = sam._mmbulge.mstar_from_mbh(sam.mtot)
        mhalo = Girelli_2020_instance.halo_mass(mstar, sam.redz)
        rho_s, rs = NFW._nfw_rho_rad(mhalo, sam.redz)

        ## FIND A BETTER WAY
        load_path = "/home/users/sti50/Codes/sidm_lambda0_yvalues_for_cvalues_1_to_5_for_500_points.npz"
        data = np.load(load_path)
        C_values, Lambda0_higher_values, Lambda0_lower_values, y_higher_values, y_lower_values = data["a1"], data["a2"], data["a3"], data["a4"], data["a5"]
        #####################
        interval_high = np.argmax(y_higher_values), np.where(y_higher_values>0)[0][-1] # Indices of the highest positive elements, and the last positive element
        # y has a max value, and then it monotonically decreases with C
        # +30 because otherwise extrapolation works badly (more explanationin /home/users/sti50/Codes/sidm_stats_check_and_results.ipynb) 08.06.2025
        C_of_y_higher_values = interp1d(y_higher_values[interval_high[0] +30: interval_high[1] + 1], C_values[interval_high[0] +30: interval_high[1] + 1], fill_value="extrapolate") 

        # for v0 < vt case
        # calculate everything for a=0
        subintervals = np.linspace(1e-10, y_higher_values[interval_high[0]], 100)
        y_a0 = np.zeros((len(sam.mtot), len(sam.redz)))
        r1_a0 = np.zeros((len(sam.mtot), len(sam.redz)))
        v0_a0 = np.zeros((len(sam.mtot), len(sam.redz)))
        for i in range(rho_s.shape[0]):
            for j in range(rho_s.shape[1]):
                def v0_y_higher_values(y): # using eq. B4# v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_lower_values(y)) * NWTG * rho_s * r_s**2 # in cgs (cm/s)
                    v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_higher_values(y)) * NWTG * rho_s[i, j] * rs[i, j]**2 # in cgs (cm/s)
                    return np.sqrt(v0_squared)
                
                def eqn2(y):
                    # term_1 = sigma0_over_m_times_t_age_per_1Gyr * GYR *  vref * (vref/v0_y_higher_values(y))**(a_0 - 1) * rho_s[i, j] / (y * (1 + y)**2)
                    # equation above is equivalent of equation below
                    term_1 = sigma0_over_m_times_t_age_per_1Gyr * GYR *  v0_y_higher_values(y) * rho_s[i, j] / (y * (1 + y)**2) # t_age is expressed in Gyrs
                    return term_1 - 1
            
                count = 0 # To see how many times the equation becomes zero
                for k in range(len(subintervals) - 1):
                        lower, upper = subintervals[k], subintervals[k + 1]
                        if eqn2(lower) * eqn2(upper) < 0:  # Sign change indicates a root in (a, b)
                            result = root_scalar(eqn2, bracket=(lower, upper), method='bisect')
                            if result.converged:
                                y_a0[i, j] = result.root
                                r1_a0[i, j] = y_a0[i, j] * rs[i, j]
                                v0_a0[i, j] = v0_y_higher_values(y_a0[i, j])
                            count = count + 1
                            break
                if count == 0:
                    y_a0[i, j] = np.max(y_higher_values)
                    r1_a0[i, j] = y_a0[i, j] * rs[i, j]
                    v0_a0[i, j] = v0_y_higher_values(y_a0[i, j])
                    
        # for v0 > vt
        # calculate for a=4, but instead of eq.3 use sigma*v = sigma0 * (vt/v0)**4 * v0 in eq.2
        subintervals = np.linspace(1e-10, y_higher_values[interval_high[0]], 100)
        y_a4 = y_a0.copy()
        r1_a4 = r1_a0.copy()
        v0_a4 = v0_a0.copy()
        for i in range(rho_s.shape[0]):
            for j in range(rho_s.shape[1]):
                if(v0_a0[i, j] > vt):
                    def v0_y_higher_values(y): # using eq. B4# v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_lower_values(y)) * G * rho_s_in_si * r_s_in_si**2 # in SI (m/s)
                        v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_higher_values(y)) * NWTG * rho_s[i, j] * rs[i, j]**2 # in cgs (cm/s)
                        return np.sqrt(v0_squared)
                    def eqn2(y):
                        # term_1 = (vt / v0_y_higher_values(y))**4 * v0_y_higher_values(y) * sigma0_over_m_times_t_age_per_1Gyr * GYR *  v0_y_higher_values(y) * rho_s[i, j] / (y * (1 + y)**2)
                        # equation above is equivalent of equation below
                        term_1 = vt**4 / v0_y_higher_values(y)**3 * sigma0_over_m_times_t_age_per_1Gyr * GYR *  rho_s[i, j] / (y * (1 + y)**2) # t_age is expressed in Gyrs
                        return term_1 - 1
                    count = 0 # To see how many times the equation becomes zero
                    for k in range(len(subintervals) - 1):
                            lower, upper = subintervals[k], subintervals[k + 1]
                            if eqn2(lower) * eqn2(upper) < 0:  # Sign change indicates a root in (a, b)
                                result = root_scalar(eqn2, bracket=(lower, upper), method='bisect')
                                if result.converged:
                                    y_a4[i, j] = result.root
                                    r1_a4[i, j] = y_a4[i, j] * rs[i, j]
                                    v0_a4[i, j] = v0_y_higher_values(y_a4[i, j])
                                count = count + 1

        # +30 because otherwise extrapolation works badly (more explanationin /home/users/sti50/Codes/sidm_stats_check_and_results.ipynb) 08.06.2025
        Lambda0_of_y_higher_values = interp1d(y_higher_values[interval_high[0] +30: interval_high[1] + 1], Lambda0_higher_values[interval_high[0] +30: interval_high[1] + 1], fill_value="extrapolate") 
        # a=0
        Lambda0_higher_a0 = Lambda0_of_y_higher_values(r1_a0 / rs)
        rho_c_a0 = rho_s / ((r1_a0 / rs) * (1 + r1_a0 / rs)**2)
        C_a0 = 4 * np.pi * NWTG * rho_c_a0 * r1_a0**2 / v0_a0**2 
        # a=4
        Lambda0_higher_a4 = Lambda0_of_y_higher_values(r1_a4 / rs)
        rho_c_a4 = rho_s / ((r1_a4 / rs) * (1 + r1_a4 / rs)**2)
        C_a4 = 4 * np.pi * NWTG * rho_c_a4 * r1_a4**2 / v0_a4**2 
        # Define the differential equation system
        def ode(w, vector, C):
            Lambda, dLambda_dw = vector
            d2Lambda_dw2 = -C * np.exp(Lambda) - 2 * dLambda_dw / w
            return [dLambda_dw, d2Lambda_dw2]
        # for rt (transition radius)
        
        rt = np.zeros((len(sam.mtot), len(sam.redz)))
        # a=0
        r_sp_a0 = NWTG * np.broadcast_to(sam.mtot[:, np.newaxis], (91, 101)) / v0_a0**2 # below acd2024 eq.3 #cgs
        # a=4
        r_sp_a4 = NWTG * np.broadcast_to(sam.mtot[:, np.newaxis], (91, 101)) / v0_a4**2 # below acd2024 eq.3 #cgs
        ######## testing ################
        interval = (1e-10 * PC, 1e10 * PC) # changed to get better calculation of rt (06.06.2025)
        subintervals = np.linspace(interval[0], interval[1], 5000)
        #################################
        # interval = (1e-1 * PC, 1e6 * PC) # r values chosen by observation of Fig.2 (seeing approximately between what values does r_t occur)
        # subintervals = np.linspace(interval[0], interval[1], 500)
        for i in range(len(sam.mtot)):
            for j in range(len(sam.redz)):
                if(v0_a0[i, j] < vt):
                    def v_r(r): # Equation B6
                        return v0_a0[i, j] * (7/11 + 4/11 * (r_sp_a0[i, j] / r)**0.5) # v_0 in cm/s , r_sp in cm, so we also must input r in cm
                    count = 0
                    def f(r): # This is what we want to become zero at r=r_t
                        return v_r(r) - vt
                    for k in range(len(subintervals) - 1):
                        a, b = subintervals[k], subintervals[k + 1]
                        # print(f(a), f(b))
                        if f(a) * f(b) < 0:  # Sign change indicates a root in (a, b)
                            result = root_scalar(f, bracket=(a, b), method='bisect')
                            if result.converged:
                                rt[i, j] = result.root
                                count = count + 1
        
        
        n_for_ivp_solver = 100
        rho_core_values_a0 = np.zeros((91, 101, n_for_ivp_solver))
        rho_core_values_a4 = np.zeros((91, 101, n_for_ivp_solver))
        r_values_a0 = np.zeros((91, 101, n_for_ivp_solver))
        r_values_a4 = np.zeros((91, 101, n_for_ivp_solver))

        for i in range(len(sam.mtot)):
            for j in range(len(sam.redz)):
                if vt < v0_a0[i, j]:
                    # only a=4 spike
                    # so, only need core profile with r1_a4 as boundary condition
                    solution_a4 = solve_ivp(ode, [1e-8, 1], [Lambda0_higher_a4[i, j], 0], args=(C_a4[i, j],), t_eval=np.linspace(1e-8, 1.0, 100))
                    Lambda_values_a4 = solution_a4.y[0]
                    w_values_a4 = solution_a4.t
                    rho_core_values_a4[i, j, :] = rho_c_a4[i, j] * np.exp(Lambda_values_a4) # cgs
                    r_values_a4[i, j, :] = r1_a4[i, j] * w_values_a4 # cm
                else:
                    # only need core profile with r1_04 as boundary condition
                    solution_a0 = solve_ivp(ode, [1e-8, 1], [Lambda0_higher_a0[i, j], 0], args=(C_a0[i, j],), t_eval=np.linspace(1e-8, 1.0, 100))
                    Lambda_values_a0 = solution_a0.y[0]  
                    w_values_a0 = solution_a0.t
                    rho_core_values_a0[i, j, :] = rho_c_a0[i, j] * np.exp(Lambda_values_a0) # cgs
                    r_values_a0[i, j, :] = r1_a0[i, j] * w_values_a0 # cm
        from scipy.special import erf
        u_1 = 11/4 * sam.mrat**(3/2) * (1 + sam.mrat)**(-3/2) # ACD eq.C3
        N1 = erf(u_1 / np.sqrt(2)) - np.sqrt(2 / np.pi) * u_1 * np.exp(-u_1**2 / 2) # ACD eq.C2
        u_2 = 11/4 * (1 + sam.mrat)**(-3/2) # ACD eq.C3
        N2 = erf(u_2 / np.sqrt(2)) - np.sqrt(2 / np.pi) * u_2 * np.exp(-u_2**2 / 2) # ACD eq.C2
        hard = hardening.Hard_SIDM(
                rt = rt,
                r_sp_a4 = r_sp_a4,
                v0_a0 = v0_a0,
                vt = vt,
                r_values_a0 = r_values_a0,
                r_values_a4 = r_values_a4,
                rho_core_values_a0 = rho_core_values_a0,
                rho_core_values_a4 = rho_core_values_a4,
                N1_array = N1,
                N2_array = N2,
                sepa_init=params['hard_sepa_init']*PC
                )
        return hard

class PS_Classic_CDM_Uniform(_PS_Classic_CDM):
    """
    2D CDM case (inspired form ACD2024) with only gamma_sp and GSMF (\psi_0) carying, uniform parameter space.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("gsmf_phi0_log10", -3.5, -1.5),
            PD_Uniform("gamma_sp", 0.5, 7.0/3.0),
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )

class PS_Classic_SIDM_Uniform(_PS_Classic_SIDM):
    """
    SIDM case (inspired form ACD2024) with only vt, sigma0_over_m, and GSMF (\psi_0) carying, uniform parameter space.

    """
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        import numpy as np
        parameters = [
            PD_Uniform("gsmf_phi0_log10", -3.5, -1.5),
            # PD_Uniform("vt", 10, 2000), # km/s (given in ACD2024 just before Conculsions)
            PD_Uniform("vt", 1, 2000), # km/s (based on fig.4 but not excluding the grey region)
            # PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.2, 100), # cm**2/g (given in ACD2024 just before Conculsions)
            PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.01, 100), # cm**2/g (based on fig.4 but not excluding the grey region)
            # PD_Uniform("log10_vt", np.log10(10), np.log10(2000)), # km/s (given in ACD2024 just before Conculsions)
            # PD_Uniform("log10_sigma0_over_m_times_t_age_per_1Gyr", np.log10(0.2), np.log10(100)), # cm**2/g (given in ACD2024 just before Conculsions)
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        
class PS_Classic_SIDM_Astro(_PS_Classic_SIDM):
    """
    SIDM case (inspired form ACD2024) with only vt, sigma0_over_m, and GSMF (\psi_0) varying, astro parameter space for psi0.

    """
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        import numpy as np
        parameters = [
            PD_Normal("gsmf_phi0_log10", -2.56, 0.4), # from Agazie2023 table B1
            # PD_Uniform("vt", 10, 2000), # km/s (given in ACD2024 just before Conculsions)
            PD_Uniform("vt", 1, 2000), # km/s (based on fig.4 but not excluding the grey region)
            # PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.2, 100), # cm**2/g (given in ACD2024 just before Conculsions)
            PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.01, 100), # cm**2/g (based on fig.4 but not excluding the grey region)
            # PD_Uniform("log10_vt", np.log10(10), np.log10(2000)), # km/s (given in ACD2024 just before Conculsions)
            # PD_Uniform("log10_sigma0_over_m_times_t_age_per_1Gyr", np.log10(0.2), np.log10(100)), # cm**2/g (given in ACD2024 just before Conculsions)
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
        
class PS_Classic_SIDM_Astro_Extended(_PS_Classic_SIDM):
    """
    SIDM case (inspired form ACD2024) with vt, sigma0_over_m, 
    GSMF (\psi_0), mchar0, mmb_mamp_log10, mmb_scatter_dex varying, astro parameter space for psi0.

    """
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        import numpy as np
        parameters = [
            PD_Normal("gsmf_phi0_log10", -2.56, 0.4), # from Agazie2023 table B1
            PD_Normal("gsmf_mchar0_log10", 10.9, 0.4), # from Agazie2023 table B1
            PD_Normal("mmb_mamp_log10", +8.6, 0.2), # from Agazie2023 table B1
            PD_Normal("mmb_scatter_dex", +0.32, 0.15), # from Agazie2023 table B1
            
            # PD_Uniform("vt", 10, 2000), # km/s (given in ACD2024 just before Conculsions)
            PD_Uniform("vt", 1, 2000), # km/s (based on fig.4 but not excluding the grey region)
            # PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.2, 100), # cm**2/g (given in ACD2024 just before Conculsions)
            PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.01, 200), # cm**2/g (based on fig.4 but not excluding the grey region)
            # PD_Uniform("log10_vt", np.log10(10), np.log10(2000)), # km/s (given in ACD2024 just before Conculsions)
            # PD_Uniform("log10_sigma0_over_m_times_t_age_per_1Gyr", np.log10(0.2), np.log10(100)), # cm**2/g (given in ACD2024 just before Conculsions)
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )  

# for fixing density profile calculation in _PS_Classic_SIDM

class _PS_Classic_SIDM_Version2(_Param_Space):
    """
    Base class for CDM (added by Shreyas) taking inspiration from class _PS_Classic_Phenom(_Param_Space).
    """
    import numpy as np
    DEFAULTS = dict(
        hard_time=3.0,          # [Gyr]
        # hard_sepa_init=1e4,     # [pc]
        hard_sepa_init=1, # [pc] changing to see if this changes results
        hard_rchar=100.0,       # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        gsmf_phi0_log10=-2.77,
        gsmf_phiz=-0.6,
        gsmf_mchar0_log10=11.5, # ACD2024 table 1 (different than in phenom case)
        gsmf_mcharz=0.11,
        gsmf_alpha0=-1.21,
        gsmf_alphaz=-0.03,

        gpf_frac_norm_allq=0.033, # ACD2024 table 1 (different than in phenom case)
        gpf_malpha=0.0,
        gpf_qgamma=0.0,
        gpf_zbeta=1.0,
        gpf_max_frac=1.0,

        gmt_norm=0.5,           # [Gyr]
        gmt_malpha=0.0,
        gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
        gmt_zbeta=-0.5,

        mmb_mamp_log10=8.7, # ACD2024 eq.A3 (different than in phenom case)
        mmb_plaw=1.10,          # average MM2013 and KH2013
        mmb_scatter_dex=0.0, # ACD2024 eq.A3 (different than in phenom case)

        vt = 500, # (km/s) just some value inspired from ACD2024
        sigma0_over_m_times_t_age_per_1Gyr = 30 # (cm**2/g) just some value inspired from ACD2024
        # log10_vt = np.log10(500), # converting to log10
        # log10_sigma0_over_m_times_t_age_per_1Gyr = np.log10(30) # converting to log10
    )

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm_allq=params['gpf_frac_norm_allq'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_Chen2019(
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
            # scatter_dex=params['mmb_scatter'], # to match the param name from pspace
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam


    @classmethod
    def _init_hard(cls, sam, params):
        # trying to give hard just density array and corresponding r values, instead of core and spike profiles separately
        # using code from /home/users/sti50/Codes/holodeck_latest_version_with_my_changes_strain_with_sidm.ipynb for density profiles as I tested it and it works
        
        vt_kms = params['vt']
        vt = vt_kms * 1e5 # cgs
        sigma0_over_m_times_t_age_per_1Gyr = params['sigma0_over_m_times_t_age_per_1Gyr']
        import numpy as np
        
        import holodeck
        from holodeck.constants import YR, NWTG, SPLC
        from holodeck.galaxy_profiles import NFW
        from scipy.interpolate import interp1d
        from scipy.optimize import root_scalar
        from scipy.integrate import solve_ivp

        Girelli_2020_instance = holodeck.host_relations.Girelli_2020()
        mstar = sam._mmbulge.mstar_from_mbh(sam.mtot)
        mhalo = Girelli_2020_instance.halo_mass(mstar, sam.redz)
        rho_s, rs = NFW._nfw_rho_rad(mhalo, sam.redz)

        ## FIND A BETTER WAY
        load_path = "/home/sti50/project_shreyas_sidm/sidm_lambda0_yvalues_for_cvalues_1_to_5_for_500_points.npz"
        data = np.load(load_path)
        C_values, Lambda0_higher_values, Lambda0_lower_values, y_higher_values, y_lower_values = data["a1"], data["a2"], data["a3"], data["a4"], data["a5"]
        #####################
        interval_high = np.argmax(y_higher_values), np.where(y_higher_values>0)[0][-1] # Indices of the highest positive elements, and the last positive element
        # y has a max value, and then it monotonically decreases with C
        # +30 because otherwise extrapolation works badly (more explanationin /home/users/sti50/Codes/sidm_stats_check_and_results.ipynb) 08.06.2025
        C_of_y_higher_values = interp1d(y_higher_values[interval_high[0] + 30: interval_high[1] + 1], C_values[interval_high[0] +30 : interval_high[1] + 1], fill_value="extrapolate") 

        # for v0 < vt case
        # calculate everything for a=0
        subintervals = np.linspace(1e-10, y_higher_values[interval_high[0]], 100)
        y = np.zeros((len(sam.mtot), len(sam.redz)))
        r1 = np.zeros((len(sam.mtot), len(sam.redz)))
        v0 = np.zeros((len(sam.mtot), len(sam.redz)))

        from tqdm import tqdm


        for i in range(rho_s.shape[0]):
            for j in range(rho_s.shape[1]):
                
                def v0_y_higher_values(y): # using eq. B4# v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_lower_values(y)) * NWTG * rho_s * r_s**2 # in cgs (cm/s)
                    v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_higher_values(y)) * NWTG * rho_s[i, j] * rs[i, j]**2 # in cgs (cm/s)
                    return np.sqrt(v0_squared)
                
                def eqn2(y):
                    term_1 = sigma0_over_m_times_t_age_per_1Gyr * GYR * v0_y_higher_values(y) * rho_s[i, j] / (y * (1 + y)**2)
                    return term_1 - 1

                count = 0 # To see how many times the equation becomes zero
                for k in range(len(subintervals) - 1):
                        lower, upper = subintervals[k], subintervals[k + 1]
                        if eqn2(lower) * eqn2(upper) < 0:  # Sign change indicates a root in (a, b)
                            result = root_scalar(eqn2, bracket=(lower, upper), method='bisect')
                            if result.converged:
                                y[i, j] = result.root
                                r1[i, j] = y[i, j] * rs[i, j]
                                v0[i, j] = v0_y_higher_values(y[i, j])
                            count = count + 1
                            break
                if count == 0:
                    # if eqn2 doesn't become zero, it means that the following
                    # if r1 is small, one can have high number of collisions over lifetime of core
                    # we want to see what is the largest possible r1 where there's at least one collision
                    # but there is a limit on how large r1 can be (y_high has a maximum value)
                    # so, in case root can't be found, we set r1 to maximum allowed (possible) value
                    y[i, j] = np.max(y_higher_values)
                    r1[i, j] = y[i, j] * rs[i, j]
                    v0[i, j] = v0_y_higher_values(y[i, j])
                    # print(f'solving eqn2, no roots found for (i, j)={(i, j)}')
                    
        ### NEXT, DO IT FOR v0>vt CASE. GET new v0 array- v0_recalculated. 
        # r1 and y is replaced in old arrays.

        v0_recalculated = v0.copy()

        for i in range(len(np.where(v0>vt)[0])):
            m_idx = np.where(v0>vt)[0][i]
            z_idx = np.where(v0>vt)[1][i]
            def v0_y_higher_values(y): # using eq. B4# v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_lower_values(y)) * NWTG * rho_s * r_s**2 # in cgs (cm/s)
                    v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_higher_values(y)) * NWTG * rho_s[m_idx, z_idx] * rs[m_idx, z_idx]**2 # in cgs (cm/s)
                    return np.sqrt(v0_squared)
            def eqn2(y):
                # sigma * v = sigma0 / (1 + (v0/vt)**4) * v0

                # term_1 = (vt / v0_y_higher_values(y))**4 * sigma0_over_m_times_t_age_by_1Gyr * GYR * v0_y_higher_values(y) * rho_s / (y * (1 + y)**2) # t_age is expressed in Gyrs
                # same as above, just rewriting so that v0_y_higher_values isn't computed multiple times
                term_1 = vt * (vt / v0_y_higher_values(y))**3 * sigma0_over_m_times_t_age_per_1Gyr * GYR * rho_s[m_idx, z_idx] / (y * (1 + y)**2) # t_age is expressed in Gyrs
                
                return term_1 - 1
            count = 0 # To see how many times the equation becomes zero
            for k in range(len(subintervals) - 1):
                lower, upper = subintervals[k], subintervals[k + 1]
                if eqn2(lower) * eqn2(upper) < 0:  # Sign change indicates a root in (a, b)
                    result = root_scalar(eqn2, bracket=(lower, upper), method='bisect')
                    if result.converged:
                        y_val = result.root
                        r1_val = y_val * rs[m_idx, z_idx]
                        v0_val = v0_y_higher_values(y_val)
                    count = count + 1
                    break
            if count == 0:
                # print(f'v0 recalculation, solving eqn2, no roots found for (i, j)={(i, j)}')
                y_val = np.max(y_higher_values)
                r1_val = y_val * rs[m_idx, z_idx]
                v0_val = v0_y_higher_values(y_val)
            ############# v0_recalculated #################################
            v0_recalculated[m_idx, z_idx] = v0_val
            ############# v0_recalculated #################################
            r1[m_idx, z_idx] = r1_val
            y[m_idx, z_idx] = y_val

        
        ############# v0_recalculated #################################
        r_sp = NWTG * np.broadcast_to(sam.mtot[:, np.newaxis], (91, 101)) / v0_recalculated**2 # below acd2024 eq.3 #cgs
        ############# v0_recalculated #################################

        rt = np.zeros((len(sam.mtot), len(sam.redz)))
        # code below is copied form /home/users/sti50/Codes/dm_spike_acd2024_fig2_version3.ipynb
        interval = (1e-11 * PC, 1e8 * PC) # r values chosen by observation of Fig.2 (seeing approximately between what values does r_t occur)
        subintervals = np.linspace(interval[0], interval[1], 5000)

        for i in range(len(sam.mtot)):
            for j in range(len(sam.redz)):
                def v_r(r): # Equation B6
                    # not necessary to change v0 here, as for the cases where it's recalculated, rt is of no use anyway
                    ############# v0_recalculated #################################
                    return v0_recalculated[i, j] * (7/11 + 4/11 * (r_sp[i, j] / r)**0.5) # v_0 in cm/s , r_sp in cm, so we also must input r in cm
                    ############# v0_recalculated #################################
                def f(r): # This is what we want to become zero at r=r_t
                    return v_r(r) - vt
                
                count = 0
                for k in range(len(subintervals) - 1):
                    a, b = subintervals[k], subintervals[k + 1]
                    # print(f(a), f(b))
                    if f(a) * f(b) < 0:  # Sign change indicates a root in (a, b)
                        result = root_scalar(f, bracket=(a, b), method='bisect')
                        if result.converged:
                            rt[i, j] = result.root
                            # print(f'Transition radius r_t = {rt[i, j] / PC} parsecs')
                            count = count + 1
        
        # for density profile
        # rt instead of r_sp in both spike profiles

        r_cutoff = 4 * NWTG * sam.mtot / SPLC**2 # cms (cgs)
        # assuming that r1, r_sp, rt, r_cutoff are already calculated

        rvals = np.logspace(-10, 10, 500) * PC # cm
        density_array = np.zeros((len(sam.mtot), len(sam.redz), len(rvals)))

        # (more explanation for changing indices by 30 in /home/users/sti50/Codes/sidm_stats_check_and_results.ipynb)
        Lambda0_of_y_higher_values = interp1d(y_higher_values[interval_high[0] + 30: interval_high[1] + 1], Lambda0_higher_values[interval_high[0] + 30: interval_high[1] + 1], fill_value="extrapolate") 

        Lambda0_higher = Lambda0_of_y_higher_values(r1 / rs)
        rho_c = rho_s / ((r1 / rs) * (1 + r1 / rs)**2)

        ############# v0_recalculated #################################
        C = 4 * np.pi * NWTG * rho_c * r1**2 / v0_recalculated**2 
        ############# v0_recalculated #################################

        # Define the differential equation system
        def ode(w, vector, C):
            Lambda, dLambda_dw = vector
            d2Lambda_dw2 = -C * np.exp(Lambda) - 2 * dLambda_dw / w
            return [dLambda_dw, d2Lambda_dw2]

        for i in range(0, 91):
            for j in range(0, 101):
                # to get core profile function for this i and j
                solution = solve_ivp(ode, [1e-8, 1], [Lambda0_higher[i, j], 0], args=(C[i, j],), t_eval=np.linspace(1e-8, 1.0, 100))
                Lambda_values = solution.y[0]  
                w_values = solution.t
                rho_core = rho_c[i, j] * np.exp(Lambda_values) # cgs
                r_values = r1[i, j] * w_values # cm
                core_profile = interp1d(r_values, rho_core, fill_value="extrapolate")

                # print(f'v0 = {v0[i, j] / 1e5} km/s & vt = {vt / 1e5} km/s')
                # print(f'v0_recalculaed = {v0_recalculated[i, j] / 1e5} km/s')
                # print(f'Core radius r1 = {"{:.1e}".format(r1[i, j] / PC)} parsecs')
                # print(f'Spike radius r_sp = {"{:.1e}".format(r_sp[i, j] / PC)} parsecs')
                # print(f'Transition radius rt = {"{:.7e}".format(rt[i, j] / PC)} parsecs')
                # print(f'Cutoff radius r_cutoff = {"{:.7e}".format(r_cutoff[i] / PC)} parsecs')

                if(v0[i, j] < vt):
                    def core_spike_profile_r_smaller_than_rt(r):
                        density = (core_profile(r)
                                    + (core_profile(r1[i, j]) * np.exp(Lambda0_higher[i, j]) 
                                                * (rt[i, j] / r)**(7 / 4))
                                                # * (r_sp[i, j] / r)**(7 / 4))
                        )
                        return density # cgs
                    def core_spike_profile_r_greater_than_rt(r):
                        density = (core_profile(r)
                                    + (core_profile(r1[i, j]) * np.exp(Lambda0_higher[i, j]) 
                                                * (rt[i, j] / r)**(3 / 4))
                                                # * (r_sp[i, j] / r)**(3 / 4))
                        )
                        return density # cgs

                    for k, r in enumerate(rvals):
                        if((r < rt[i, j]) & (r > r_cutoff[i])):
                            density_array[i, j, k] = core_spike_profile_r_smaller_than_rt(r)
                        if((r > rt[i, j]) & (r < r1[i, j])):
                            density_array[i, j, k] = core_spike_profile_r_greater_than_rt(r)
                        if(r > r1[i, j]): # NFW
                            density_array[i, j, k] = rho_s[i, j] / (r / rs[i, j]) / (1 + r / rs[i, j])**2

                else:
                    def core_spike_profile(r):
                        density = (core_profile(r)
                                + (core_profile(r1[i, j]) * np.exp(Lambda0_higher[i, j]) 
                                            * (r_sp[i, j] / r)**(7 / 4))
                        )
                        return density # cgs
                    
                    for k, r in enumerate(rvals):
                        if(r > r_cutoff[i]):
                            density_array[i, j, k] = core_spike_profile(r)
                        if(r > r1[i, j]): # NFW
                            density_array[i, j, k] = rho_s[i, j] / (r / rs[i, j]) / (1 + r / rs[i, j])**2
        
        from scipy.special import erf
        u_1 = 11/4 * sam.mrat**(3/2) * (1 + sam.mrat)**(-3/2) # ACD eq.C3
        N1 = erf(u_1 / np.sqrt(2)) - np.sqrt(2 / np.pi) * u_1 * np.exp(-u_1**2 / 2) # ACD eq.C2
        u_2 = 11/4 * (1 + sam.mrat)**(-3/2) # ACD eq.C3
        N2 = erf(u_2 / np.sqrt(2)) - np.sqrt(2 / np.pi) * u_2 * np.exp(-u_2**2 / 2) # ACD eq.C2
        hard = hardening.Hard_SIDM_Version2(
                density_3d_array = density_array,
                rvals_array = rvals,
                N1_array = N1,
                N2_array = N2,
                sepa_init=params['hard_sepa_init']*PC
                )
        return hard
class PS_Classic_SIDM_Astro_Extended_Version2(_PS_Classic_SIDM_Version2):
    """
    SIDM case (inspired form ACD2024) with vt, sigma0_over_m, 
    GSMF (\psi_0), mchar0, mmb_mamp_log10, mmb_scatter_dex varying, astro parameter space for psi0.

    """
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        import numpy as np
        parameters = [
            PD_Normal("gsmf_phi0_log10", -2.56, 0.4), # from Agazie2023 table B1
            PD_Normal("gsmf_mchar0_log10", 10.9, 0.4), # from Agazie2023 table B1
            PD_Normal("mmb_mamp_log10", +8.6, 0.2), # from Agazie2023 table B1
            PD_Normal("mmb_scatter_dex", +0.32, 0.15), # from Agazie2023 table B1
            
            # PD_Uniform("vt", 10, 2000), # km/s (given in ACD2024 just before Conculsions)
            PD_Uniform("vt", 1, 2000), # km/s (based on fig.4 but not excluding the grey region)
            # PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.2, 100), # cm**2/g (given in ACD2024 just before Conculsions)
            PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.01, 200), # cm**2/g (based on fig.4 but not excluding the grey region)
            # PD_Uniform("log10_vt", np.log10(10), np.log10(2000)), # km/s (given in ACD2024 just before Conculsions)
            # PD_Uniform("log10_sigma0_over_m_times_t_age_per_1Gyr", np.log10(0.2), np.log10(100)), # cm**2/g (given in ACD2024 just before Conculsions)
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        ) 

class PS_Classic_SIDM_Astro_Uniform_Extended_Version2(_PS_Classic_SIDM_Version2):
    """
    SIDM case (inspired form ACD2024) with vt, sigma0_over_m, 
    GSMF (\psi_0), mchar0, mmb_mamp_log10, mmb_scatter_dex varying
    PD_Uniform instead of PD_Normal for astro params to train GPs better near the boundaries. 
    """
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        import numpy as np
        parameters = [
            PD_Uniform("gsmf_phi0_log10", -3.5, -1.5),
            PD_Uniform("gsmf_mchar0_log10", 10.5, 12.5),   # [log10(Msol)]
            PD_Uniform("mmb_mamp_log10", +7.6, +9.0),      # [log10(Msol)]
            PD_Uniform("mmb_scatter_dex", +0.0, +0.9),
            PD_Uniform("vt", 1, 2000), # km/s (based on fig.4 but not excluding the grey region)
            PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.01, 200), # cm**2/g (based on fig.4 but not excluding the grey region)
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )  

###### for testing GPs, making them more robust
class _PS_Classic_SIDM_GP_Improvement(_Param_Space):
    """
    Base class for CDM (added by Shreyas) taking inspiration from class _PS_Classic_Phenom(_Param_Space).
    """
    import numpy as np
    DEFAULTS = dict(
        hard_time=3.0,          # [Gyr]
        hard_sepa_init=1e4,     # [pc]
        hard_rchar=100.0,       # [pc]
        hard_gamma_inner=-1.0,
        hard_gamma_outer=+2.5,

        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        # gsmf_phi0_log10=-2.77,
        gsmf_phi0_log10=-2.56, # based on Agazie2023 mean value
        gsmf_phiz=-0.6,
        # gsmf_mchar0_log10=11.5, # ACD2024 table 1 (different than in phenom case)
        gsmf_mchar0_log10=10.9, # based on Agazie2023 mean value
        gsmf_mcharz=0.11,
        gsmf_alpha0=-1.21,
        gsmf_alphaz=-0.03,

        gpf_frac_norm_allq=0.033, # ACD2024 table 1 (different than in phenom case)
        gpf_malpha=0.0,
        gpf_qgamma=0.0,
        gpf_zbeta=1.0,
        gpf_max_frac=1.0,

        gmt_norm=0.5,           # [Gyr]
        gmt_malpha=0.0,
        gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
        gmt_zbeta=-0.5,

        # mmb_mamp_log10=8.7, # ACD2024 eq.A3 (different than in phenom case)
        mmb_mamp_log10=8.6, # based on Agazie2023 mean value
        mmb_plaw=1.10,          # average MM2013 and KH2013
        # mmb_scatter_dex=0.0, # ACD2024 eq.A3 (different than in phenom case)
        mmb_scatter_dex=0.32, # based on Agazie2023 mean value
        
        vt = 500, # (km/s) just some value inspired from ACD2024
        sigma0_over_m_times_t_age_per_1Gyr = 30 # (cm**2/g) just some value inspired from ACD2024
        # log10_vt = np.log10(500), # converting to log10
        # log10_sigma0_over_m_times_t_age_per_1Gyr = np.log10(30) # converting to log10
    )

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm_allq=params['gpf_frac_norm_allq'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_Chen2019(
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
            # scatter_dex=params['mmb_scatter'], # to match the param name from pspace
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam


    @classmethod
    def _init_hard(cls, sam, params):
        
        vt_kms = params['vt']
        vt = vt_kms * 1e5 # cgs
        sigma0_over_m_times_t_age_per_1Gyr = params['sigma0_over_m_times_t_age_per_1Gyr']
        import numpy as np
        # log10_vt_kms = params['log10_vt']
        # vt_kms = np.power(10, log10_vt_kms)
        # vt = vt_kms * 1e5 # cgs
        # log10_sigma0_over_m_times_t_age_per_1Gyr = params['log10_sigma0_over_m_times_t_age_per_1Gyr']
        # sigma0_over_m_times_t_age_per_1Gyr = np.power(10, log10_sigma0_over_m_times_t_age_per_1Gyr)
        
        import holodeck
        from holodeck.constants import YR, NWTG, SPLC
        from holodeck.galaxy_profiles import NFW
        from holodeck import utils
        from holodeck.librarian import DEF_NUM_FBINS
        from scipy.interpolate import interp1d
        from scipy.optimize import root_scalar
        from scipy.integrate import solve_ivp, quad
        import kalepy as kale

        Girelli_2020_instance = holodeck.host_relations.Girelli_2020()
        mstar = sam._mmbulge.mstar_from_mbh(sam.mtot)
        mhalo = Girelli_2020_instance.halo_mass(mstar, sam.redz)
        rho_s, rs = NFW._nfw_rho_rad(mhalo, sam.redz)

        ## FIND A BETTER WAY
        load_path = "/home/users/sti50/Codes/sidm_lambda0_yvalues_for_cvalues_1_to_5_for_500_points.npz"
        data = np.load(load_path)
        C_values, Lambda0_higher_values, Lambda0_lower_values, y_higher_values, y_lower_values = data["a1"], data["a2"], data["a3"], data["a4"], data["a5"]
        #####################
        interval_high = np.argmax(y_higher_values), np.where(y_higher_values>0)[0][-1] # Indices of the highest positive elements, and the last positive element
        # y has a max value, and then it monotonically decreases with C
        C_of_y_higher_values = interp1d(y_higher_values[interval_high[0]: interval_high[1] + 1], C_values[interval_high[0] : interval_high[1] + 1], fill_value="extrapolate") 

        # for v0 < vt case
        # calculate everything for a=0
        subintervals = np.linspace(1e-10, y_higher_values[interval_high[0]], 100)
        y_a0 = np.zeros((len(sam.mtot), len(sam.redz)))
        r1_a0 = np.zeros((len(sam.mtot), len(sam.redz)))
        v0_a0 = np.zeros((len(sam.mtot), len(sam.redz)))
        for i in range(rho_s.shape[0]):
            for j in range(rho_s.shape[1]):
                def v0_y_higher_values(y): # using eq. B4# v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_lower_values(y)) * NWTG * rho_s * r_s**2 # in cgs (cm/s)
                    v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_higher_values(y)) * NWTG * rho_s[i, j] * rs[i, j]**2 # in cgs (cm/s)
                    return np.sqrt(v0_squared)
                
                def eqn2(y):
                    # term_1 = sigma0_over_m_times_t_age_per_1Gyr * GYR *  vref * (vref/v0_y_higher_values(y))**(a_0 - 1) * rho_s[i, j] / (y * (1 + y)**2)
                    # equation above is equivalent of equation below
                    term_1 = sigma0_over_m_times_t_age_per_1Gyr * GYR *  v0_y_higher_values(y) * rho_s[i, j] / (y * (1 + y)**2) # t_age is expressed in Gyrs
                    return term_1 - 1
            
                count = 0 # To see how many times the equation becomes zero
                for k in range(len(subintervals) - 1):
                        lower, upper = subintervals[k], subintervals[k + 1]
                        if eqn2(lower) * eqn2(upper) < 0:  # Sign change indicates a root in (a, b)
                            result = root_scalar(eqn2, bracket=(lower, upper), method='bisect')
                            if result.converged:
                                y_a0[i, j] = result.root
                                r1_a0[i, j] = y_a0[i, j] * rs[i, j]
                                v0_a0[i, j] = v0_y_higher_values(y_a0[i, j])
                            count = count + 1
                            break
                if count == 0:
                    y_a0[i, j] = np.max(y_higher_values)
                    r1_a0[i, j] = y_a0[i, j] * rs[i, j]
                    v0_a0[i, j] = v0_y_higher_values(y_a0[i, j])
                    
        # for v0 > vt
        # calculate for a=4, but instead of eq.3 use sigma*v = sigma0 * (vt/v0)**4 * v0 in eq.2
        subintervals = np.linspace(1e-10, y_higher_values[interval_high[0]], 100)
        y_a4 = y_a0.copy()
        r1_a4 = r1_a0.copy()
        v0_a4 = v0_a0.copy()
        for i in range(rho_s.shape[0]):
            for j in range(rho_s.shape[1]):
                if(v0_a0[i, j] > vt):
                    def v0_y_higher_values(y): # using eq. B4# v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_lower_values(y)) * G * rho_s_in_si * r_s_in_si**2 # in SI (m/s)
                        v0_squared = 4 * np.pi * y / ((1 + y)**2 * C_of_y_higher_values(y)) * NWTG * rho_s[i, j] * rs[i, j]**2 # in cgs (cm/s)
                        return np.sqrt(v0_squared)
                    def eqn2(y):
                        # term_1 = (vt / v0_y_higher_values(y))**4 * v0_y_higher_values(y) * sigma0_over_m_times_t_age_per_1Gyr * GYR *  v0_y_higher_values(y) * rho_s[i, j] / (y * (1 + y)**2)
                        # equation above is equivalent of equation below
                        term_1 = vt**4 / v0_y_higher_values(y)**3 * sigma0_over_m_times_t_age_per_1Gyr * GYR *  rho_s[i, j] / (y * (1 + y)**2) # t_age is expressed in Gyrs
                        return term_1 - 1
                    count = 0 # To see how many times the equation becomes zero
                    for k in range(len(subintervals) - 1):
                            lower, upper = subintervals[k], subintervals[k + 1]
                            if eqn2(lower) * eqn2(upper) < 0:  # Sign change indicates a root in (a, b)
                                result = root_scalar(eqn2, bracket=(lower, upper), method='bisect')
                                if result.converged:
                                    y_a4[i, j] = result.root
                                    r1_a4[i, j] = y_a4[i, j] * rs[i, j]
                                    v0_a4[i, j] = v0_y_higher_values(y_a4[i, j])
                                count = count + 1

        Lambda0_of_y_higher_values = interp1d(y_higher_values[interval_high[0]: interval_high[1] + 1], Lambda0_higher_values[interval_high[0] : interval_high[1] + 1], fill_value="extrapolate") 
        # a=0
        Lambda0_higher_a0 = Lambda0_of_y_higher_values(r1_a0 / rs)
        rho_c_a0 = rho_s / ((r1_a0 / rs) * (1 + r1_a0 / rs)**2)
        C_a0 = 4 * np.pi * NWTG * rho_c_a0 * r1_a0**2 / v0_a0**2 
        # a=4
        Lambda0_higher_a4 = Lambda0_of_y_higher_values(r1_a4 / rs)
        rho_c_a4 = rho_s / ((r1_a4 / rs) * (1 + r1_a4 / rs)**2)
        C_a4 = 4 * np.pi * NWTG * rho_c_a4 * r1_a4**2 / v0_a4**2 
        # Define the differential equation system
        def ode(w, vector, C):
            Lambda, dLambda_dw = vector
            d2Lambda_dw2 = -C * np.exp(Lambda) - 2 * dLambda_dw / w
            return [dLambda_dw, d2Lambda_dw2]
        # for rt (transition radius)
        
        rt = np.zeros((len(sam.mtot), len(sam.redz)))
        # a=0
        r_sp_a0 = NWTG * np.broadcast_to(sam.mtot[:, np.newaxis], (91, 101)) / v0_a0**2 # below acd2024 eq.3 #cgs
        # a=4
        r_sp_a4 = NWTG * np.broadcast_to(sam.mtot[:, np.newaxis], (91, 101)) / v0_a4**2 # below acd2024 eq.3 #cgs
        interval = (1e-1 * PC, 1e6 * PC) # r values chosen by observation of Fig.2 (seeing approximately between what values does r_t occur)
        subintervals = np.linspace(interval[0], interval[1], 500)
        for i in range(len(sam.mtot)):
            for j in range(len(sam.redz)):
                if(v0_a0[i, j] < vt):
                    def v_r(r): # Equation B6
                        return v0_a0[i, j] * (7/11 + 4/11 * (r_sp_a0[i, j] / r)**0.5) # v_0 in cm/s , r_sp in cm, so we also must input r in cm
                    count = 0
                    def f(r): # This is what we want to become zero at r=r_t
                        return v_r(r) - vt
                    for k in range(len(subintervals) - 1):
                        a, b = subintervals[k], subintervals[k + 1]
                        # print(f(a), f(b))
                        if f(a) * f(b) < 0:  # Sign change indicates a root in (a, b)
                            result = root_scalar(f, bracket=(a, b), method='bisect')
                            if result.converged:
                                rt[i, j] = result.root
                                count = count + 1
        
        
        n_for_ivp_solver = 100
        rho_core_values_a0 = np.zeros((91, 101, n_for_ivp_solver))
        rho_core_values_a4 = np.zeros((91, 101, n_for_ivp_solver))
        r_values_a0 = np.zeros((91, 101, n_for_ivp_solver))
        r_values_a4 = np.zeros((91, 101, n_for_ivp_solver))

        for i in range(len(sam.mtot)):
            for j in range(len(sam.redz)):
                if vt < v0_a0[i, j]:
                    # only a=4 spike
                    # so, only need core profile with r1_a4 as boundary condition
                    solution_a4 = solve_ivp(ode, [1e-8, 1], [Lambda0_higher_a4[i, j], 0], args=(C_a4[i, j],), t_eval=np.linspace(1e-8, 1.0, 100))
                    Lambda_values_a4 = solution_a4.y[0]
                    w_values_a4 = solution_a4.t
                    rho_core_values_a4[i, j, :] = rho_c_a4[i, j] * np.exp(Lambda_values_a4) # cgs
                    r_values_a4[i, j, :] = r1_a4[i, j] * w_values_a4 # cm
                else:
                    # only need core profile with r1_04 as boundary condition
                    solution_a0 = solve_ivp(ode, [1e-8, 1], [Lambda0_higher_a0[i, j], 0], args=(C_a0[i, j],), t_eval=np.linspace(1e-8, 1.0, 100))
                    Lambda_values_a0 = solution_a0.y[0]  
                    w_values_a0 = solution_a0.t
                    rho_core_values_a0[i, j, :] = rho_c_a0[i, j] * np.exp(Lambda_values_a0) # cgs
                    r_values_a0[i, j, :] = r1_a0[i, j] * w_values_a0 # cm
        from scipy.special import erf
        u_1 = 11/4 * sam.mrat**(3/2) * (1 + sam.mrat)**(-3/2) # ACD eq.C3
        N1 = erf(u_1 / np.sqrt(2)) - np.sqrt(2 / np.pi) * u_1 * np.exp(-u_1**2 / 2) # ACD eq.C2
        u_2 = 11/4 * (1 + sam.mrat)**(-3/2) # ACD eq.C3
        N2 = erf(u_2 / np.sqrt(2)) - np.sqrt(2 / np.pi) * u_2 * np.exp(-u_2**2 / 2) # ACD eq.C2
        hard = hardening.Hard_SIDM(
                rt = rt,
                r_sp_a4 = r_sp_a4,
                v0_a0 = v0_a0,
                vt = vt,
                r_values_a0 = r_values_a0,
                r_values_a4 = r_values_a4,
                rho_core_values_a0 = rho_core_values_a0,
                rho_core_values_a4 = rho_core_values_a4,
                N1_array = N1,
                N2_array = N2,
                sepa_init=params['hard_sepa_init']*PC
                )
        return hard
class PS_Classic_SIDM_Uniform_GP_Improvement(_PS_Classic_SIDM_GP_Improvement):
    """
    SIDM case (inspired form ACD2024) with only vt, sigma0_over_m varying, uniform parameter space.

    """
    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        import numpy as np
        parameters = [
            PD_Uniform("vt", 1, 2000), # km/s (based on fig.4 but not excluding the grey region)
            PD_Uniform("sigma0_over_m_times_t_age_per_1Gyr", 0.01, 175), # cm**2/g (based on fig.4 but not excluding the grey region)
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )
###############################################


class _PS_Classic_3BS(_Param_Space):
    """Base class for classic phenomenological parameter space used in 15yr analysis.
    """

    DEFAULTS = dict(
        # Parameters are based on `sam-parameters.ipynb` fit to [Tomczak+2014]
        # gsmf_phi0=-2.77,
        gsmf_phi0_log10=-2.27, # chen2024 table 1
        # because in class _Param_Space from librarian/lib_tools.py lines 112, 181
        # gsmf_phi0 is replaced by gsmf_phi0_log10. so it should be replaced here too.
        gsmf_phiz=-0.6,
        gsmf_mchar0_log10=11.15, # chen2024 table 1
        gsmf_mcharz=0.11,
        gsmf_alpha0=-1.21,
        gsmf_alphaz=-0.03,

        gpf_frac_norm=0.033, # chen2024 table 1
        gpf_malpha=0.0,
        gpf_qgamma=0.0,
        gpf_zbeta=1.0,
        gpf_max_frac=1.0, # REVIEW LATER IF DOESN'T MATCH

        gmt_norm=0.5,           # [Gyr]
        gmt_malpha=0.0,
        gmt_qgamma=-1.0,        # Boylan-Kolchin+2008
        gmt_zbeta=-0.5,

        mmb_mamp_log10=8.65, # chen2024 table 1
        mmb_plaw=1.10,          # average MM2013 and KH2013
        mmb_scatter_dex=0.32, # chen2024 table 1
        
        gamma_3bs=0.08, # chen2024 best fit value foe e=0 case
        rho_pc_by_1e5_log10=1.3, # chen2024 best fit value foe e=0 case
        H=15 # chen2024 can be between 15-20
    )

    @classmethod
    def _init_sam(cls, sam_shape, params):
        gsmf = sams.GSMF_Schechter(
            phi0=params['gsmf_phi0_log10'],
            phiz=params['gsmf_phiz'],
            mchar0_log10=params['gsmf_mchar0_log10'],
            mcharz=params['gsmf_mcharz'],
            alpha0=params['gsmf_alpha0'],
            alphaz=params['gsmf_alphaz'],
        )
        gpf = sams.GPF_Power_Law(
            frac_norm=params['gpf_frac_norm'],
            malpha=params['gpf_malpha'],
            qgamma=params['gpf_qgamma'],
            zbeta=params['gpf_zbeta'],
            max_frac=params['gpf_max_frac'],
        )
        gmt = sams.GMT_Power_Law(
            time_norm=params['gmt_norm']*GYR,
            malpha=params['gmt_malpha'],
            qgamma=params['gmt_qgamma'],
            zbeta=params['gmt_zbeta'],
        )
        mmbulge = host_relations.MMBulge_KH2013(
            mamp_log10=params['mmb_mamp_log10'],
            mplaw=params['mmb_plaw'],
            scatter_dex=params['mmb_scatter_dex'],
            # scatter_dex=params['mmb_scatter'], # to match the param name from pspace
        )

        sam = sams.Semi_Analytic_Model(
            gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
            shape=sam_shape,
        )
        return sam

    @classmethod
    def _init_hard(cls, sam, params):
        
        gamma = params['gamma_3bs']
        rho_pc_by_1e5_log10 = params['rho_pc_by_1e5_log10']
        
        import numpy as np
        from holodeck.constants import NWTG, MSOL, PC
        
        rho_pc_by_1e5_log10 = 1.3
        rho_pc = 10**rho_pc_by_1e5_log10 * 1e5 * MSOL / PC**3
        r_i = (2 * sam.mtot / (rho_pc * PC**gamma * 4 * np.pi / (3 - gamma)))**(1 / (3 - gamma))

        rho_i_1d_array = rho_pc * (r_i / PC)**(-gamma) # chen2024 eq.6 (shape (91,))
        sigma_i_1d_array = np.sqrt(NWTG * sam.mtot / r_i) # chen2024 after eq.6 (shape (91,))

        hard = hardening.Hard_3BS(
            rho_i_1d_array = rho_i_1d_array,
            sigma_i_1d_array = sigma_i_1d_array,
            H = params['H_3bs'],
        )
        return hard


class PS_Classic_3BS_Uniform(_PS_Classic_3BS):
    """
    3 Body scattering case (inspired form chen2024) with only gamma_3bs, rho_pc_by_1e5_log10, and H_3bs uniform parameter space.

    """

    def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None):
        parameters = [
            PD_Uniform("gamma_3bs", 0.0, 3.0), # based on chen2024 fig3 left
            PD_Uniform("rho_pc_by_1e5_log10", -4.0, 4.0), # based on chen2024 fig3 left
            PD_Uniform("H_3bs", 15, 20),
        ]

        super().__init__(
            parameters,
            log=log, nsamples=nsamples, sam_shape=sam_shape, seed=seed,
        )

_param_spaces_dict = {
    "PS_Test": PS_Test,
    "PS_Classic_Phenom_Uniform": PS_Classic_Phenom_Uniform,    # PS_Uniform_09B
    "PS_Classic_Phenom_Astro_Extended": PS_Classic_Phenom_Astro_Extended,
    "PS_Classic_GWOnly_Uniform": PS_Classic_GWOnly_Uniform,
    "PS_Classic_GWOnly_Astro_Extended": PS_Classic_GWOnly_Astro_Extended,
    "PS_Classic_CDM_Uniform": PS_Classic_CDM_Uniform,
    "PS_Classic_SIDM_Uniform": PS_Classic_SIDM_Uniform,
    "PS_Classic_SIDM_Astro": PS_Classic_SIDM_Astro,
    "PS_Classic_SIDM_Astro_Extended": PS_Classic_SIDM_Astro_Extended,
    "PS_Classic_3BS_Uniform": PS_Classic_3BS_Uniform,
    "PS_Classic_SIDM_Uniform_GP_Improvement": PS_Classic_SIDM_Uniform_GP_Improvement,
    "PS_Classic_SIDM_Astro_Extended_Version2": PS_Classic_SIDM_Astro_Extended_Version2,
    "PS_Classic_SIDM_Astro_Uniform_Extended_Version2": PS_Classic_SIDM_Astro_Uniform_Extended_Version2,
}

