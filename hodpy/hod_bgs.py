#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad
from scipy.optimize import minimize, root


from hodpy.stellar_mass_function import StellarMassFunctionTargetBGS
from hodpy.mass_function import MassFunctionPino
from hodpy.cosmology import CosmologyPino
from hodpy.kmass_correction import JPAS_KMCorrection
from hodpy import spline
from hodpy.hod import HOD
from hodpy import lookup


def M_function(log_mass, L_s, M_t, a_m):
    """
    Function describing how HOD parameters Mmin and M1 vary with log(stellar mass).
    This function returns log(stellar mass) as a function of halo mass, and has the same
    functional form as Eq. 11 from Zehavi 2011.

    Args:
        log_mass: array of the log10 of halo mass (Msun/h)
        L_s:      median stellar mass of central galaxies (Msun)
        M_t:      transition halo mass (Msun/h)
        a_m:      high-mass power law index
    Returns:
        array of log(stellar mass) [Msun]
    """
    s_mass = L_s*(10**log_mass/M_t)**a_m * np.exp(1-(M_t/10**log_mass))
    log_stell_mass = np.log10(s_mass)
    return log_stell_mass


def sigma_function(log_stell_mass, s_faint, s_bright, M_step, width):
    """
    Smooth step function describing how HOD parameter sigma_logM varies
    as a function of log(stellar mass)

    Args:
        log_stell_mass: array of stellar magnitudes
        s_faint:   sigma_logM for faint samples
        s_bright:  sigma_logM for bright samples
        M_step:    position of step
        width:     sets the width of step
    Returns:
        array of sigma_logM
    """
    sigma = s_faint + (s_bright-s_faint) / (1.+np.exp((M_step - log_stell_mass)*width))

    # Add possibility of setting sigma to 0 (if s_faint=s_bright=0)
    # In that case, set it to a small non-zero value (to avoid problems later)
    sigma[sigma == 0] = 1e-6

    return sigma


class HOD_BGS(HOD):
    """
    HOD class containing the HODs used to create the mock catalogue described in Smith et al. 2017,
    adapted to use of log(stellar mass), and J-PAS parameters

    args:
        [mass_function]: hodpy.MassFunction object, the mass functino of the simulation (default is MassFunctionPino)
        [cosmology]:     hodpy.Cosmology object, the cosmology of the simulation (default is CosmologyPino)
        [mag_faint]:     faint apparent magnitude limit (default is 20.0)
        [kcorr]:         hodpy.KMCorrection object, the k-mass-correction (default is JPAS_KMCorrection)
        [hod_param_file]: location of file which contains HOD parameters
        [slide_file]:    location of file which contains 'slide' factors for evolving HODs. Will be created
                         automatically if the file doesn't already exist
        [central_lookup_file]: location of lookup file of central log(stellar masses). Will be created if the file
                               doesn't already exist
        [satellite_lookup_file]: location of lookup file of satellite log(stellar masses). Will be created if the file
                                 doesn't already exist
        [target_smf_file]: location of file containing the target stellar mass function. Will be created if the
                          file doesn't already exist
        [smf_param_file]: location of file which contains the parameters of the evolving miniJPAS stellar mass function
        [replace_central_lookup]: if set to True, will replace central_lookup_file even if the file exists
        [replace_satellite_lookup]: if set to True, will replace satellite_lookup_file even if the file exists
    """

    def __init__(self, mass_function=MassFunctionPino(), cosmology=CosmologyPino(),
                 mag_faint=20.0, kcorr=JPAS_KMCorrection(CosmologyPino()),
                 hod_param_file=lookup.bgs_hod_parameters,
                 slide_file=lookup.bgs_hod_slide_factors,
                 central_lookup_file=lookup.central_lookup_file,
                 satellite_lookup_file=lookup.satellite_lookup_file,
                 target_smf_file=lookup.target_smf,
                 smf_param_file=lookup.smf_params,
                 replace_central_lookup=False, replace_satellite_lookup=False):

        self.Mmin_Ls, self.Mmin_Mt, self.Mmin_am, self.M1_Ls, self.M1_Mt, self.M1_am, \
            self.M0_A, self.M0_B, self.alpha_A, self.alpha_B, self.alpha_C, self.sigma_A, \
            self.sigma_B, self.sigma_C, self.sigma_D  = lookup.read_hod_param_file(hod_param_file)

        self.mf = mass_function
        self.cosmo = cosmology
        self.smf = StellarMassFunctionTargetBGS(target_smf_file, smf_param_file,
                                              HOD_BGS_Simple(hod_param_file))
        self.kcorr = kcorr
        self.mag_faint = mag_faint

        self.__logMmin_interpolator = \
            self.__initialize_mass_interpolator(self.Mmin_Ls, self.Mmin_Mt,
                                                self.Mmin_am)
        self.__logM1_interpolator = \
            self.__initialize_mass_interpolator(self.M1_Ls, self.M1_Mt, self.M1_am)

        self.__slide_interpolator = \
            self.__initialize_slide_factor_interpolator(slide_file)

        self.__central_interpolator = \
            self.__initialize_central_interpolator(central_lookup_file, replace_central_lookup)

        self.__satellite_interpolator = \
            self.__initialize_satellite_interpolator(satellite_lookup_file, replace_satellite_lookup)

    def __integration_function(self, logM, log_sm, z, f):
        # function to integrate (number density of haloes * number of galaxies from HOD)

        # mean number of galaxies per halo
        N_gal = self.number_galaxies_mean(np.array([logM,]),log_sm,z,f)[0]
        # number density of haloes
        n_halo = self.mf.number_density(np.array([logM,]), z)

        return (N_gal * n_halo)[0]


    def get_n_HOD(self, log_stell_mass, redshift, f):
        """
        Returns the number density of galaxies predicted by the HOD. This is evaluated from
        integrating the halo mass function multiplied by the HOD. The arguments must be
        arrays of length 1.
        Args:
            log_stell_mass: log of the stellar mass threshold [Msun]
            redshift:  redshifts
            f:         evolution parameter
        Returns:
            number density
        """
        return quad(self.__integration_function, 10, 18,
                    args=(log_stell_mass, redshift, f),
                    epsrel=1e-16)[0]


    def __root_function(self, f, log_sm, z):
        # function used in root finding procedure for calculating the slide factors
        log_sm = np.array([log_sm,])
        z = np.array([z,])
        log_n_targ = np.log10(self.smf.Phi_cumulative(log_sm, z))
        log_n_hod = np.log10(self.get_n_HOD(log_sm, z, f))
        return log_n_targ - log_n_hod

    def __slide_factor_calc(self, log_sm, z):
        """
        Function that calculates the evolution parameter ('slide factor')
        for a single (log(stellar mass), redshift) point.
        Will be used below for initializing the interpolator.
        """
        f0 = 1.0 #initial guess
        x = root(self.__root_function, f0,
                 args=(log_sm, z))
        # sometimes root finding fails with this initial guess
        # make f0 smaller and try again
        while x["x"][0]==f0 and f0>0:
            f0 -= 0.1
            # print("Trying f0 = %.1f"%f0)
            x = root(self.__root_function, f0,
                     args=(log_sm, z))

        return x["x"][0]

    def __initialize_slide_factor_interpolator(self, slide_file):
        # creates a RegularGridInterpolator object used for finding
        # the 'slide factor' as a function of mag and z

        log_stell_masses = np.arange(5, 13, 0.1)
        redshifts = np.arange(0, 1.01, 0.05)

        try:
            # try to read file
            factors = np.loadtxt(slide_file)

        except:
            # file doesn't exist - calculate factors
            # Note: this is very slow!!
            # Import needed for parallel calculation
            from pathos.pools import ProcessPool

            print("Calculating evolution parameters")

            factors = np.zeros((len(log_stell_masses),len(redshifts)))

            # loop through magnitudes and redshifts. At each, find f required to get target LF
            for i in range(len(redshifts)):
                print("z = %.2f" % redshifts[i])
                with ProcessPool() as p:
                    factors[:, i] = p.map(lambda x: self.__slide_factor_calc(x, redshifts[i]), log_stell_masses)
            np.savetxt(slide_file, factors)

        return RegularGridInterpolator((log_stell_masses, redshifts), factors,
                                       bounds_error=False, fill_value=None)


    def __initialize_mass_interpolator(self, L_s, M_t, a_m):
        # creates a RegularGridInterpolator object used for finding
        # the HOD parameters Mmin or M1 (at z=zref) as a function of log_stell_mass

        log_mass = np.arange(11, 18, 0.001)

        log_stell_masses = M_function(log_mass, L_s, M_t, a_m)
        return RegularGridInterpolator((log_stell_masses,), log_mass,
                                       bounds_error=False, fill_value=None)

    def __initialize_central_interpolator(self, central_lookup_file, replace_central_lookup=False):
        # creates a RegularGridInterpolator object used for finding
        # the log(stellar mass) of central galaxies as a function of log_mass,
        # z, and random number x from spline kernel distribution

        # arrays of mass, x, redshift, and 3d array of magnitudes
        # x is the scatter in the central luminosity from the mean
        log_masses = np.arange(10, 18, 0.02)
        redshifts = np.arange(0, 1, 0.02)
        xs = np.arange(-3.5, 3.501, 0.02)
        log_stell_masses = np.zeros((len(log_masses), len(redshifts), len(xs)))

        try:
            if replace_central_lookup: raise IOError

            # try to read 3d array of magnitudes from file
            log_stell_masses = np.load(central_lookup_file)

            if log_stell_masses.shape != (len(log_masses), len(redshifts), len(xs)):
                raise ValueError("Central lookup table has unexpected shape")

        except IOError:
            # file doesn't exist - fill in array of log(stellar masses)
            print("Generating lookup table of central galaxy stellar masses")
            log_sm = np.arange(5, 13, 0.01)[::-1] # invert to make 'searchsorted' work properly later
            arr_ones = np.ones(len(log_sm), dtype="f")
            for i in range(len(log_masses)):
                for j in range(len(redshifts)):

                    x = np.sqrt(2) * (log_masses[i]-np.log10(self.Mmin(log_sm, arr_ones*redshifts[j]))) / self.sigma_logM(log_sm, arr_ones*redshifts[j])

                    if x[-1] < 3.5: continue

                    # find this in the array xs
                    idx = np.searchsorted(x, xs)

                    # interpolate
                    f = (xs - x[idx-1]) / (x[idx] - x[idx-1])
                    log_stell_masses[i, j, :] = log_sm[idx-1] + f*(log_sm[idx] - log_sm[idx-1])
            print("Saving lookup table to file")
            np.save(central_lookup_file, log_stell_masses)

        # create RegularGridInterpolator object
        return RegularGridInterpolator((log_masses, redshifts, xs),
                              log_stell_masses, bounds_error=False, fill_value=None)


    def __initialize_satellite_interpolator(self, satellite_lookup_file, replace_satellite_file=False):
        # creates a RegularGridInterpolator object used for finding
        # the log(stellar mass) of satellite galaxies as a function of log_mass,
        # z, and random number log_x (x is uniform random between 0 and 1)

        # arrays of mass, x, redshift, and 3d array of log(stell mass)
        # x is the ratio of Nsat(log_stell_mass,mass)/Nsat(log_stell_mass_faint,mass)
        log_masses = np.arange(10, 18, 0.02)
        # For some (unknown) reason, I get nan's for z=0, so avoid this value
        # redshifts = np.arange(0, 1, 0.02)
        redshifts = np.arange(0.01, 1, 0.02) 
        log_xs = np.arange(-12, 0.01, 0.05)
        log_stell_masses = np.zeros((len(log_masses), len(redshifts), len(log_xs)))

        try:
            if replace_satellite_file: raise IOError

            # try to read 3d array of stellar masses from file
            log_stell_masses = np.load(satellite_lookup_file)

            if log_stell_masses.shape!=(len(log_masses), len(redshifts), len(log_xs)):
                raise ValueError("Satellite lookup table has unexpected shape")

        except IOError:
            # file doesn't exist - fill in array of log(stellar masses)
            print("Generating lookup table of satellite galaxy stellar masses")

            log_sm = np.arange(5, 13, 0.01)[::-1] # invert to make 'searchsorted' work properly later

            log_sm_faint = self.kcorr.log_stellar_mass_faint(redshifts, self.mag_faint)
            arr_ones = np.ones(len(log_sm))
            for i in range(len(log_masses)):
                for j in range(len(redshifts)):
                    Nsat = self.number_satellites_mean(arr_ones*log_masses[i], log_sm,
                                                   arr_ones*redshifts[j])
                    Nsat_faint = self.number_satellites_mean(arr_ones*log_masses[i],
                                   arr_ones*log_sm_faint[j],arr_ones*redshifts[j])

                    log_x = np.log10(Nsat) - np.log10(Nsat_faint)

                    if log_x[-1] == -np.inf: continue

                    # find this in the array log_xs
                    idx = np.searchsorted(log_x, log_xs)

                    # interpolate
                    f = (log_xs - log_x[idx-1]) / (log_x[idx] - log_x[idx-1])
                    log_stell_masses[i, j, :] = log_sm[idx-1] + f*(log_sm[idx] - log_sm[idx-1])

                    # Deal with NaN values
                    # if NaN for small x but not large x, replace all
                    # NaN values with faintest mag
                    idx = np.isnan(log_stell_masses[i,j,:])
                    num_nan = np.count_nonzero(idx)
                    if num_nan < len(idx) and num_nan>0:
                        log_stell_masses[i,j,idx] = \
                            log_stell_masses[i,j, np.where(idx)[0][-1]+1]
                    # if previous mass bin contains all NaN, copy current
                    # mass bin
                    if i>0 and np.count_nonzero(np.isnan(log_stell_masses[i-1,j,:]))\
                                                ==len(log_stell_masses[i,j,:]):
                        log_stell_masses[i-1,j,:] = log_stell_masses[i,j,:]
                    # if all NaN and j>0, copy prev j to here
                    if j>0 and np.count_nonzero(np.isnan(log_stell_masses[i,j,:]))\
                                                ==len(log_stell_masses[i,j,:]):
                        log_stell_masses[i,j,:] = log_stell_masses[i,j-1,:]

            print("Saving lookup table to file")
            np.save(satellite_lookup_file, log_stell_masses)

        # create RegularGridInterpolator object
        return RegularGridInterpolator((log_masses, redshifts, log_xs),
                              log_stell_masses, bounds_error=False, fill_value=None)


    def slide_factor(self, log_stell_mass, redshift):
        """
        Factor by when the HOD mass parameters (ie Mmin, M0 and M1) must
        be multiplied by in order to produce the number density of
        galaxies as specified by the target luminosity function
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of slide factors
        """
        points = np.array(list(zip(log_stell_mass, redshift)))
        return self.__slide_interpolator(points)


    def Mmin(self, log_stell_mass, redshift, f=None):
        """
        HOD parameter Mmin, which is the mass at which a halo has a 50%
        change of containing a central galaxy brighter than the stellar mass
        threshold
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of Mmin
        """
        # use target SMF to convert stellar mass to number density
        n = self.smf.Phi_cumulative(log_stell_mass, redshift)

        # find stellar mass at z=zref which corresponds to the same number density
        log_stell_mass_zref = self.smf.log_stell_mass(n, np.ones(len(n))*self.smf.zref)

        # find Mmin
        Mmin = 10**self.__logMmin_interpolator(log_stell_mass_zref)
        # use slide factor to evolve Mmin
        if not f is None:
            return Mmin * f
        else:
            return Mmin * self.slide_factor(log_stell_mass, redshift)


    def M1(self, log_stell_mass, redshift, f=None):
        """
        HOD parameter M1, which is the mass at which a halo contains an
        average of 1 satellite brighter than the stellar mass threshold
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of M1
        """
        # use target SMF to convert stellar mass to number density
        n = self.smf.Phi_cumulative(log_stell_mass, redshift)

        # find stellar mass at z=zref which corresponds to the same number density
        log_stell_mass_zref = self.smf.log_stell_mass(n, np.ones(len(n))*self.smf.zref)

        # find Mmin
        M1 = 10**self.__logM1_interpolator(log_stell_mass_zref)
        # use slide factor to evolve Mmin
        if not f is None:
            return M1 * f
        else:
            return M1 * self.slide_factor(log_stell_mass, redshift)


    def M0(self, log_stell_mass, redshift, f=None):
        """
        HOD parameter M0, which sets the cut-off mass scale for satellites
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of M0
        """
        # Add option for 'simplified' model with M0=Mmin.
        # This can be defined in [hod_param_file] setting (M0_A, M0_B) = (0, 0)
        if self.M0_A == 0 and self.M0_B == 0:
            return self.Mmin(log_stell_mass, redshift, f)
        else:
            # use target SMF to convert stellar mass to number density
            n = self.smf.Phi_cumulative(log_stell_mass, redshift)

            # find stellar mass at z=zref which corresponds to the same number density
            log_stell_mass_zref = self.smf.log_stell_mass(n, np.ones(len(n))*self.smf.zref)

            # find M0
            M0 = 10**(self.M0_A*log_stell_mass_zref + self.M0_B)

            # use slide factor to evolve M0
            if not f is None:
                return M0 * f
            else:
                return M0 * self.slide_factor(log_stell_mass, redshift)


    def alpha(self, log_stell_mass, redshift):
        """
        HOD parameter alpha, which sets the slope of the power law for
        satellites
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of alpha
        """
        # use target SMF to convert stellar mass to number density
        n = self.smf.Phi_cumulative(log_stell_mass, redshift)

        # find stellar mass at z=zref which corresponds to the same number density
        log_stell_mass_zref = self.smf.log_stell_mass(n, np.ones(len(n))*self.smf.zref)

        # find alpha
        a = np.log10(self.alpha_C + (self.alpha_A*log_stell_mass_zref)**self.alpha_B)

        # alpha is kept fixed with redshift
        return a


    def sigma_logM(self, log_stell_mass, redshift):
        """
        HOD parameter sigma_logM, which sets the amount of scatter in
        the stellar mass of central galaxies
        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of sigma_logM
        """
        # use target SMF to convert stellar mass to number density
        n = self.smf.Phi_cumulative(log_stell_mass, redshift)

        # find stellar mass at z=zref which corresponds to the same number density
        log_stell_mass_zref = self.smf.log_stell_mass(n, np.ones(len(n))*self.smf.zref)

        # find sigma_logM
        sigma = sigma_function(log_stell_mass_zref, self.sigma_A, self.sigma_B, self.sigma_C, self.sigma_D)

        # sigma_logM is kept fixed with redshift
        return sigma


    def number_centrals_mean(self, log_mass, log_stell_mass, redshift, f=None):
        """
        Average number of central galaxies in each halo brighter than
        some stellar mass threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            log_stell_mass: log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of mean number of central galaxies
        """

        # use pseudo gaussian spline kernel
        return spline.cumulative_spline_kernel(log_mass,
                mean=np.log10(self.Mmin(log_stell_mass, redshift, f)),
                sig=self.sigma_logM(log_stell_mass, redshift)/np.sqrt(2))


    def number_satellites_mean(self, log_mass, log_stell_mass, redshift, f=None):
        """
        Average number of satellite galaxies in each halo brighter than
        some stellar mass threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            log_stell_mass: log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of mean number of satellite galaxies
        """
        num_cent = self.number_centrals_mean(log_mass, log_stell_mass, redshift, f)

        num_sat = num_cent * ((10**log_mass - self.M0(log_stell_mass, redshift, f))/\
            self.M1(log_stell_mass, redshift, f))**self.alpha(log_stell_mass, redshift)

        num_sat[np.where(np.isnan(num_sat))[0]] = 0

        return num_sat


    def number_galaxies_mean(self, log_mass, log_stell_mass, redshift, f=None):
        """
        Average total number of galaxies in each halo brighter than
        some stellar mass threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            log_stell_mass: log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of mean number of galaxies
        """
        return self.number_centrals_mean(log_mass, log_stell_mass, redshift, f) + \
            self.number_satellites_mean(log_mass, log_stell_mass, redshift, f)


    def get_number_satellites(self, log_mass, redshift):
        """
        Randomly draw the number of satellite galaxies in each halo,
        brighter than mag_faint, from a Poisson distribution
        Args:
            log_mass: array of the log10 of halo mass (Msun/h)
            redshift: array of halo redshifts
        Returns:
            array of number of satellite galaxies
        """
        # Get stellar mass threshold corresponding to mag_faint
        log_sm_faint = self.kcorr.log_stellar_mass_faint(redshift, self.mag_faint)

        # mean number of satellites in each halo brighter than the
        # faint magnitude threshold
        number_mean = self.number_satellites_mean(log_mass, log_sm_faint, redshift)

        # draw random number from Poisson distribution
        return np.random.poisson(number_mean)


    def get_lstellmass_centrals(self, log_mass, redshift):
        """
        Use the HODs to draw a random log(stellar mass) for each central galaxy
        Args:
            log_mass: array of the log10 of halo mass (Msun/h)
            redshift: array of halo redshifts
        Returns:
            array of central galaxy log(stellar mass)
        """
        # random number from spline kernel distribution
        x = spline.random(size=len(log_mass))

        # return corresponding central log(stellar masses)
        points = np.array(list(zip(log_mass, redshift, x)))
        return self.__central_interpolator(points)


    def get_lstellmass_satellites(self, log_mass, number_satellites, redshift):
        """
        Use the HODs to draw a random log(stellar mass) for each satellite galaxy
        Args:
            log_mass:          array of the log10 of halo mass (Msun/h)
            redshift:          array of halo redshifts
            number_satellites: array of number of satellites in each halo
        Returns:
            array of the index of each galaxy's halo in the input arrays
            array of satellite galaxy log(stellar mass)
        """
        # create arrays of log_mass, redshift and halo_index for galaxies
        halo_index = np.arange(len(log_mass))
        halo_index = np.repeat(halo_index, number_satellites)
        log_mass_satellite = np.repeat(log_mass, number_satellites)
        redshift_satellite = np.repeat(redshift, number_satellites)

        # uniform random number x
        # x=1 corresponds to mag_faint
        # mag -> -infinity as x -> 0
        log_x = np.log10(np.random.rand(len(log_mass_satellite)))

        # find corresponding satellite log(stellar masses)
        points = np.array(list(zip(log_mass_satellite, redshift_satellite,
                                   log_x)))
        return halo_index, self.__satellite_interpolator(points)



class HOD_BGS_Simple(HOD):
    """
    Simplified version of the class HOD_BGS.
    This class should only be used when calculating the target stellar mass function.
    TODO: a good fraction of the code here is repeated from above, we should use
          sub-classing in a more efficient way to define these two classes!
    """

    def __init__(self, hod_param_file):

        self.Mmin_Ls, self.Mmin_Mt, self.Mmin_am, self.M1_Ls, self.M1_Mt, self.M1_am, \
            self.M0_A, self.M0_B, self.alpha_A, self.alpha_B, self.alpha_C, self.sigma_A, \
            self.sigma_B, self.sigma_C, self.sigma_D  = lookup.read_hod_param_file(hod_param_file)

        self.mf = MassFunctionPino()

        self.__logMmin_interpolator = \
            self.__initialize_mass_interpolator(self.Mmin_Ls, self.Mmin_Mt,
                                               self.Mmin_am)
        self.__logM1_interpolator = \
            self.__initialize_mass_interpolator(self.M1_Ls, self.M1_Mt, self.M1_am)


    def __integration_function(self, logM, log_sm, z, f):
        # function to integrate (number density of haloes * number of galaxies from HOD)

        # mean number of galaxies per halo
        N_gal = self.number_galaxies_mean(np.array([logM,]),log_sm,z,f)[0]
        # number density of haloes
        n_halo = self.mf.number_density(np.array([logM,]), z)

        return (N_gal * n_halo)[0]


    def get_n_HOD(self, log_stell_mass, redshift, f):
        """
        Returns the number density of galaxies predicted by the HOD. This is evaluated from
        integrating the halo mass function multiplied by the HOD. The arguments must be
        arrays of length 1.
        Args:
            log_stell_mass: log of the stellar mass threshold [Msun]
            redshift:  redshifts
            f:         evolution parameter
        Returns:
            number density
        """
        return quad(self.__integration_function, 10, 18,
                    args=(log_stell_mass, redshift, f),
                    epsrel=1e-16)[0]


    def __initialize_mass_interpolator(self, L_s, M_t, a_m):
        # creates a RegularGridInterpolator object used for finding
        # the HOD parameters Mmin or M1 (at z=zref) as a function of log_stell_mass

        log_mass = np.arange(11, 18, 0.001)
        log_stell_masses = M_function(log_mass, L_s, M_t, a_m)
        return RegularGridInterpolator((log_stell_masses,), log_mass,
                                       bounds_error=False, fill_value=None)


    def Mmin(self, log_stell_mass, redshift, f=1.0):
        """
        HOD parameter Mmin, which is the mass at which a halo has a 50%
        change of containing a central galaxy brighter than the stellar mass
        threshold
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of Mmin
        """
        return 10**self.__logMmin_interpolator(log_stell_mass) * f


    def M1(self, log_stell_mass, redshift, f=1.0):
        """
        HOD parameter M1, which is the mass at which a halo contains an
        average of 1 satellite brighter than the stellar mass threshold
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of M1
        """
        return 10**self.__logM1_interpolator(log_stell_mass) * f


    def M0(self, log_stell_mass, redshift, f=1.0):
        """
        HOD parameter M0, which sets the cut-off mass scale for satellites
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of M0
        """

        # Add option for 'simplified' model with M0=Mmin.
        # This can be defined in [hod_param_file] setting (M0_A, M0_B) = (0, 0)
        if self.M0_A == 0 and self.M0_B == 0:
            return self.Mmin(log_stell_mass, redshift, f)
        else:
            return 10**(self.M0_A*log_stell_mass + self.M0_B) * f


    def alpha(self, log_stell_mass, redshift):
        """
        HOD parameter alpha, which sets the slope of the power law for
        satellites
        Args:
            log_stell_mass: array of log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of alpha
        """

        return np.log10(self.alpha_C + (self.alpha_A*log_stell_mass)**self.alpha_B)


    def sigma_logM(self, log_stell_mass, redshift):
        """
        HOD parameter sigma_logM, which sets the amount of scatter in
        the stellar mass of central galaxies
        Args:
            magnitude: array of absolute magnitude threshold
            redshift:  array of halo redshifts
        Returns:
            array of sigma_logM
        """
        sigma = sigma_function(log_stell_mass, self.sigma_A, self.sigma_B, self.sigma_C, self.sigma_D)

        return sigma


    def number_centrals_mean(self, log_mass, log_stell_mass, redshift, f=None):
        """
        Average number of central galaxies in each halo brighter than
        some stellar mass threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            log_stell_mass: log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of mean number of central galaxies
        """

        # use pseudo gaussian spline kernel
        return spline.cumulative_spline_kernel(log_mass,
                mean=np.log10(self.Mmin(log_stell_mass, redshift, f)),
                sig=self.sigma_logM(log_stell_mass, redshift)/np.sqrt(2))


    def number_satellites_mean(self, log_mass, log_stell_mass, redshift, f=None):
        """
        Average number of satellite galaxies in each halo brighter than
        some stellar mass threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            log_stell_mass: log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of mean number of satellite galaxies
        """
        num_cent = self.number_centrals_mean(log_mass, log_stell_mass, redshift, f)

        num_sat = num_cent * ((10**log_mass - self.M0(log_stell_mass, redshift, f))/\
            self.M1(log_stell_mass, redshift, f))**self.alpha(log_stell_mass, redshift)

        num_sat[np.where(np.isnan(num_sat))[0]] = 0

        return num_sat


    def number_galaxies_mean(self, log_mass, log_stell_mass, redshift, f=None):
        """
        Average total number of galaxies in each halo brighter than
        some stellar mass threshold
        Args:
            log_mass:  array of the log10 of halo mass (Msun/h)
            log_stell_mass: log of the stellar mass threshold [Msun]
            redshift:  array of halo redshifts
        Returns:
            array of mean number of galaxies
        """
        return self.number_centrals_mean(log_mass, log_stell_mass, redshift, f) + \
            self.number_satellites_mean(log_mass, log_stell_mass, redshift, f)
