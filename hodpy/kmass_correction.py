#! /usr/bin/env python
import numpy as np
from scipy.interpolate import RegularGridInterpolator, splrep, splev

from hodpy import lookup


class KMCorrection(object):
    """
    K-Mass-correction base class
    """

    def apparent_magnitude(self, log_stellar_mass, redshift):
        pass

    def log_stellar_mass(self, apparent_magnitude, redshift):
        pass

    def log_stellar_mass_faint(self, redshift):
        pass


class JPAS_KMCorrection(KMCorrection):
    """
    Colour-dependent polynomial fit to the K-Mass-correction,
    used to convert between JPAS i-band apparent magnitudes,
    and (log of) stellar masses.
    The colour used to define the K-Mass corrections is the
    (u-r) rest-frame colour (from BaySeaGal analysis).

    Args:
        k_corr_file: file of polynomial coefficients for each colour bin
        cosmology: object of type hodpy.Cosmology
        [z0]: reference redshift. Default value is z0=0.4
        [cubic_interpolation]: if set to True, will use cubic spline interpolation.
                               Default value is False (linear interpolation).
    """
    def __init__(self, cosmology, k_corr_file=lookup.kcorr_file, z0=0.4, cubic_interpolation=False):

        # read file of parameters of polynomial fit to k-correction
        cmin, cmax, A, B, C, D, E, cmed = \
            np.loadtxt(k_corr_file, unpack=True)

        self.Nc = len(cmin) # Number of colour bins
        self.z0 = z0 # reference redshift
        self.cubic = cubic_interpolation

        # Polynomial fit parameters
        if cubic_interpolation:
            # cubic spline interpolation
            self.__A_interpolator = self.__initialize_parameter_interpolator_spline(A,cmed)
            self.__B_interpolator = self.__initialize_parameter_interpolator_spline(B,cmed)
            self.__C_interpolator = self.__initialize_parameter_interpolator_spline(C,cmed)
            self.__D_interpolator = self.__initialize_parameter_interpolator_spline(D,cmed)
            self.__E_interpolator = self.__initialize_parameter_interpolator_spline(E,cmed)
        else:
            # linear interpolation
            self.__A_interpolator = self.__initialize_parameter_interpolator(A,cmed)
            self.__B_interpolator = self.__initialize_parameter_interpolator(B,cmed)
            self.__C_interpolator = self.__initialize_parameter_interpolator(C,cmed)
            self.__D_interpolator = self.__initialize_parameter_interpolator(D,cmed)
            self.__E_interpolator = self.__initialize_parameter_interpolator(E,cmed)

        self.colour_min = np.min(cmed)
        self.colour_max = np.max(cmed)
        self.colour_med = cmed

        # Will use the following in 'log_stellar_mass_faint'
        self.colour_bluest = np.min(cmin)
        self.colour_reddest = np.max(cmax)

        self.cosmo = cosmology

        # Linear extrapolation
        self.__X_interpolator = lambda x: None
        self.__Y_interpolator = lambda x: None
        self.__X_interpolator, self.__Y_interpolator = \
                                 self.__initialize_line_interpolators()


    def __initialize_parameter_interpolator(self, parameter, median_colour):
        # interpolated polynomial coefficient as a function of colour
        return RegularGridInterpolator((median_colour,), parameter,
                                       bounds_error=False, fill_value=None)


    def __initialize_parameter_interpolator_spline(self, parameter, median_colour):
        # interpolated polynomial coefficient as a function of colour
        tck = splrep(median_colour, parameter)
        return tck


    def __initialize_line_interpolators(self):
        # linear coefficients for z>0.8
        X = np.zeros(self.Nc)
        Y = np.zeros(self.Nc)
        # find X, Y at each colour

        # For miniJPAS, will fit the linear extrapolation
        # to the range [0.6, 0.8]
        redshift = np.array([0.6, 0.8])

        arr_ones = np.ones(len(redshift))
        for i in range(self.Nc):
            k = self.k(redshift, arr_ones*self.colour_med[i])
            X[i] = (k[1]-k[0]) / (redshift[1]-redshift[0])
            Y[i] = k[0] - X[i]*redshift[0]

        X_interpolator = RegularGridInterpolator((self.colour_med,), X,
                                       bounds_error=False, fill_value=None)
        Y_interpolator = RegularGridInterpolator((self.colour_med,), Y,
                                       bounds_error=False, fill_value=None)
        return X_interpolator, Y_interpolator

    def __A(self, colour):
        # coefficient of the z**4 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__A_interpolator(colour_clipped)

    def __B(self, colour):
        # coefficient of the z**3 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__B_interpolator(colour_clipped)

    def __C(self, colour):
        # coefficient of the z**2 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__C_interpolator(colour_clipped)

    def __D(self, colour):
        # coefficient of the z**1 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__D_interpolator(colour_clipped)

    def __E(self, colour):
        # coefficient of the z**0 term (i.e. independent term)
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__E_interpolator(colour_clipped)

    def __A_spline(self, colour):
        # coefficient of the z**4 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__A_interpolator)

    def __B_spline(self, colour):
        # coefficient of the z**3 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__B_interpolator)

    def __C_spline(self, colour):
        # coefficient of the z**2 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__C_interpolator)

    def __D_spline(self, colour):
        # coefficient of the z**1 term
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__D_interpolator)

    def __E_spline(self, colour):
        # coefficient of the z**0 term (i.e. independent term)
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return splev(colour_clipped, self.__E_interpolator)

    def __X(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__X_interpolator(colour_clipped)

    def __Y(self, colour):
        colour_clipped = np.clip(colour, self.colour_min, self.colour_max)
        return self.__Y_interpolator(colour_clipped)

    def k(self, redshift, colour):
        """
        Polynomial fit to the J-PAS K-Mass-correction for z<0.8
        The K-Mass-correction is extrapolated linearly for z>0.8

        Args:
            redshift: array of redshifts
            colour:   array of rest-frame (u-r) colour
        Returns:
            array of KM-corrections
        """
        K = np.zeros(len(redshift))

        z_lin = 0.8
        idx = redshift <= z_lin

        if self.cubic:
            K[idx] = self.__A_spline(colour[idx])*(redshift[idx]-self.z0)**4 + \
                     self.__B_spline(colour[idx])*(redshift[idx]-self.z0)**3 + \
                     self.__C_spline(colour[idx])*(redshift[idx]-self.z0)**2 + \
                     self.__D_spline(colour[idx])*(redshift[idx]-self.z0) + \
                     self.__E_spline(colour[idx])
        else:
            K[idx] = self.__A(colour[idx])*(redshift[idx]-self.z0)**4 + \
                     self.__B(colour[idx])*(redshift[idx]-self.z0)**3 + \
                     self.__C(colour[idx])*(redshift[idx]-self.z0)**2 + \
                     self.__D(colour[idx])*(redshift[idx]-self.z0) + \
                     self.__E(colour[idx])

        idx = redshift > z_lin

        K[idx] = self.__X(colour[idx])*redshift[idx] + self.__Y(colour[idx])

        return K

    def apparent_magnitude(self, log_stellar_mass, redshift, colour):
        """
        Convert stellar mass to apparent magnitude

        Args:
            log_stellar_mass: array of log10(stellar mass) [M_sun]
            redshift:           array of redshifts
            colour:             array of rest-frame (u-r) colour
        Returns:
            array of apparent magnitudes
        """
        # Luminosity distance
        D_L = (1. + redshift) * self.cosmo.comoving_distance(redshift)

        return -2.5*(log_stellar_mass
                     - 2*np.log10(D_L)
                     - self.k(redshift,colour))

    def log_stellar_mass(self, apparent_magnitude, redshift, colour):
        """
        Convert apparent magnitude to (log of) stellar mass

        Args:
            apparent_magnitude: array of apparent magnitudes
            redshift:           array of redshifts
            colour:             array of rest-frame (u-r) colour
        Returns:
            array of log10(stellar mass) [M_sun]
        """
        # Luminosity distance
        D_L = (1. + redshift) * self.cosmo.comoving_distance(redshift)

        return 2*np.log10(D_L) - 0.4*apparent_magnitude + self.k(redshift,colour)

    def log_stellar_mass_faint(self, redshift, mag_faint):
        """
        Convert faintest apparent magnitude to the corresponding
        lowest stellar mass

        Args:
            redshift: array of redshifts
            mag_faint: faintest apparent magnitude
        Returns:
            array of log10(stellar mass) [M_sun]
        """
        # convert faint apparent magnitude to absolute magnitude
        # for bluest and reddest galaxies
        # use the actual 'reddest' and 'bluest' colours as defined in the
        # k_corr_file
        arr_ones = np.ones(len(redshift))
        log_smass_blue = self.log_stellar_mass(arr_ones*mag_faint,
                                               redshift,
                                               arr_ones*self.colour_bluest)
        log_smass_red = self.log_stellar_mass(arr_ones*mag_faint,
                                              redshift,
                                              arr_ones*self.colour_reddest)

        # find faintest absolute magnitude, add small amount to be safe
        # find lowest stellar mass, substract small amount to be safe
        log_smass_faint = np.minimum(log_smass_blue, log_smass_red) - 0.01

        # avoid 'infinity'
        log_smass_faint = np.maximum(log_smass_faint, 4)

        return log_smass_faint
