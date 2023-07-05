#! /usr/bin/env python
from __future__ import print_function
import numpy as np
from scipy.special import gamma, gammaincc
from scipy.interpolate import RegularGridInterpolator, splrep, splev
from scipy.optimize import curve_fit

class StellarMassFunction(object):
    """
    Base Stellar Mass Function
    """
    def __init__(self):
        pass

    def __initialize_interpolator(self):
        #Initializes a RegularGridInterpolator for converting number densities
        #at a certain to redshift to the corresponding log_stell_mass threshold
        
        # arrays of z and log_n, and empty 2d array log_stell_masses
        redshifts = np.arange(0, 1, 0.01)
        log_number_densities = np.arange(-12, -0.5, 0.01)
        log_stell_masses = np.zeros((len(redshifts), len(log_number_densities)))

        # Fill in 2d array of log stellar masses 
        log_sm = np.arange(4, 18, 0.001)
        for i in range(len(redshifts)):
            # find number density at each stellar mass in log_sm 
            log_ns = np.log10(self.Phi_cumulative(log_sm, redshifts[i]))
            
            # find this number density in the array log_number_densities
            idx = np.searchsorted(log_ns, log_number_densities)
            
            # interpolate to find log(stellar mass) at this number density
            f = (log_number_densities - log_ns[idx-1]) / \
                                     (log_ns[idx] - log_ns[idx-1])
            log_stell_masses[i,:] = log_sm[idx-1] + f*(log_sm[idx] - log_sm[idx-1])

            
        # create RegularGridInterpolator object
        return RegularGridInterpolator((redshifts, log_number_densities),
                                        log_stell_masses, 
                                        bounds_error=False, fill_value=None)


    def Phi(self, log_stell_mass, redshift):
        """
        Stellar Mass function as a function of log(stellar mass) and redshift.
        
        This function implements the evolution part, and uses other functions
        to compute the SMF at the reference redshift.
        
        We assume linear evolution for log(stellar mass), via
            log_10(M^*)(z) = log_10(M^*)(z=zref) + Q*(redshift - zref)
        and a quadratic evolution for number density normalisation, via
            log_10(\Phi^*)(z) = log_10(\Phi^*)(z=zref) + P1*(redshift - zref) + P2*(redshift - zref)^2
        
        Args:
            log_stell_mass: array of log10(stellar mass) [in M_sun]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """
        log_stell_mass_ref = log_stell_mass + self.Q * (redshift - self.zref)

        # find interpolated number density at z=zref 
        log_smf_ref = self.__Phi_zref(log_stell_mass_ref)
        
        # shift back to redshift 
        log_smf = log_smf_ref + self.P1 * (redshift - self.zref) \
                              + self.P2 * ((redshift - self.zref)**2)
        
        return 10**log_smf

    def __Phi_zref(self, log_stell_mass):
        # returns a spline fit to the SMF at z=zref (using the cumulative SMF)
        delta = 0.001
        log_sm = np.arange(4, 18, delta)
        phi_cums = self.Phi_cumulative(log_sm, self.zref)
        phi = (phi_cums[:-1] - phi_cums[1:]) / delta
        tck = splrep(log_sm[:-1] + (delta/2), np.log10(phi))
        return splev(log_stell_mass, tck)
        
    def Phi_cumulative(self, log_stell_mass, redshift):
        raise NotImplementedError


    def log_stell_mass(self, number_density, redshift):
        """
        Convert number density to log(stellar mass) threshold
        Args:
            number_density: array of number densities [h^3/Mpc^3]
            redshift: array of redshift
        Returns:
            array of log(stellar mass) [M_sun]
        """
        points = np.array(list(zip(redshift, np.log10(number_density))))
        return self._interpolator(points)
