#! /usr/bin/env python
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.misc import derivative

from hodpy.power_spectrum import PowerSpectrum
from hodpy.cosmology import CosmologyPino
from hodpy import lookup


class MassFunction(object):
    """
    Class for fitting a Sheth-Tormen-like mass function to measurements from a simulation snapshot

    Args:
        cosmology: hodpy.Cosmology object, in the cosmology of the simulation
        redshift: redshift of the simulation snapshot
        [fit_params]: if provided, sets the best fit parameters to these values. 
                      fit_params is an array of [dc, A, a, p]
        [measured_mass_function]: mass function measurements to fit to, if provided.
                      measured_mass_function is an array of [log mass bins, mass function]
    """
    def __init__(self, cosmology, redshift, fit_params=None, measured_mass_function=None):
        
        self.cosmology = cosmology
        self.power_spectrum = PowerSpectrum(self.cosmology)
        self.redshift = redshift
        
        if not fit_params is None:
            self.dc, self.A, self.a, self.p = fit_params
            
        if not measured_mass_function is None:
            self.__mass_bins = measured_mass_function[0]
            self.__mass_func = measured_mass_function[1]
            
     
    def __func(self, sigma, dc, A, a, p):
        # Sheth-Tormen mass function
        mf = A * np.sqrt(2*a/np.pi)
        mf *= 1 + (sigma**2 / (a * dc**2))**p
        mf *= dc / sigma
        mf *= np.exp(-a * dc**2 / (2*sigma**2))

        return np.log10(mf)
        
    
    def get_fit(self):
        """
        Fits the Sheth-Tormen mass function to the measured mass function, returning the
        best fit parameters
        
        Returns:
            an array of [dc, A, a, p]
        """
        sigma = self.power_spectrum.sigma(10**self.__mass_bins, self.redshift)
        mf = self.__mass_func / self.power_spectrum.cosmo.mean_density(0) * 10**self.__mass_bins
        
        popt, pcov = curve_fit(self.__func, sigma, np.log10(mf), p0=[1,0.1,1.5,-0.5])
        
        self.update_params(popt)
        
        return popt

    
    def update_params(self, fit_params):
        '''
        Update the values of the best fit params
        
        Args:
            fit_params: an array of [dc, A, a, p]
        '''     
        self.dc, self.A, self.a, self.p = fit_params
        
    
    def mass_function(self, log_mass, redshift=None):
        '''
        Returns the halo mass function as a function of mass and redshift
        (where f is defined as Eq. 4 of Jenkins 2000)

        Args:
            log_mass: array of log_10 halo mass, where halo mass is in units Msun/h
        Returns:
            array of halo mass function
        '''        
        
        sigma = self.power_spectrum.sigma(10**log_mass, self.redshift)

        return 10**self.__func(sigma, self.dc, self.A, self.a, self.p)

    
    def number_density(self, log_mass, redshift=None):
        '''
        Returns the number density of haloes as a function of mass and redshift

        Args:
            log_mass: array of log_10 halo mass, where halo mass is in units Msun/h
        Returns:
            array of halo number density in units (Mpc/h)^-3
        '''  
        mf = self.mass_function(log_mass)

        return mf * self.power_spectrum.cosmo.mean_density(0) / 10**log_mass
        

        

class MassFunctionPino(object):
    """
    Class containing the fits to the Pino halo mass function

    Args:
        mf_fits_file: Tabulated file of the best fit mass function parameters
    """
    def __init__(self, mf_fits_file=lookup.Pino_mass_function):
        
        #self.power_spectrum = power_spectrum
        self.cosmology = CosmologyPino()
        self.power_spectrum = PowerSpectrum(self.cosmology)
        
        # read in Pino mass function fit parameters
        snap, redshift, A, a, p = \
                   np.loadtxt(mf_fits_file, skiprows=1, unpack=True)
        
        # interpolate parameters
        self._A = RegularGridInterpolator((redshift,), A, bounds_error=False, 
                                          fill_value=None)

        self._a = RegularGridInterpolator((redshift,), a, bounds_error=False, 
                                          fill_value=None)

        self._p = RegularGridInterpolator((redshift,), p, bounds_error=False, 
                                          fill_value=None)

    def A(self, redshift):
        return self._A(redshift)

    def a(self, redshift):
        return self._a(redshift)

    def p(self, redshift):
        return self._p(redshift)

    def mass_function(self, log_mass, redshift):
        '''
        Returns the halo mass function as a function of mass and redshift
        (where f is defined as Eq. 4 of Jenkins 2000)

        Args:
            log_mass: array of log_10 halo mass, where halo mass is in units Msun/h
            redshift: array of redshift
        Returns:
            array of halo mass function
        '''        
        
        sigma = self.power_spectrum.sigma(10**log_mass, redshift)

        ## xiu's change: following I use Eq. 12 of Watson 2013 instead of Eq.4 of Jenkins 2000, and the parameters of equation fixed.
        
        A_watson = 0.282
        alpha_watson = 2.163
        beta_watson = 1.406
        gamma_watson = 1.210
        mf = (beta_watson/sigma)**alpha_watson+1
        mf = A_watson*mf *np.exp(-gamma_watson/sigma**2)
        
        return mf

    def ln_invsigma_log10M(self, log_mass, redshift):
        '''
        Function that gives the (natural) logarithm of the inverse variance (sigma) as function of the
        (base-10) logarithm of the mass.

        It will be needed for the conversion from f(sigma) to dn/dlog10M

        :param log_mass: base-10 logarithm of the halo mass (M_sun/h)
        :param redshift: redshift
        :return: natural logarithm of the inverse of the corresponding variance (sigma)
        '''

        return np.log(1./self.power_spectrum.sigma(10**log_mass, redshift))

    def d_lninvsigma_d_log10M(self, log_mass, redshift, delta_m_rel=1e-4):
        '''
        Estimates the derivative of the `ln_invsigma_log10M()` function for a given value of the mass.

        :param log_mass: base-10 logarithm of the halo mass (M_sun/h)
        :param redshift: redshift
        :param delta_m_rel: parameter that sets the relative change in log10(M) used for the
            finite-differences estimate of the derivative
        :return: derivative of the ln_invsigma_log10M function
        '''

        return derivative(func=self.ln_invsigma_log10M, x0=log_mass,
                            dx=delta_m_rel*log_mass, n=1, args=[redshift,])



    def number_density(self, log_mass, redshift):
        '''
        Returns the number density of haloes as a function of mass and redshift

        We make the correct conversion from f(sigma) to dN/dlog10M following
        equation (5) in Watson-2013

        Args:
            log_mass: array of log_10 halo mass, where halo mass is in units 
                      Msun/h
            redshift: array of redshift
        Returns:
            array of halo number density in units (Mpc/h)^-3
        '''  
        mf = self.mass_function(log_mass, redshift)

        return mf * self.power_spectrum.cosmo.mean_density(0) * self.d_lninvsigma_d_log10M(log_mass, redshift) / (10 ** log_mass)
        

