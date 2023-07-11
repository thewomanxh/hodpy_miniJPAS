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
            array of number densities/d(log(stellar_mass)) [h^3/Mpc^3]
        """
        log_stell_mass_ref = log_stell_mass - self.Q * (redshift - self.zref)

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


class StellarMassFunctionSchechter(StellarMassFunction):
    """
    Schecter Stellar Mass function with evolution
    We assume linear evolution for log(stellar mass), via
        log_10(M^*)(z) = log_10(M^*)(z=zref) + Q*(redshift - zref)
    and a quadratic evolution for number density normalisation, via
        log_10(\Phi^*)(z) = log_10(\Phi^*)(z=zref) + P1*(redshift - zref) + P2*(redshift - zref)^2

    Args:
        Phi_star: LF normalization [h^3/Mpc^3]
        log_M_star: log10 of characteristic stellar mass [M_sun]
        alpha: faint end slope
        P1, P2: number density evolution parameters
        Q: magnitude evolution parameter
        zref: reference redshift
    """
    def __init__(self, Phi_star, log_M_star, alpha, P1, P2, Q, zref=0.4):

        # Evolving Shechter luminosity function parameters
        self.Phi_star = Phi_star
        self.log_M_star = log_M_star
        self.alpha = alpha
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.zref = zref

    def Phi(self, log_stell_mass, redshift):
        """
        (Differential) Stellar Mass function as a function of log(stellar mass)
        and redshift
        Args:
            log_stell_mass: array of log10(stellar mass) [in M_sun]
            redshift: array of redshift
        Returns:
            array of number densities/d(log(stellar_mass)) [h^3/Mpc^3]
        """

        # evolve log_M_star and Phi_star to redshift
        log_M_star_z = self.log_M_star + self.Q * (redshift - self.zref)
        log_Phi_star_z = np.log10(self.Phi_star) + self.P1 * (redshift - self.zref) \
                                                 + self.P2 * ((redshift - self.zref)**2)
        Phi_star_z = 10**log_Phi_star_z

        # calculate stellar mass function
        lf = np.log(10) * Phi_star_z
        lf *= (10**(log_stell_mass - log_M_star_z))**(self.alpha + 1)
        lf *= np.exp(-10**(log_stell_mass - log_M_star_z))

        return lf


    def Phi_cumulative(self, log_stell_mass, redshift):
        """
        Cumulative stellar mass function as a function of log(stellar mass)
        and redshift
        Args:
            log_stell_mass: array of log10(stellar mass) [in M_sun]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # evolve M_star and Phi_star to redshift
        log_M_star_z = self.log_M_star + self.Q * (redshift - self.zref)
        log_Phi_star_z = np.log10(self.Phi_star) + self.P1 * (redshift - self.zref) \
                                                 + self.P2 * ((redshift - self.zref)**2)
        Phi_star_z = 10**log_Phi_star_z

        # calculate cumulative stellar mass function
        t = 10**(log_stell_mass - log_M_star_z)
        lf = Phi_star_z*(gammaincc(self.alpha + 2, t)*gamma(self.alpha + 2) - \
                           t**(self.alpha + 1)*np.exp(-t)) / (self.alpha + 1)

        return lf
        
class StellarMassFunctionTabulated(StellarMassFunction):
    """
    Stellar Mass function from tabulated file, with evolution
    We assume linear evolution for log(stellar mass), via
        log_10(M^*)(z) = log_10(M^*)(z=zref) + Q*(redshift - zref)
    and a quadratic evolution for number density normalisation, via
        log_10(\Phi^*)(z) = log_10(\Phi^*)(z=zref) + P1*(redshift - zref) + P2*(redshift - zref)^2

    Args:
        filename: path to ascii file containing tabulated values of cumulative
                  stellar mass function
        P1, P2: number density evolution parameters
        Q: magnitude evolution parameter
        zref: reference redshift
    """
    def __init__(self, filename, P1, P2, Q, zref=0.4):
        
        self.log_stell_mass, self.log_number_density = \
                                 np.loadtxt(filename, unpack=True)
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.zref = zref

        self.__smf_interpolator = \
            RegularGridInterpolator((self.log_stell_mass[::-1],), self.log_number_density[::-1],
                                    bounds_error=False, fill_value=None)

    def Phi_cumulative(self, log_stell_mass, redshift):
        """
        Cumulative stellar mass function as a function of log(stellar mass) 
        and redshift
        Args:
            log_stell_mass: array of log10(stellar mass) [M_sun]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        # shift stellar masses to z=zref 
        log_stell_mass_ref = log_stell_mass - self.Q * (redshift - self.zref)

        # find interpolated number density at z=zref
        log_smf_ref = self.__smf_interpolator(log_stell_mass_ref)

        # shift back to redshift
        log_smf = log_smf_ref + self.P1 * (redshift - self.zref) \
                              + self.P2 * ((redshift - self.zref)**2)
        
        return 10**log_smf
        
class StellarMassFunctionTargetBGS(StellarMassFunction):
    """
    Used to calculate the target stellar mass function at z=zref,
    used to create the BGS mock catalogue.
    This is the result of integrating halo mass function multiplied by the HOD.

    The resulting SMF smoothly transitions to the Schechter miniJPAS SMF 
    at the faint end.

    Args:
        target_smf_file: tabulated file of SMF at z=zref 
                        (if it does not exist, it will be created at __init__)
        smf_param_file: file containing Schechter SMF parameters:
            Phi_star, log_M_star, alpha, P1, P2, Q, zref, log_M_transition
    """
    
    def __init__(self, target_smf_file, smf_param_file, hod_bgs_simple):

        self.Phi_star, self.log_M_star, self.alpha, \
            self.P1, self.P2, self.Q, \
            self.zref, self.log_M_transition = \
                np.loadtxt(smf_param_file, comments='#', delimiter=",")

        self.hod_bgs_simple = hod_bgs_simple

        try:
            self.smf_miniJPAS = \
                StellarMassFunctionTabulated(target_smf_file, self.P1, self.P2, 
                                             self.Q, self.zref)
        except IOError:
            self.smf_miniJPAS = self.__initialize_target_smf(target_smf_file)

        self._interpolator = \
                 self._StellarMassFunction__initialize_interpolator()
        

    def __initialize_target_smf(self, target_smf_file):
        # Create a file of the z=zref target SMF

        print("Calculating target stellar mass function")

        # array of log stellar masses and corresponding number densities
        log_sm_faint = 3.0
        log_sm_bright = 19.0
        log_sm_step = 0.1
        log_stell_masses = np.arange(log_sm_faint, log_sm_bright, log_sm_step)
        ns = np.zeros(len(log_stell_masses))

        # loop through each stellar mass
        for i in range(len(log_stell_masses)):
            f = np.array([1.0,]) # HOD 'slide factor', set to 1
            z = np.array([self.zref,]) # Calculate at zref
            log_sm = np.array([log_stell_masses[i],])
            ns[i] = self.hod_bgs_simple.get_n_HOD(log_sm,z,f) #cumulative SMF
           
        # convert to differential SMF
        log_stell_mass = (log_stell_masses[1:] + log_stell_masses[:-1]) / 2.
        n = (ns[:-1] - ns[1:]) / log_sm_step
        
        # do a spline fit to the differential LF
        log_sm_step_table = 0.001
        log_stell_masses = np.arange(log_sm_faint, 
                                     log_sm_bright + log_sm_step_table,
                                     log_sm_step_table)[::-1] # Inverse order needed to use cumsum later
        zs = np.ones(len(log_stell_masses)) * self.zref 
        
        tck = splrep(log_stell_mass, np.log10(n))
        ns = 10**splev(log_stell_masses, tck)

        # Transition to miniJPAS Schechter SMF at the faint end
        smf_schechter = StellarMassFunctionSchechter(self.Phi_star, 
                                                     self.log_M_star,
                                                     self.alpha,
                                                     self.P1, self.P2, self.Q,
                                                     self.zref)

        ns_schechter = smf_schechter.Phi(log_stell_masses, zs)
        T = 1. / (1. + np.exp(5*(log_stell_masses - self.log_M_transition)))
        ns = ns*T + ns_schechter*(1 - T)

        # convert back to cumulative SMF
        data = np.zeros((len(log_stell_masses), 2))
        data[:,0] = log_stell_masses   # I remove the '+step/2' here
        data[:,1] = np.log10(np.cumsum(ns*log_sm_step_table))
        # save to file
        np.savetxt(target_smf_file, data)

        return StellarMassFunctionTabulated(target_smf_file, self.P1, self.P2,
                                           self.Q, self.zref)


    def Phi(self, log_stell_mass, redshift):
        """
        (Differential) Stellar Mass function as a function of log(stellar mass)
        and redshift
        Args:
            log_stell_mass: array of log10(stellar mass) [in M_sun]
            redshift: array of redshift
        Returns:
            array of number densities/d(log(stellar_mass)) [h^3/Mpc^3]
        """
        
        smf_miniJPAS = self.smf_miniJPAS.Phi(log_stell_mass, redshift)

        return smf_miniJPAS
    
    def Phi_cumulative(self, log_stell_mass, redshift):
        """
        Cumulative stellar mass function as a function of log(stellar mass)
        and redshift
        Args:
            log_stell_mass: array of log10(stellar mass) [in M_sun]
            redshift: array of redshift
        Returns:
            array of number densities [h^3/Mpc^3]
        """

        smf_miniJPAS = self.smf_miniJPAS.Phi_cumulative(log_stell_mass, redshift)
        
        return smf_miniJPAS
