#! /usr/bin/env python
import numpy as np
import h5py

from hodpy.cosmology import CosmologyPino
from hodpy.catalogue import Catalogue
from hodpy.power_spectrum import PowerSpectrum
from hodpy import lookup

# Critical density in units of
# h^2 M_sol Mpc^-3 (from Peacock book)
# Independent of cosmology, just definition of rho_crit
# and using H0 and G in adequate units
# This is adequate for the units used here
RHO_CRIT_UNITS = 2.7752E+11


class HaloCatalogue(Catalogue):
    """
    Parent class for a halo catalogue
    """
    def __init__(self, cosmology):
        self._quantities = {}
        self.size = 0
        self.cosmology = cosmology
        self.power_spectrum = PowerSpectrum(self.cosmology)

    def get(self, prop):
        """
        Get property from catalogue

        Args:
            prop: string of the name of the property
        Returns:
            array of property
        """
        # calculate properties not directly stored
        if prop == "log_mass":
            return np.log10(self._quantities["mass"])
        elif prop == "r200":
            return self.get_r200()
        elif prop == "rvir":
            return self.get_rvir()
        elif prop == "conc":
            return self.get_concentration()
        
        # property directly stored
        return self._quantities[prop]

    def get_r200(self, comoving=True):
        """
        Returns R200mean of each halo

        Args:
            comoving: (optional) if True convert to comoving distance
        Returns:
            array of R200mean [Mpc/h]
        """
        rho_mean = self.cosmology.mean_density(self.get("zcos"))
        r200 = (3./(800*np.pi) * self.get("mass") / rho_mean)**(1./3)
        
        if comoving:
            return r200 * (1.+self.get("zcos"))
        else:
            return r200
    
    def get_rvir(self):
        """
        Returns the virial radius of each halo.
        
        We obtain it from equation (A11) in Coupon et al. (2012)
        """
        rho_mean_0 = self.cosmology.mean_density(0)
        Dvir = self.cosmology.Delta_vir(self.get("zcos"))
        
        rvir = (3. * self.get("mass") / (4. * np.pi * rho_mean_0 * Dvir))**(1./3.)
        
        return rvir

    def get_concentration(self):
        """
        Returns NFW concentration of each halo, calculated from the halo mass.
        We use equation (A.10) from Coupon et al. (2012) (but also used,
        e.g. in Zehavi et al. 2011)
        
        Returns:
            array of halo concentrations
        """
        c_zero = 11.0
        beta = 0.13
        mass_star = self.power_spectrum.mass_nonlin0()
        
        conc = (c_zero/(1.+ self.get("zcos")))*((self.get("mass")/mass_star)**(-beta))
        
        return np.clip(conc, 0.1, 1e4)


    

class PinoCatalogue(HaloCatalogue):
    """
    Pino halo lightcone catalogue
    """

    def __init__(self, file_name):

        self.cosmology = CosmologyPino()
        self.power_spectrum = PowerSpectrum(self.cosmology)

        # read halo catalogue file
        halo_cat = h5py.File(file_name, "r")

        self._quantities = {
            'ra':    self.__read_property(halo_cat, 'ra'),
            'dec':   self.__read_property(halo_cat, 'dec'),
            'mass':  self.__read_property(halo_cat, 'M200m') * 1e10,
            'zobs':  self.__read_property(halo_cat, 'z_obs'),
            'zcos':  self.__read_property(halo_cat, 'z_cos'),
            'rvmax': self.__read_property(halo_cat, 'rvmax')
            }
        halo_cat.close()

        self.size = len(self._quantities['ra'][...])


    def __read_property(self, halo_cat, prop):
        # read property from halo file
        return halo_cat["Data/"+prop][...]
    
