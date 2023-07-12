#! /usr/bin/env python
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

from hodpy.catalogue import Catalogue
from hodpy.cosmology import CosmologyPino


class GalaxyCatalogue(Catalogue):
    """
    Galaxy catalogue for a lightcone
    Args:
        haloes:    halo catalogue
        cosmology: object of the class Cosmology
    """
    def __init__(self, haloes, cosmology):
        self._quantities = {}
        self.size = 0
        self.haloes = haloes
        self.cosmology = cosmology


    def get(self, prop):
        """
        Get property from catalogue
        Args:
            prop: string of the name of the property
        Returns:
            array of property
        """
        # properties not directly stored
        if prop == "is_sat":
            return np.invert(self.get("is_cen"))

        # stored properties
        return self._quantities[prop]


    def cut(self, keep):
        """
        Cut catalogue to mask
        Args:
            keep: boolean array
        """
        for quantity in self._quantities:

            if quantity == "cen_ind":
                # need to figure out what the central index is in the new arrays
                # index will be set to -1 if central is cut from catalogue
                cen_ind_old = self._quantities["cen_ind"]
                cen_ind_new = np.ones(self.size, dtype="i") * -1
                ind = np.arange(self.size)[keep]
                cen_ind_new[ind] = np.arange(len(ind))
                cen_ind_new = (cen_ind_new[cen_ind_old])[keep]
                self._quantities["cen_ind"] = cen_ind_new

            else:
                self._quantities[quantity] = self._quantities[quantity][keep]

        self.size = np.count_nonzero(keep)
        
        # Log message
        print(f"After the cut, there are {self.size} galaxies left in the catalogue")


    def get_halo(self, prop):
        """
        Get property from halo catalogue for each galaxy
        Args:
            prop: string of the name of the property
        Returns:
            array of property
        """
        return self.haloes.get(prop)[self.get("halo_ind")]


    def add_galaxies(self, hod):
        """
        Use hod to randomly generate galaxy stellar masses.
        Adds stellar masses, central index, halo index,
        and central/satellite flag to the catalogue.
        Args:
            hod: object of the class HOD
        """
        # assign log(stellar masses)
        log_smass_cen = hod.get_lstellmass_centrals(self.haloes.get("log_mass"),
                                                    self.haloes.get("zcos"))
        n_cen = len(log_smass_cen)
        num_sat = hod.get_number_satellites(self.haloes.get("log_mass"),
                                            self.haloes.get("zcos"))
        ind_cen, log_smass_sat = \
            hod.get_lstellmass_satellites(self.haloes.get("log_mass"), num_sat,
                                         self.haloes.get("zcos"))

        # update size of catalogue
        self.size = len(log_smass_cen) + len(log_smass_sat)

        # add quantities to catalogue

        # add (log of) stellar masses
        log_smass = np.concatenate([log_smass_cen, log_smass_sat])
        self.add("log_stell_mass", log_smass)

        # add index of central galaxy
        ind_cen = np.concatenate([np.arange(n_cen), ind_cen])
        self.add("cen_ind", ind_cen)

        # add boolean array of is central galaxy
        is_cen = np.zeros(self.size, dtype="bool")
        is_cen[:n_cen] = True
        self.add("is_cen", is_cen)

        # add index of host halo in halo catalogue
        halo_ind_cen = np.arange(n_cen)
        halo_ind = halo_ind_cen[ind_cen]
        self.add("halo_ind", halo_ind)
        
        # Log message
        print(f"Added {self.size} galaxies to the catalogue.")


    def _get_distances(self):
        # gets random distance of satellite to central
        distance = np.zeros(self.size)

        is_sat = self.get("is_sat")
        conc = self.get_halo("conc")[is_sat]
        rvir = self.get_halo("rvir")[is_sat]
        u = np.random.rand(len(conc))

        interpolator = self.__nfw_interpolator()
        points = np.array(list(zip(np.log10(conc), np.log10(u))))
        distance[is_sat] = 10**interpolator(points)
        distance[is_sat] *= rvir
        return distance


    def _get_relative_positions(self, distance):
        # relative position of galaxy to centre of halo
        pos_rel = np.zeros((self.size,3))

        # position centrals at centre of halo
        # for satellites, randomly position them around central

        is_sat = self.get("is_sat")
        Nsat = np.count_nonzero(is_sat)

        # generate random x,y,z coordinates in cube
        # if distance to centre greater than 1, reject
        for i in range(3):
            pos_rel[is_sat,i] = 2 * np.random.rand(Nsat) - 1
        dist2 = np.sum(pos_rel**2, axis=1)
        ind = dist2 > 1 # reject these
        num = np.count_nonzero(ind)
        while num > 0:
            for i in range(3):
                pos_rel[ind,i] = 2 * np.random.rand(num) - 1
            dist2[ind] = np.sum(pos_rel[ind,:]**2, axis=1)
            ind = dist2 > 1
            num = np.count_nonzero(ind)

        #scale pos_rel so the distance is correct
        dist = np.sqrt(dist2)
        for i in range(3):
            pos_rel[is_sat,i] = pos_rel[is_sat,i]*distance[is_sat]/dist[is_sat]

        return pos_rel


    def _get_positions(self, distance):
        # positions satellites randomly at the specified distance from the
        # central. Returns ra, dec, z

        # 3d position of halo
        pos_halo = self.equitorial_to_pos3d(self.get_halo("ra"),
                                self.get_halo("dec"), self.get_halo("zcos"))

        pos_rel = self._get_relative_positions(distance)

        ra, dec, z_cos = self.pos3d_to_equitorial(pos_halo + pos_rel)

        return ra, dec, z_cos


    def _get_velocities(self):
        # gets random line of sight velocity of each galaxy

        # line of sight velocity of halo
        vel_los_halo = self.zobs_to_vel(self.get_halo("zcos"),
                                        self.get_halo("zobs"))

        # velocity dispersion from Eq. 12 of Skibba+2006 (in proper km/s)
        vel_disp = np.sqrt(2.151e-9 * (self.get_halo("mass")*\
                          (1.+self.get_halo("zcos"))/self.get_halo("r200")))

        # random line of sight velocity relative to halo
        vel_rel = vel_disp*np.random.normal(loc=0.0, scale=1.0, size=self.size)

        return vel_los_halo + vel_rel


    def position_galaxies(self):
        """
        Position galaxies in haloes and give them random line of sight
        velocities. Centrals are positioned at the centre of the halo,
        satellites are positioned randomly following a NFW profile.
        Adds ra, dec, cosmological redshift and observed redshift
        to the catalogue.
        """
        # random distance to halo centre
        distance = self._get_distances()

        # position around halo centre
        ra, dec, z_cos = self._get_positions(distance)

        # random line of sight velocity
        vel_los = self._get_velocities()

        # use line of sight velocity to get observed redshift
        z_obs = self.vel_to_zobs(z_cos, vel_los)

        # add properties to catalogue
        self.add("ra", ra)
        self.add("dec", dec)
        self.add("zcos", z_cos)
        self.add("zobs", z_obs)
        
        # Log message
        print(f"Added the positions for {self.size} galaxies in the catalogue")


    def __f(self, x):
        return np.log(1.+x) - x/(1.+x)

    def __nfw_interpolator(self):
        # creates a RegularGridInterpolator object used for generating
        # random distances from haloes that follow a NFW profile

        # arrays of log(concentration), log(u) where u is a uniform random
        # number in the range (0,1) and log(s) where s = R/R200
        log_cs = np.arange(-2, 2.7, 0.01)
        log_us = np.arange(-10, 0.001, 0.01)
        log_ss = np.zeros((len(log_cs), len(log_us)))

        log_s = np.arange(-8, 0.0001, 0.001)
        s = 10**log_s
        arr_ones = np.ones(len(log_s))

        for i in range(len(log_cs)):
            c = arr_ones * 10**log_cs[i]
            # f(c*s)/f(c) is the mass enclosed by s divided by M200
            log_u = np.clip(np.log10(self.__f(c*s) / self.__f(c)), -12, 0)

            # find this in the array log_us
            idx = np.searchsorted(log_u, log_us)

            # interpolate
            f = (log_us - log_u[idx-1]) / (log_u[idx] - log_u[idx-1])
            log_ss[i,:] = log_s[idx-1] + f*(log_s[idx]-log_s[idx-1])

        return RegularGridInterpolator((log_cs, log_us), log_ss,
                                       bounds_error=False, fill_value=None)


    def add_apparent_magnitude(self, km_correction):
        """
        Add apparent magnitude to catalogue
        Args:
            km_correction: object of the class KMCorrection
        """
        app_mag = km_correction.apparent_magnitude(self.get("log_stell_mass"),
                                                  self.get("zcos"))
        self.add("app_mag", app_mag)


    def save_to_file(self, file_name, format, properties=None,
                     halo_properties=None):
        """
        Save catalogue to file. The properties to store can be specified
        using the properties argument. If no properties are specified,
        the full catalogue will be saved.

        Args:
            file_name: string of file_name
            format:    string of file format
            properties: (optional) list of properties to save
            halo_properties: (optional) list of halo properties to save
        """

        directory = '/'.join(file_name.split('/')[:-1])
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

        if format == "hdf5":
            import h5py

            f = h5py.File(file_name, "a")

            if properties is None:
                # save every property
                for quantity in self._quantities:
                    f.create_dataset(quantity, data=self._quantities[quantity],
                                     compression="gzip")
            else:
                # save specified properties
                for quantity in properties:
                    f.create_dataset(quantity, data=self._quantities[quantity],
                                     compression="gzip")

            if not halo_properties is None:
                # save specified halo properties
                for quantity in halo_properties:
                    f.create_dataset("halo_"+quantity, compression="gzip",
                                     data=self.get_halo(quantity))
            f.close()

        elif format == "fits":
            from astropy.table import Table

            if properties is None:
                # save every property
                t = Table(list(self._quantities.values()),
                          names=list(self._quantities.keys()))
                t.write(file_name, format="fits")
            else:
                # save specified properties
                data = [None] * len(properties)
                for i, prop in enumerate(properties):
                    data[i] = self._quantities[prop]
                t = Table(data, names=properties)
                t.write(file_name, format="fits")

            if not halo_properties is None:
                # save specified halo properties
                data = [None] * len(halo_properties)
                for i, prop in enumerate(halo_properties):
                    data[i] = self.get_halo(prop)
                    halo_properties[i] = "halo_" + halo_properties[i]
                t = Table(data, names=halo_properties)
                t.write(file_name, format="fits")

        # can add more file formats...

        else:
            raise ValueError("Invalid file format")

        # Log message
        print(f"The catalogue with {self.size} galaxies was saved to file {file_name}.")

class BGSGalaxyCatalogue(GalaxyCatalogue):
    """
    BGS galaxy catalogue for a lightcone
    Args:
        haloes: halo catalogue
    """
    def __init__(self, haloes):
        self._quantities = {}
        self.size = 0
        self.haloes = haloes
        self.cosmology = CosmologyPino()


    def add_colours(self, colour):
        """
        Add colours to the galaxy catalogue.
        Args:
            colour: object of the class Colour
        """
        col = np.zeros(self.size)
        col_class_red = np.zeros(self.size)

        is_cen = self.get("is_cen")
        is_sat = self.get("is_sat")
        log_smass = self.get("log_stell_mass")
        z = self.get("zcos")

        col[is_cen], col_class_red[is_cen] = colour.get_central_colour(log_smass[is_cen], z[is_cen])
        col[is_sat], col_class_red[is_sat] = colour.get_satellite_colour(log_smass[is_sat], z[is_sat])

        self.add("col", col)
        self.add("col_class_red", col_class_red)
        
        # Log message
        print(f"Added colour information for {self.size} galaxies in the catalogue")


    def add_apparent_magnitude(self, km_correction):
        """
        Add apparent magnitude to catalogue, using a colour-dependent
        k-mass-correction
        Args:
            km_correction: object of the class JPAS_KMCorrection
        """
        app_mag = km_correction.apparent_magnitude(self.get("log_stell_mass"),
                                         self.get("zcos"), self.get("col"))
        self.add("app_mag", app_mag)

        # Log message
        print(f"Added apparent magnitude for {self.size} galaxies in the catalogue")
        
        