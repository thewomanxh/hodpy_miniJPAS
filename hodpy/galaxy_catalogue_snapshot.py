#! /usr/bin/env python
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

from hodpy.cosmology import CosmologyMXXL
from hodpy.galaxy_catalogue import GalaxyCatalogue


class GalaxyCatalogueSnapshot(GalaxyCatalogue):
    """
    Galaxy catalogue for a simuation snapshot
    Args:
        haloes:    halo catalogue
        cosmology: object of the class Cosmology
        box_size:  comoving simulation box length (Mpc/h)
    """
    def __init__(self, haloes, cosmology, box_size):
        self._quantities = {}
        self.size = 0
        self.haloes = haloes
        self.box_size = box_size
        self.cosmology = cosmology
        

    def _get_positions(self, distance):
        # positions satellites randomly at the specified distance from the
        # central. Returns 3d position vector in box (comoving Mpc/h)

        # 3d position of halo
        pos_halo = self.get_halo("pos")

        # relative position of galaxy to centre of halo
        pos_rel = self._get_relative_positions(distance)

        # 3d position of galaxy
        pos = pos_halo + pos_rel

        # deal with periodic boundary
        idx = pos > self.box_size
        pos[idx] -= self.box_size
        idx = pos < 0
        pos[idx] += self.box_size

        return pos


    def _get_velocities(self):
        # gets random velocity vector of each galaxy

        # velocity of halo
        vel_halo = self.get_halo("vel")

        # velocity dispersion from Eq. 12 of Skibba+2006 (in proper km/s)
        vel_disp = np.sqrt(2.151e-9 * (self.get_halo("mass")*\
                          (1.+self.get_halo("zcos"))/self.get_halo("r200")))

        # random velocity along each axis
        vel_rel = np.zeros(vel_halo.shape)
        for i in range(3):
            vel_rel[:,i] = vel_disp*np.random.normal(loc=0.0, scale=1.0, 
                                                     size=self.size)

        return vel_halo + vel_rel


    def position_galaxies(self):
        """
        Position galaxies in haloes and give them random
        velocities. Centrals are positioned at the centre of the halo,
        satellites are positioned randomly following a NFW profile.
        Adds position, velocity and cosmological redshift
        to the catalogue.
        """
        # random distance to halo centre
        distance = self._get_distances()

        # position around halo centre
        pos = self._get_positions(distance)

        # random velocity vector
        vel = self._get_velocities()

        # add properties to catalogue
        self.add("pos", pos)
        self.add("vel", vel)
        self.add("zcos", np.ones(len(distance))*self.get_halo('zcos')[0])

        
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


class BGSGalaxyCatalogueSnapshot(GalaxyCatalogueSnapshot):
    """
    BGS galaxy catalogue for a simulation snapshot

    Args:
        haloes: halo catalogue
    """
    def __init__(self, haloes):
        self._quantities = {}
        self.size = 0
        self.haloes = haloes
        self.box_size = 3000.
        self.cosmology = CosmologyMXXL()

        
    def add_colours(self, colour):
        """
        Add colours to the galaxy catalogue.

        Args:
            colour: object of the class Colour
        """
        col = np.zeros(self.size)
        
        is_cen = self.get("is_cen")
        is_sat = self.get("is_sat")
        abs_mag = self.get("abs_mag")
        z = self.get("zcos")

        col[is_cen] = colour.get_central_colour(abs_mag[is_cen], z[is_cen])
        col[is_sat] = colour.get_satellite_colour(abs_mag[is_sat], z[is_sat])

        self.add("col", col)


    def add_apparent_magnitude(self, k_correction):
        """
        Add apparent magnitude to catalogue, using a colour-dependent
        k-correction

        Args:
            k_correction: object of the class GAMA_KCorrection
        """
        app_mag = k_correction.apparent_magnitude(self.get("abs_mag"),
                                         self.get("zcos"), self.get("col"))
        self.add("app_mag", app_mag)
