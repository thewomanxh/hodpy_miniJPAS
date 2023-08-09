#! /usr/bin/env python
from __future__ import print_function
import numpy as np

from hodpy.halo_catalogue import PinoCatalogue
from hodpy.galaxy_catalogue import BGSGalaxyCatalogue
from hodpy.cosmology import CosmologyPino
from hodpy.mass_function import MassFunctionPino
from hodpy.hod_bgs import HOD_BGS
from hodpy.kmass_correction import JPAS_KMCorrection
from hodpy.colour import Colour
from hodpy import lookup


def main(input_file, output_file, mag_faint, rand_seed=None,
         input_format='hdf5', input_nplc=64):

    import warnings
    warnings.filterwarnings("ignore")

    # Initialise random generator (for reproducibility if seed is not None)
    np.random.seed(rand_seed)

    # create halo catalogue
    halo_cat = PinoCatalogue(input_file, input_format, input_nplc)

    # empty galaxy catalogue
    gal_cat  = BGSGalaxyCatalogue(halo_cat)

    # use hods to populate galaxy catalogue
    hod = HOD_BGS(mag_faint=mag_faint)
    gal_cat.add_galaxies(hod)

    # position galaxies around their haloes
    gal_cat.position_galaxies()

    # add g-r colours
    col = Colour()
    gal_cat.add_colours(col)

    # use colour-dependent k-correction to get apparent magnitude
    kcorr = JPAS_KMCorrection(CosmologyPino())
    gal_cat.add_apparent_magnitude(kcorr)

    # cut to galaxies brighter than apparent magnitude threshold
    gal_cat.cut(gal_cat.get("app_mag") <= mag_faint)

    # save catalogue to file
    gal_cat.save_to_file(output_file, format="hdf5", halo_properties=["mass",])



if __name__ == "__main__":

    input_file = "input/halo_catalogue_pinocchio.hdf5"
    input_format = 'hdf5'
    input_nplc = 0  # Only relevant if input_format = 'plc'

    output_file = "output/galaxy_catalogue_pinocchio.hdf5"
    mag_faint = 20.0 # faintest apparent magnitude
    rand_seed = 4242 # seed for random number generator

    main(input_file, output_file, mag_faint,
         rand_seed=rand_seed, input_format=input_format, input_nplc=input_nplc)
