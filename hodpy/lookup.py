#! /usr/bin/env python
import numpy as np
import os
import hodpy

def get_lookup_dir():
    """
    Returns the directory containing the lookup files
    """
    path = os.path.abspath(hodpy.__file__)
    path = path.split("/")[:-1]
    path[-1] = "lookup"
    return "/".join(path)


def read_hod_param_file(hod_param_file):
    """
    Read the HOD parameter file
    """
    params = np.loadtxt(hod_param_file, skiprows=8, delimiter=",")
    Mmin_Ls, Mmin_Mt, Mmin_am          = params[0,:3]
    M1_Ls,   M1_Mt,   M1_am            = params[1,:3]
    M0_A,    M0_B                      = params[2,:2]
    alpha_A, alpha_B, alpha_C          = params[3,:3]
    sigma_A, sigma_B, sigma_C, sigma_D = params[4,:4]
    return Mmin_Ls, Mmin_Mt, Mmin_am, M1_Ls, M1_Mt, M1_am, M0_A, M0_B, \
            alpha_A, alpha_B, alpha_C, sigma_A, sigma_B, sigma_C, sigma_D

path = get_lookup_dir()

# Pino simulation
Pino_mass_function = path+"/mf_fits.dat"
Pino_snapshots     = path+"/Pino_snapshots.dat"

# HOD parameters for BGS mock
bgs_hod_parameters    = path+"/hod_params_miniJPAS.dat"
bgs_hod_slide_factors = path+"/slide_factors.dat" # will be created if doesn't exist

# lookup files for central/satellite magnitudes
central_lookup_file   = path+"/central_magnitudes.npy"   # will be created if doesn't exist
satellite_lookup_file = path+"/satellite_magnitudes.npy" # will be created if doesn't exist

# k-corrections
kcorr_file = path+"/k_corr_iband_z04.dat"
#kcorr_file = path+"/k_corr_rband_z01.dat"

# miniJPAS luminosity functions
smf_params      = path+"/smf_params.dat" # describe the miniJPAS catalogue.
target_smf         = path+"/target_smf.dat" # will be created if doesn't exist
