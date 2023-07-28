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

def get_input_dir():
    """
    Returns the directory containing the input files
    """
    path = os.path.abspath(hodpy.__file__)
    path = path.split("/")[:-1]
    path[-1] = "input_model"
    return "/".join(path)

def read_hod_param_file(hod_param_file):
    """
    Read the HOD parameter file
    """
    params = np.loadtxt(hod_param_file, comments='#', delimiter=",")
    Mmin_Ls, Mmin_Mt, Mmin_am          = params[0,:3]
    M1_Ls,   M1_Mt,   M1_am            = params[1,:3]
    M0_A,    M0_B                      = params[2,:2]
    alpha_A, alpha_B, alpha_C          = params[3,:3]
    sigma_A, sigma_B, sigma_C, sigma_D = params[4,:4]
    return Mmin_Ls, Mmin_Mt, Mmin_am, M1_Ls, M1_Mt, M1_am, M0_A, M0_B, \
            alpha_A, alpha_B, alpha_C, sigma_A, sigma_B, sigma_C, sigma_D

## Input files
# These have to be present *always*
input_path = get_input_dir()

# HOD parameters for BGS mock
bgs_hod_parameters    = input_path+"/hod_params_miniJPAS.dat"

# k-corrections
kcorr_file = input_path+"/k_mass_corr_imag_z04.dat"

# miniJPAS luminosity functions
smf_params = input_path+"/smf_params.dat" # describe the miniJPAS catalogue.


## Lookup files
# These will be created if do not exist
# Have to be removed when the model (defined by input files) is changed
path = get_lookup_dir()

# HOD parameters for BGS mock
bgs_hod_slide_factors = path+"/slide_factors.dat" # will be created if doesn't exist

# lookup files for central/satellite magnitudes
central_lookup_file   = path+"/central_stellmasses.npy"   # will be created if doesn't exist
satellite_lookup_file = path+"/satellite_stellmasses.npy" # will be created if doesn't exist

# miniJPAS luminosity functions
target_smf         = path+"/target_smf.dat" # will be created if doesn't exist
