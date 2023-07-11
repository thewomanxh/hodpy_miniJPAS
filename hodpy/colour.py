#! /usr/bin/env python
import numpy as np
from scipy.special import erfc

class Colour(object):
    """
    Class containing methods for randomly assigning galaxies a u-r
    (rest-frame) colour from our parametrisation of the miniJPAS
    colour-stellar mass diagram.
    We use stellar masses (in M_sun) and (u-r)_rest colours from the
    BaySeaGal analysis.
    """

    def red_mean(self, log_stellar_mass, redshift):
        """
        Mean of the red sequence as a function of stellar mass and redshift

		Comes from fit to miniJPAS data.

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of u-r rest-frame colours
        """
        colour = 2.09 + 0.282*(log_stellar_mass - 10) - 0.401*(redshift - 0.4)
        return colour

    def red_rms(self, log_stellar_mass, redshift):
        """
        RMS of the red sequence as a function of stellar mass and redshift

		Comes from fit to miniJPAS data.

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of u-r rest-frame colours
        """
        colour = 0.155 - 0.010*(log_stellar_mass - 10) + 0.025*(redshift - 0.4)
        return colour

    def blue_mean(self, log_stellar_mass, redshift):
        """
        Mean of the blue sequence as a function of stellar mass and redshift

		Comes from fit to miniJPAS data.

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of u-r rest-frame colours
        """
        colour = 1.581 + 0.689*(log_stellar_mass - 10) - 1.244*(redshift - 0.4)
        return colour

    def blue_rms(self, log_stellar_mass, redshift):

        """
        RMS of the blue sequence as a function of stellar mass and redshift

		Comes from fit to miniJPAS data.

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of u-r rest-frame colours
        """
        colour = 0.232 + 0.036*(log_stellar_mass - 10) - 0.165*(redshift - 0.4)
        return colour

    def satellite_mean(self, log_stellar_mass, redshift):
        """
        Mean satellite colour as a function of stellar mass and redshift

		For now, we use a first guess based only on our blue and red
        sequences. This should be refined to get the right
		clustering as function of colour.

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of u-r rest-frame colours
        """
        colour = 2.003 + 0.340*(log_stellar_mass - 10) - 0.401*(redshift - 0.4)
        return colour

    def fraction_blue(self, log_stellar_mass, redshift):
        """
        Fraction of blue galaxies as a function of stellar mass and redshift

		Comes from a functional fit to the best model in
        Diaz-Garcia et al. (in prep.) (from separately studying SMF for blue
        and red populations in miniJPAS)

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of fraction of blue galaxies
        """
        zref = 0.4
        par_logM50 = 10.520 + 0.352*(redshift - zref) - 0.895*((redshift - zref)**2)
        par_Delta = 0.353 - 0.556*(redshift - zref) + 0.594*((redshift - zref)**2)

        frac_blue = 1./(1. + np.exp((log_stellar_mass - par_logM50)/par_Delta))
        return np.clip(frac_blue, 0, 1)


    def fraction_central(self, log_stellar_mass, redshift):
        """
        Fraction of central galaxies as a function of stellar mass and redshift


        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of fraction of central galaxies (number of satellites divided by number of centrals)
        """
        # number of satellites divided by number of centrals
        nsat_ncen = 0.0849 * (2 - erfc((10.94 - log_stellar_mass)/0.371))
        return 1 / (1 + nsat_ncen)


    def probability_red_satellite(self, log_stellar_mass, redshift):
        """
        Probability a satellite is red as a function of stellar mass and redshift

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of probabilities
        """

        sat_mean  = self.satellite_mean(log_stellar_mass, redshift)
        blue_mean = self.blue_mean(log_stellar_mass, redshift)
        red_mean  = self.red_mean(log_stellar_mass, redshift)

        p_red = np.clip(np.absolute(sat_mean-blue_mean) / \
                        np.absolute(red_mean-blue_mean), 0, 1)
        f_blue = self.fraction_blue(log_stellar_mass, redshift)
        f_cen = self.fraction_central(log_stellar_mass, redshift)

        return np.minimum(p_red, ((1-f_blue)/(1-f_cen)))


    def get_satellite_colour(self, log_stellar_mass, redshift):
        """
        Randomly assigns a satellite galaxy a u-r colour

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of u-r rest-frame colours
        """

        num_galaxies = len(log_stellar_mass)

        # probability the satellite should be drawn from the red sequence
        prob_red = self.probability_red_satellite(log_stellar_mass, redshift)

        # random number for each galaxy 0 <= u < 1
        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)

        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(log_stellar_mass[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(log_stellar_mass[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(log_stellar_mass[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(log_stellar_mass[is_blue], redshift[is_blue])

        # randomly select colour from Gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour, is_red


    def get_central_colour(self, log_stellar_mass, redshift):
        """
        Randomly assigns a central galaxy a u-r colour

        Args:
            log_stellar_mass: array of log(stellar mass) [M_sun]
            redshift:  array of redshifts
        Returns:
            array of u-r rest-frame colours
        """
        num_galaxies = len(log_stellar_mass)

        # find probability the central should be drawn from the red sequence
        prob_red_sat  = self.probability_red_satellite(log_stellar_mass, redshift)
        prob_blue_sat = 1. - prob_red_sat

        frac_cent = self.fraction_central(log_stellar_mass, redshift)
        frac_blue = self.fraction_blue(log_stellar_mass, redshift)

        prob_blue = frac_blue/frac_cent - prob_blue_sat/frac_cent + \
                                                          prob_blue_sat
        prob_red = 1. - prob_blue

        # random number for each galaxy 0 <= u < 1
        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)

        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(log_stellar_mass[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(log_stellar_mass[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(log_stellar_mass[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(log_stellar_mass[is_blue], redshift[is_blue])

        # randomly select colour from gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour, is_red
