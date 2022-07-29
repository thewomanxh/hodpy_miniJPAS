#! /usr/bin/env python
import numpy as np
from scipy.special import erfc

class Colour(object):
    """
    Class containing methods for randomly assigning galaxies a g-r
    colour from the parametrisation of the GAMA colour magnitude diagram
    in Smith et al. 2017. r-band absolute magnitudes are k-corrected
    to z=0.1 and use h=1. g-r colours are also k-corrected to z=0.1
    """

    def red_mean(self, magnitude, redshift):
        """
        Mean of the red sequence as a function of magnitude and redshift

		Comes from fit to miniJPAS data.

        Args:
            magnitude: array of absolute i-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        
        colour = 0.62 - 0.04*(magnitude + 21) - 0.06*(redshift - 0.4)
        return colour


    def red_rms(self, magnitude, redshift):
        """
        RMS of the red sequence as a function of magnitude and redshift

		Comes from fit to miniJPAS data.

        Args:
            magnitude: array of absolute i-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = 0.086 + 0.001*(magnitude + 21) + 0.033*(redshift - 0.4)
        return colour


    def blue_mean(self, magnitude, redshift):
        """
        Mean of the blue sequence as a function of magnitude and redshift

		Comes from fit to miniJPAS data.

        Args:
            magnitude: array of absolute i-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = 0.57 - 0.09*(magnitude + 21) - 0.21*(redshift - 0.4)
        return colour


    def blue_rms(self, magnitude, redshift):

        """
        RMS of the blue sequence as a function of magnitude and redshift

		Comes from fit to miniJPAS data.

        Args:
            magnitude: array of absolute i-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = 0.160 - 0.022*(magnitude + 21) - 0.080*(redshift - 0.4)
        return colour


    def satellite_mean(self, magnitude, redshift):
        """
        Mean satellite colour as a function of magnitude and redshift

		For now, we fix this to the same function as the mean colour
		for the red population (this is approach '(i)' in
		Skibba & Sheth 2009). This should be refined to get the rigth
		clustering as function of colour.

        Args:
            magnitude: array of absolute i-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = 0.62 - 0.04*(magnitude + 21) - 0.06*(redshift - 0.4)
        return colour


    def fraction_blue(self, magnitude, redshift):
        """
        Fraction of blue galaxies as a function of magnitude and redshift

		Comes from fit to miniJPAS data.

        Args:
            magnitude: array of absolute i-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of fraction of blue galaxies
        """
        frac_blue = 0.75 + 0.28*(magnitude + 21) + 1.32*(redshift - 0.4) - 1.91*((redshift - 0.4)**2)
        return np.clip(frac_blue, 0, 1)


    def fraction_central(self, magnitude, redshift):
        """
        Fraction of central galaxies as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of fraction of central galaxies
        """
        # number of satellites divided by number of centrals
        nsat_ncen = 0.35 * (2 - erfc(0.6*(magnitude+20.5)))
        return 1 / (1 + nsat_ncen)


    def probability_red_satellite(self, magnitude, redshift):
        """
        Probability a satellite is red as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of probabilities
        """

        sat_mean  = self.satellite_mean(magnitude, redshift)
        blue_mean = self.blue_mean(magnitude, redshift)
        red_mean  = self.red_mean(magnitude, redshift)

        p_red = np.clip(np.absolute(sat_mean-blue_mean) / \
                        np.absolute(red_mean-blue_mean), 0, 1)
        f_blue = self.fraction_blue(magnitude, redshift)
        f_cen = self.fraction_central(magnitude, redshift)

        return np.minimum(p_red, ((1-f_blue)/(1-f_cen)))


    def get_satellite_colour(self, magnitude, redshift):
        """
        Randomly assigns a satellite galaxy a g-r colour

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """

        num_galaxies = len(magnitude)

        # probability the satellite should be drawn from the red sequence
        prob_red = self.probability_red_satellite(magnitude, redshift)

        # random number for each galaxy 0 <= u < 1
        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)

        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(magnitude[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(magnitude[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(magnitude[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(magnitude[is_blue], redshift[is_blue])

        # randomly select colour from Gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour


    def get_central_colour(self, magnitude, redshift):
        """
        Randomly assigns a central galaxy a g-r colour

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        num_galaxies = len(magnitude)

        # find probability the central should be drawn from the red sequence
        prob_red_sat  = self.probability_red_satellite(magnitude, redshift)
        prob_blue_sat = 1. - prob_red_sat

        frac_cent = self.fraction_central(magnitude, redshift)
        frac_blue = self.fraction_blue(magnitude, redshift)

        prob_blue = frac_blue/frac_cent - prob_blue_sat/frac_cent + \
                                                          prob_blue_sat
        prob_red = 1. - prob_blue

        # random number for each galaxy 0 <= u < 1
        u = np.random.rand(num_galaxies)

        # if u <= p_red, draw from red sequence, else draw from blue sequence
        is_red = u <= prob_red
        is_blue = np.invert(is_red)

        mean = np.zeros(num_galaxies, dtype="f")
        mean[is_red]  = self.red_mean(magnitude[is_red],   redshift[is_red])
        mean[is_blue] = self.blue_mean(magnitude[is_blue], redshift[is_blue])

        stdev = np.zeros(num_galaxies, dtype="f")
        stdev[is_red]  = self.red_rms(magnitude[is_red],   redshift[is_red])
        stdev[is_blue] = self.blue_rms(magnitude[is_blue], redshift[is_blue])

        # randomly select colour from gaussian
        colour = np.random.normal(loc=0.0, scale=1.0, size=num_galaxies)
        colour = colour * stdev + mean

        return colour
