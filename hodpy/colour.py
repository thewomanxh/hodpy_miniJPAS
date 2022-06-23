#! /usr/bin/env python
import numpy as np
from scipy.special import erfc

# chm, La Laguna, June 2022
# Here I modify the script version from Xiu-hui, where I include my fits to <M_g-M_r> vs (M_i,z) and RMS[M_g-M_r] vs (M_i,z)
#
from scipy.interpolate import interp1d
filein='../lookup/rawdata_gmr_vs_Mr_from_miniJPAS.npz'
ddin=np.load(filein,allow_pickle=True) #np.savez_compressed('rawdata_gmr_vs_Mr_from_miniJPAS.npz',MrB=MrB,zbin=zbin,gmrAV=gmrAV,gmrRMS=gmrRMS)
#
def interpol_generic(lab,jcol,magnitude,redshift):
	#
	jmag= ( (magnitude-np.min(ddin['MrB']))/(ddin['MrB'][1]-ddin['MrB'][0]) ).astype(int)
	jz= ( (redshift-np.min(ddin['zbin']))/(ddin['zbin'][1]-ddin['zbin'][0]) ).astype(int)
	ng=np.size(redshift)
	#
	cc=ddin[lab][0,jcol,:]!=0. ; ncc=np.sum(cc)
	if (ncc>1):
		f1=interp1d(ddin['MrB'][cc],ddin[lab][0,jcol,cc],fill_value="extrapolate")
	else:
		def f1(x):
			out=ddin[lab][cc][0]
			return out
	#
	cc=ddin[lab][1,jcol,:]!=0.; ncc=np.sum(cc)
	if (ncc>1):
		f2=interp1d(ddin['MrB'][cc],ddin[lab][1,jcol,cc],fill_value="extrapolate")
	else:
		def f2(x):
			out=ddin[lab][cc][0]
			return out
	cc=ddin[lab][2,jcol,:]!=0. ; ncc=np.sum(cc)
	if (ncc>1):
		f3=interp1d(ddin['MrB'][cc],ddin[lab][2,jcol,cc],fill_value="extrapolate")
	else:
		def f3(x):
			out=ddin[lab][cc][0]
			return out
	cc=ddin[lab][3,jcol,:]!=0. ; ncc=np.sum(cc)
	if (ncc>1):
		f4=interp1d(ddin['MrB'][cc],ddin[lab][3,jcol,cc],fill_value="extrapolate")
	else:
		def f4(x):
			out=ddin[lab][cc][0]
			return out
	#
	weiz=np.zeros( (len(ddin['zbin']),ng) ) ; Dz=ddin['zbin'][1]-ddin['zbin'][0]
	for iz in range(len(ddin['zbin'])):
		weiz[iz,:]= np.exp( -(redshift-ddin['zbin'][iz])**2/(2.*Dz**2) )
	#
	output=(f1(magnitude)*weiz[0,:]+f2(magnitude)*weiz[1,:]+f3(magnitude)*weiz[2,:]+f4(magnitude)*weiz[3,:])/np.sum(weiz,axis=0)
	#
	return output
	

class Colour(object):
	"""
	Class containing methods for randomly assigning galaxies a g-r
	colour from the parametrisation of the GAMA colour magnitude diagram
	in Smith et al. 2017. r-band absolute magnitudes are k-corrected
	to z=0.1 and use h=1. g-r colours are also k-corrected to z=0.1
	"""
	'''
    def red_mean(self, magnitude, redshift):
        """
        Mean of the red sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """

        colour = 0.932 - 0.032 * (magnitude + 20)
        ind = redshift > 0.1
        colour[ind] -= 0.18 * (np.clip(redshift[ind], 0, 0.4)-0.1)

        return colour


    def red_rms(self, magnitude, redshift):
        """
        RMS of the red sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = 0.07 + 0.01 * (magnitude + 20)
        ind = redshift > 0.1
        colour[ind] += (0.05 + (redshift[ind]-0.1)*0.1) * (redshift[ind]-0.1)
        
        return colour


    def blue_mean(self, magnitude, redshift):
        """
        Mean of the blue sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour_bright = 0.62 - 0.11 * (magnitude + 20)
        colour_faint = 0.4 - 0.0286*(magnitude + 16)
        colour = np.log10(1e9**colour_bright + 1e9**colour_faint)/9
        ind = redshift > 0.1
        colour[ind] -= 0.25 * (np.clip(redshift[ind],0,0.4) - 0.1)
                                                          
        return colour


    def blue_rms(self, magnitude, redshift):

        """
        RMS of the blue sequence as a function of magnitude and redshift

        Args:
            magnitude: array of absolute r-band magnitudes (with h=1)
            redshift:  array of redshifts
        Returns:
            array of g-r colours
        """
        colour = np.clip(0.12 + 0.02 * (magnitude + 20), 0, 0.15)
        ind = redshift > 0.1
        colour[ind] += 0.2*(redshift[ind]-0.1)

        return colour
        '''
        
#[CHM] Here I add my version of the fits -------


	def red_mean(self, magnitude, redshift):
		"""
		Mean of the red sequence as a function of magnitude and redshift

		Args:
		    magnitude: array of absolute i-band magnitudes (with h=1)
		    redshift:  array of redshifts
		Returns:
		    array of g-r colours for Quiescent population
		"""
		lab="gmrAV" ; jcolour=0
		colour=interpol_generic(lab,jcolour,magnitude,redshift)
		#
		return colour
		
	def red_rms(self, magnitude, redshift):
		"""
		RMS of the red sequence as a function of magnitude and redshift

		Args:
		    magnitude: array of absolute i-band magnitudes (with h=1)
		    redshift:  array of redshifts
		Returns:
		    array of g-r RMS for Quiescent population
		"""
		lab="gmrRMS" ; jcolour=0
		colour=interpol_generic(lab,jcolour,magnitude,redshift)
		#
		return colour

	def blue_mean(self, magnitude, redshift):
		"""
		Mean of the red sequence as a function of magnitude and redshift

		Args:
		    magnitude: array of absolute i-band magnitudes (with h=1)
		    redshift:  array of redshifts
		Returns:
		    array of g-r colours for SF population
		"""
		lab="gmrAV" ; jcolour=1
		colour=interpol_generic(lab,jcolour,magnitude,redshift)
		#
		return colour
		
	def blue_rms(self, magnitude, redshift):
		"""
		RMS of the red sequence as a function of magnitude and redshift

		Args:
		    magnitude: array of absolute i-band magnitudes (with h=1)
		    redshift:  array of redshifts
		Returns:
		    array of g-r RMS for SF population
		"""
		lab="gmrRMS" ; jcolour=1
		colour=interpol_generic(lab,jcolour,magnitude,redshift)
		#
		return colour
        

	def satellite_mean(self, magnitude, redshift):
		"""
		Mean satellite colour as a function of magnitude and redshift

		Args:
		    magnitude: array of absolute r-band magnitudes (with h=1)
		    redshift:  array of redshifts
		Returns:
		    array of g-r colours
		"""

		colour = 0.86 - 0.065 * (magnitude + 20)
		ind = redshift > 0.1
		colour[ind] -= 0.18 * (redshift[ind]-0.1) 

		return colour

#[CHM]  -------


	def fraction_blue(self, magnitude, redshift):
		"""
		Fraction of blue galaxies as a function of magnitude and redshift

		Args:
		    magnitude: array of absolute r-band magnitudes (with h=1)
		    redshift:  array of redshifts
		Returns:
		    array of fraction of blue galaxies
		"""
		frac_blue = 0.2*magnitude + \
		    np.clip(4.4 + (1.2 + 0.5*(redshift-0.1))*(redshift-0.1), 4.45, 10)
		frac_blue_skibba = 0.46 + 0.07*(magnitude + 20)

		frac_blue = np.maximum(frac_blue, frac_blue_skibba)

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


