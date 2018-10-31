import numpy as np
import scipy.signal

# z	Values to fit.
# ra,rb	Radius of elliptical neighborhood, ra=major axis.
# theta	Orientation of fit (i.e. of minor axis).
# OUTPUT
# a,b,c	Coefficients of fit: a + bx + cx^2
def fitparab(z, ra, rb, theta, filt):
    """Fit cylindrical parabolas to elliptical patches of z at each pixel"""
    # compute the interior quickly with convolutions
    a = scipy.signal.convolve2d(z, filt[:, :, 0], 'same')
    # fix border with mex file
    a = savgol_border(a, z, ra, rb, theta)
    
    return a
