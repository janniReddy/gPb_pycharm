import numpy as np
import cv2
import math
import scipy.signal

from oe_filters import oe_filters
from textons import textons

def grayscale(L, a, b):
    """Compute a grayscale image from an RGB image."""
    print("inside gray scale function.............")
    g_im = np.empty(L.shape)
    g_im.dtype = L.dtype
    g_im = (0.29894 * L) + (0.58704 * a) + (0.11402 * b)
    return g_im

def rgb_to_lab(L, a, b):
    """Convert from RGB color space to Lab color space."""
    print("inside rgb to lab")
    # convert RGB to XYZ
    x_l = (0.412453 * L) +  (0.357580 * a) + (0.180423 * b)
    y_a = (0.212671 * L) +  (0.715160 * a) + (0.072169 * b)
    z_b = (0.019334 * L) +  (0.119193 * a) + (0.950227 * b)
    # D65 white point reference
    x_ref = 0.950456
    y_ref = 1.000000
    z_ref = 1.088754
    # threshold value
    threshold = 0.008856
    # convert XYZ to Lab
    for i in range(0, L.shape[0]):
        for j in range(0, L.shape[1]):
            x = x_l[i][j] / x_ref
            y = y_a[i][j] / y_ref
            z = z_b[i][j] / z_ref
            # compute fx, fy, fz
            if x > threshold:
                fx = math.pow(x,(1.0/3.0))
            else:
                fx = (7.787*x + (16.0/116.0))

            if y > threshold:
                fy = math.pow(y,(1.0/3.0))
            else:
                fy = (7.787*y + (16.0/116.0))
            
            if z > threshold:
                fz = math.pow(z,(1.0/3.0))
            else:
                fz = (7.787*z + (16.0/116.0))

            # compute Lab color value
            if y > threshold:
                x_l[i][j] = (116*math.pow(y,(1.0/3.0)) - 16)
            else:
                x_l[i][j] = 903.3*y
            
            y_a[i][j] = 500 * (fx - fy)
            z_b[i][j] = 200 * (fy - fz)
    return [x_l, y_a, z_b]

def lab_normalize(l, a, b):
    """Normalize an Lab image so that values for each channel lie in [0,1]."""
    print("inside lab_normalize")
    # range for a, b channels
    ab_min = -73
    ab_max = 95
    ab_range = ab_max - ab_min
    # normalize Lab image
    for i in range(0, l.shape[0]):
        for j in range(0, l.shape[1]):
            l_val = l[i][j] / 100.0
            a_val = (a[i][j] - ab_min) / ab_range
            b_val = (b[i][j] - ab_min) / ab_range
            if l_val < 0:
                l_val = 0
            elif l_val > 1:
                l_val = 1
            
            if a_val < 0:
                a_val = 0
            elif a_val > 1:
                a_val = 1

            if b_val < 0:
                b_val = 0
            elif b_val > 1:
                b_val = 1
            l[i][j] = l_val
            a[i][j] = a_val
            b[i][j] = b_val
    return [l, a, b]


def quantize_values(src, n_bins):
    if(0 == n_bins):
        print("n_bins must be > 0")
        return
    dest = np.empty(src.shape)
    # dest.dtype = src.dtype
    for i in range(0, src.shape[0]):
        for j in range(0, src.shape[1]):
            d_bin = int(math.floor(src[i][j]*float(n_bins)))
            if d_bin == n_bins:
                d_bin = n_bins - 1
            dest[i][j] = d_bin
    return dest

def texton_filters(n_ori):
    """computes texton filters"""
    start_sigma = 1
    num_scales = 2
    scaling = math.sqrt(2)
    elongation = 2
    support = 3
    filter_bank = np.zeros((len(range(1, n_ori + 1))*2, len(range(1, num_scales + 1)))).tolist()
    for idx0, scale in enumerate(range(1, num_scales + 1)):
        sigma = start_sigma * (scaling**(scale - 1))
        for orient in range(1, n_ori + 1):
            theta = (orient-1)/float(n_ori) * math.pi
            filter_bank[(2*orient)-2][idx0] = oe_filters([sigma*elongation, sigma], support, theta, 2, 0)
            filter_bank[(2*orient)-1][idx0] = oe_filters([sigma*elongation, sigma], support, theta, 2, 1)

    return filter_bank

def compute_textons(g_im, border, filters, k):
    return textons(g_im, border, filters, k)

def gaussian(sigma = 1, deriv = 0, hlbrt = False):
    """
    * Gaussian kernel (1D).
    *
    * Specify the standard deviation and (optionally) the support.
    * The length of the returned vector is 2*support + 1.
    * The support defaults to 3*sigma.
    * The kernel is normalized to have unit L1 norm.
    * If returning a 1st or 2nd derivative, the kernel has zero mean."""
    print("inside gaussian function")
    support = math.ceil(3*sigma)
    # enlarge support so that hilbert transform can be done efficiently
    support_big = support
    if hlbrt:
        support_big = 1
        temp = support
        while temp > 0:
            support_big *= 2
            temp /= 2

    # compute constants
    sigma2_inv = 1/(sigma * sigma)
    neg_two_sigma2_inv = (-0.5) * sigma2_inv
    # compute gaussian (or gaussian derivative)
    size = 2 * support_big + 1
    m = np.zeros((size))
    print(m.shape)
    x = -(support_big)
    if deriv == 0:
        # compute gaussian
        for n in range(0, size):
            m[n] = math.exp(x * x * neg_two_sigma2_inv)
            x += 1
    elif deriv == 1:
        # compute gaussian first derivative
        for n in range(0, size):
            m[n] = math.exp(x * x * neg_two_sigma2_inv) * (-x)
            x += 1
    elif deriv == 2:
        # compute gaussian second derivative
        for n in range(0, size):
            x2 = x * x
            m[n] = math.exp(x2 * neg_two_sigma2_inv) * (x2 * sigma2_inv -1)
            x += 1
    else:
        print(" only derivatives 0,1,2 supported")

    # take hilbert transform (if requested)
    if hlbrt:
        # grab power of two sized submatrix (ignore last element)
        m = scipy.signal.hilbert(m).imag
    
    #zero mean
    if deriv>0:
        m = m - np.mean(m)

    #unit L1-norm
    sumf = np.sum(np.abs(m))
    if sumf>0:
        m = m / sumf
    
    return m

def border_trim_2D(m, r):
    return m[r:-r,r:-r]

def convert2OneD(src):
    """converts a 2 dimensional numpy array to 1 dimensional numpy array"""
    print("inside convert2OneD")
    nrows = src.shape[0]
    ncols = src.shape[1]
    res = np.zeros((nrows * ncols))
    n = 0
    for i in range(0, nrows):
        for j in range(0, ncols):
            res[n] = src[i][j]
            n += 1
    print("before shape printed")
    print(res.shape)
    return res

def convert2TwoD(src, nrows, ncols):
    """converts a 1 dimensional numpy array to 2 dimensional numpy array"""
    res = np.zeros((nrows, ncols))
    ind = 0
    for i in range(0, nrows):
        for j in range(0, ncols):
            res[i][j] = src[ind]
            ind += 1
    return res

def weight_matrix_disc(r):
    """Construct weight matrix for circular disc of the given radius."""
    # initialize weights array
    size = 2 * r + 1
    weights = np.zeros((size * size))
    # set values in disc to 1
    radius = int(r)
    r_sq = radius * radius
    ind = 0
    for x in range(-radius, radius + 1):
        x_sq = x * x
        for y in range(-radius, radius + 1):
            # check if index is within disc
            y_sq = y * y
            if ((x_sq + y_sq) <= r_sq):
                weights[ind] = 1
            ind += 1       
    return weights




def hist_gradient_2D(labels, r, n_ori, smoothing_kernel):
    """* Compute the distance between histograms of label values in oriented
    * half-dics of the specified radius centered at each location in the 2D
    * matrix.  Return one distance matrix per orientation.
    *
    * Alternatively, instead of specifying label values at each point, the user
    * may specify a histogram at each point, in which case the histogram for
    * a half-disc is the sum of the histograms at points in that half-disc.
    *
    * The half-disc orientations are k*pi/n for k in [0,n) where n is the 
    * number of orientation requested.
    *
    * The user may optionally specify a nonempty 1D smoothing kernel to 
    * convolve with histograms prior to computing the distance between them.
    *
    * The user may also optionally specify a custom functor for computing the
    * distance between histograms."""
    # construct weight matrix for circular disc
    weights = weight_matrix_disc(r)

    # # check arguments - weights
    # if len(weights.shape) != 2:
    #     print("weight matrix must be 2D")
    #     return

    # check arguments - labels
    if len(labels.shape) != 2:
        print("label matrix must be 2D")
        return

    w_size_x = 2 * r + 1
    w_size_y = 2 * r + 1
    label_nrows = labels.shape[0]
    label_ncols = labels.shape[1]
    if (((w_size_x/2) == ((w_size_x+1)/2)) or ((w_size_y/2) == ((w_size_y+1)/2))):
        print ("dimensions of weight matrix must be odd")

    # allocate result gradient
    grad_temp = np.zeros((n_ori)).tolist()
    
    # check that result is nontrivial
    if n_ori == 0:
        return grad_temp
    # to hold histograms of each slice
    slice_hist = np.zeros((2 * n_ori)).tolist()
    hist_length = int(labels.max() + 1)
    for i in range(0, 2*n_ori):
        slice_hist[i] = np.zeros((hist_length))
    # build orientation slice lookup map 
    slice_map = orientation_slice_map(w_size_x, w_size_y, n_ori)
    labels_1D = convert2OneD(labels)
    # compute histograms and histogram differences at each location
    gradients = compute_hist_gradient_2D(labels_1D, weights, slice_map, smoothing_kernel, slice_hist, n_ori, label_nrows, label_ncols, w_size_x, w_size_y)
    for no in range(0, n_ori):
        gradients[no] = convert2TwoD(gradients[no], label_nrows, label_ncols)
    return gradients

def compute_hist_gradient_2D(labels, weights, slice_map, smoothing_kernel, slice_hist, n_ori, size0_x, size0_y, size1_x, size1_y):
    """Compute histograms and histogram differences at each location."""
    # allocate result gradient
    gradients = np.zeros((n_ori)).tolist()

    for i in range(0, n_ori):
        gradients[i] = np.zeros((size0_x * size0_y))
    
    # set start position for gradient matrices
    pos_start_x = int(size1_x/2)
    pos_start_y = int(size1_y/2)
    pos_bound_y = int(pos_start_y + size0_y)
    # initialize position in result
    pos_x = int(pos_start_x)
    pos_y = int(pos_start_y)
    # compute initial range of offset_x
    if (pos_x + 1) > size0_x:
        offset_min_x = int(pos_x + 1 - size0_x)
    else:
        offset_min_x = 0
    
    if pos_x < size1_x:
        offset_max_x = int(pos_x)
    else:
        offset_max_x = int(size1_x - 1)

    ind0_start_x = int((pos_x - offset_min_x) * size0_y)
    ind1_start_x = int((offset_min_x) * size1_y)
    size = labels.size
    # determine whether to use smoothing kernel
    use_smoothing = not (smoothing_kernel.size == 0)
    temp_conv = np.zeros((slice_hist[0].shape))
    size_hist = slice_hist[0].size
    # allocate half disc histograms
    hist_left = np.zeros((slice_hist[0].shape))
    hist_right = np.zeros((slice_hist[0].shape))
    print(size)
    for n in range(0, size):
        # print(n)
        # compute range of offset_y
        if (pos_y + 1) > size0_y:
            offset_min_y = int(pos_y + 1 - size0_y)
        else:
            offset_min_y = 0
        
        if pos_y < size1_y:
            offset_max_y = int(pos_y)
        else:
            offset_max_y = int(size1_y - 1)
        
        offset_range_y = offset_max_y - offset_min_y
        # initialize indices
        ind0 = int(ind0_start_x + (pos_y - offset_min_y))
        ind1 = int(ind1_start_x + offset_min_y)

        # update histograms
        for o_x in range(offset_min_x, offset_max_x + 1):
            for o_y in range(offset_min_y, offset_max_y):
                # update histogram value 
                slice_hist[int(slice_map[ind1])][int(labels[ind0])] += weights[ind1]
                # update linear positions
                ind0 -= 1
                ind1 += 1
            # update last histogram value
            slice_hist[int(slice_map[ind1])][int(labels[ind0])] += weights[ind1]
            # update linear positions
            ind0 = ind0 + offset_range_y - size0_y
            ind1 = ind1 - offset_range_y + size1_y
        
        # smooth bins
        if use_smoothing:
            for o in range(0, 2 * n_ori):
                sh = slice_hist[o]
                temp_conv = conv_in_place_1D(sh, smoothing_kernel)
                for nh in range(0, size_hist):
                    slice_hist[o][nh] = temp_conv[nh]
        
        # L1 normalize bins
        for o in range(0, 2 * n_ori):
            sum_slice_hist = np.sum(slice_hist[o])
            if sum_slice_hist != 0:
                slice_hist[o] /= sum_slice_hist

        # compute circular gradients - initialize histograms
        hist_left.fill(0)
        hist_right.fill(0)
        for o in range(0, n_ori):
            hist_left += slice_hist[o]
            hist_right += slice_hist[o+n_ori]

        # compute circular gradients - spin the disc
        for o in range(0, n_ori):
            gradients[o][n] = X2_distance(hist_left, hist_right)
            hist_left -= slice_hist[o]
            hist_left += slice_hist[o+n_ori]
            hist_right += slice_hist[o]
            hist_right -= slice_hist[o+n_ori]
        
        # update position
        pos_y += 1
        if pos_y == pos_bound_y:
            # reset y position, increment x position
            pos_y = pos_start_y
            pos_x += 1
            # update range of offset_x
            if (pos_x + 1) > size0_x:
                offset_min_x = pos_x + 1 - size0_x
            else:
                offset_min_x = 0

            if pos_x < size1_x:
                offset_max_x = pos_x
            else:
                offset_max_x = size1_x - 1
            
            ind0_start_x = (pos_x - offset_min_x) * size0_y
            ind1_start_x = (offset_min_x) * size1_y
    return gradients

def X2_distance(m0, m1):
    dist = 0
    size = m0.size
    for n in range(0, size):
        diff = m1[n] - m0[n]
        summ = m1[n] + m0[n]
        if diff != 0:
            dist += diff * diff / summ
    return dist / 2

def conv_in_place_1D(m0, m1):
    """Compute convolution in place (for 1D matrices)"""
    # print("inside conv_in_place_1D function")
    # get size of each matrix
    size0 = m0.size
    size1 = m1.size
    # set dimensions for result matrix no larger than left input
    if (size0 > 0) and (size1 > 0):
        size = size0
    else:
        size = 0
    # set start position for result matrix no larger than left input
    pos_start = int(size1/2)
    # initialize position in result
    pos = pos_start
    m = np.zeros((m0.shape))
    # print ("before entering into loop")
    # print(size)
    # print(size0)
    # print(size1)
    for n in range(0, size):
        # compute range of offset
        if (pos + 1) > size0:
            offset_min = pos + 1 - size0
        else:
            offset_min = 0
        
        if pos < size1:
            offset_max = int(pos)
        else:
            offset_max = int(size1 - 1)
        
        # multiply and add corresponing elements
        ind0 = pos - offset_min
        ind1 = int(offset_min)
        while ind1 <= offset_max:
            # update result value
            m[n] += m0[int(ind0)] * m1[int(ind1)]
            # update linear positions
            ind0 -= 1
            ind1 += 1
        # update position
        pos += 1
    return m

def orientation_slice_map(size_x, size_y, n_ori):
    print("inside orientation_slice_map")
    # Initialize map
    slice_map = np.zeros((size_x * size_y))
    # compute orientation of each element from center
    ind = 0
    x = -(size_x) / 2
    for n_x in range(0,size_x):
        y = -(size_y) / 2
        for n_y in range(0, size_y):
            # compute orientation index
            ori = math.atan2(y, x) + math.pi
            idx = int(math.floor(ori / math.pi * float(n_ori)))
            if idx >= (2 * n_ori):
                idx = 2 * n_ori -1
            slice_map[ind] = int(idx)
            ind += 1
            y += 1
        x += 1
    print("before slicemap output")
    print(slice_map.shape)
    print(slice_map)
    return slice_map
