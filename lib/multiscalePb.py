import numpy as np
import math
import cv2
# from det_mPb import det_mPb
from det_mPb import det_mPb
from fitparab import fitparab


def multiscalePb(im, rsz=1.0):
    """compute local contour cues of an image."""
    print("inside multiscalePb")
    # default feature weights
    if im.shape[2] == 3:
        weights = [0.0146, 0.0145, 0.0163, 0.0210, 0.0243, 0.0287, 0.0166, 0.0185, 0.0204, 0.0101, 0.0111, 0.0141]
    else:
        im[:, :, 2] = im[:, :, 1]
        im[:, :, 3] = im[:, :, 1]
        weights = [0.0245, 0.0220, 0.0233, 0, 0, 0, 0, 0, 0, 0.0208, 0.0210, 0.0229]
    print(im)
    print(im.size)
    print(im.shape)
    # get gradients
    [bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3, textons] = det_mPb(im)

    # smooth cues
    gtheta = [1.5708, 1.1781, 0.7854, 0.3927, 0, 2.7489, 2.3562, 1.9635]
    filters = make_filters([3, 5, 10, 20], gtheta)
    print(filters.shape)
    print(filters)
    for o in range(0, tg1.shape[2]):
        print("after")
        bg1[:, :, o] = fitparab(bg1[:, :, o], 3, 3/4, gtheta[o], filters[0, o])
        bg2[:, :, o] = fitparab(bg2[:, :, o], 5, 5/4, gtheta[o], filters[1, o])
        bg3[:, :, o] = fitparab(bg3[:, :, o], 10, 10/4, gtheta[o], filters[2, o])

        cga1[:, :, o] = fitparab(cga1[:, :, o], 5, 5/4, gtheta[o], filters[1, o])
        cga2[:, :, o] = fitparab(cga2[:, :, o], 10, 10/4, gtheta[o], filters[2, o])
        cga3[:, :, o] = fitparab(cga3[:, :, o], 20, 20/4, gtheta[o], filters[3, o])

        cgb1[:, :, o] = fitparab(cgb1[:, :, o], 5, 5/4, gtheta[o], filters[1, o])
        cgb2[:, :, o] = fitparab(cgb2[:, :, o], 10, 10/4, gtheta[o], filters[2, o])
        cgb3[:, :, o] = fitparab(cgb3[:, :, o], 20, 20/4, gtheta[o], filters[3, o])

        tg1[:, :, o] = fitparab(tg1[:, :, o], 5, 5/4, gtheta[o], filters[1, o])
        tg2[:, :, o] = fitparab(tg2[:, :, o], 10, 10/4, gtheta[o], filters[2, o])
        tg3[:, :, o] = fitparab(tg3[:, :, o], 20, 20/4, gtheta[o], filters[3, o])

    # Computing mPb at full scale
    mPb_all = np.zeros((tg1.shape))
    for o in range(0, mPb_all.shape[2]):
        l1 = weights[0] * bg1[:, :, o]
        l2 = weights[1] * bg2[:, :, o]
        l3 = weights[2] * bg3[:, :, o]

        a1 = weights[3] * cga1[:, :, o]
        a2 = weights[4] * cga2[:, :, o]
        a3 = weights[5] * cga3[:, :, o]

        b1 = weights[6] * cgb1[:, :, o]
        b2 = weights[7] * cgb2[:, :, o]
        b3 = weights[8] * cgb3[:, :, o]

        t1 = weights[9] * tg1[:, :, o]
        t2 = weights[10] * tg2[:, :, o]
        t3 = weights[11] * tg3[:, :, o]

        mPb_all[:,:, o] = l1 + a1 + b1 + t1 + l2 + a2 + b2 + t2 + l3 + a3 + b3 + t3

    # non-maximum suppression
    mPb_nmax = nonmax_channels(mPb_all)
    mPb_nmax = np.maximum(0, np.minimum(1, 1.2 * mPb_nmax))

    # compute mPb_nmax resized if necessary
    if rsz < 1:
        # refer about resize here: https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
        mPb_all = cv2.resize(tg1, rsz)
        mPb_all[:] = 0
        for o in range(0, mPb_all.shape[2]):
            l1 = weights[0] * bg1[:, :, o]
            l2 = weights[1] * bg2[:, :, o]
            l3 = weights[2] * bg3[:, :, o]

            a1 = weights[3] * cga1[:, :, o]
            a2 = weights[4] * cga2[:, :, o]
            a3 = weights[5] * cga3[:, :, o]

            b1 = weights[6] * cgb1[:, :, o]
            b2 = weights[7] * cgb2[:, :, o]
            b3 = weights[8] * cgb3[:, :, o]

            t1 = weights[9] * tg1[:, :, o]
            t2 = weights[10] * tg2[:, :, o]
            t3 = weights[11] * tg3[:, :, o]

            mPb_all[:,:, o] = cv2.resize(l1 + a1 + b1 + t1 + l2 + a2 + b2 + t2 + l3 + a3 + b3 + t3, rsz)

        mPb_nmax_rsz = nonmax_channels(mPb_all)
        mPb_nmax_rsz = np.maximum(0, np.minimum(1, 1.2 * mPb_nmax_rsz))
    else:
        mPb_nmax_rsz = mPb_nmax

    # return [100, 99, 98, 90, 89, 88, 80, 79, 78, 70, 69, 68, 60, 59, 58]
    return [mPb_nmax, mPb_nmax_rsz, bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3, textons]


def make_filters(radii, gtheta):
    d = 2
    filters = np.zeros((len(radii), len(gtheta))).tolist()
    for r in range(0, len(radii)):
        for t in range(0, len(gtheta)):
            ra = radii[r]
            rb = ra / 4
            theta = gtheta[t]

            ra = max(1.5, ra)
            rb = max(1.5, rb)
            ira2 = 1 / ra ^ 2
            irb2 = 1 / rb ^ 2
            wr = math.floor(max(ra, rb))
            wd = 2 * wr + 1
            sint = math.sin(theta)
            cost = math.cos(theta)

            # 1. compute linear filters for coefficients
            # (a) compute inverse of least-squares problem matrix
            filt = np.zeros((wd, wd, d + 1))
            xx = np.zeros((2 * d + 1, 1))
            for u in range(-wr, wr + 1):
                for v in range(-wr, wr + 1):
                    ai = -u * sint + v * cost  # distance along major axis
                    bi = u * cost + v * sint  # distance along minor axis
                    if (ai * ai * ira2 + bi * bi * irb2) > 1:
                        continue
                    temp = ai + np.zeros((2 * d + 1, 1))
                    temp[0] = 1
                    xx = xx + np.cumprod(temp).reshape(2 * d + 1, 1)
            A = np.zeros((d + 1, d + 1))
            for i in range(0, d + 1):
                A[:, i] = xx[i:i + d + 1]

            # (b) solve least-squares problem for delta function at each pixel
            for u in range(-wr, wr + 1):
                for v in range(-wr, wr):
                    ai = -u * sint + v * cost  # distance along major axis
                    bi = u * cost + v * sint;  # distance along minor axis
                    if (ai * ai * ira2 + bi * bi * irb2) > 1:
                        continue
                    temp = ai + np.zeros(d + 1, 1)
                    temp[0] = 1
                    yy = np.cumprod(temp).reshape(d + 1, 1)
                    filt[v + wr, u + wr, :] = np.linalg.lstsq(A, yy)[0]

            filters[r, t] = filt
    return filters


def nonmax_channels(pb, nonmax_ori_tol = math.pi / 8):
    n_ori = pb.shape[2]
    oris = np.arange(0, n_ori) / n_ori * math.pi
    y, i = pb.max(2)
    i = oris[i]
    y[y<0] = 0
    nmax = nonmax_oriented_2D(y, i, nonmax_ori_tol) # y and i are 2D matrix of same shape(i.e 300x200 for example image) nonmax_ori_tol is 0.392
    return nmax

def nonmax_oriented_2D(m, m_ori, o_tol, allow_boundary = False):
    """
     * Oriented non-max suppression (2D).
     *
     * Perform non-max suppression orthogonal to the specified orientation on
     * the given 2D matrix using linear interpolation in a 3x3 neighborhood.
     *
     * A local maximum must be greater than the interpolated values of its
     * adjacent elements along the direction orthogonal to this orientation.
     *
     * Elements which have a neighbor on only one side of this direction are
     * only considered as candidates for a local maximum if the flag to allow
     * boundary candidates is set.
     *
     * The same orientation angle may be specified for all elements, or the
     * orientation may be specified for each matrix element.
     *
     * If an orientation is specified per element, then the elements themselves
     * may optionally be treated as oriented vectors by specifying a value less
     * than pi/2 for the orientation tolerance.  In this case, neighboring
     * vectors are projected along a line in the local orientation direction and
     * the lengths of the projections are used in determining local maxima.
     * When projecting, the orientation tolerance is subtracted from the true
     * angle between the vector and the line (with a result less than zero
     * causing the length of the projection to be the length of the vector).
     *
     * Non-max elements are assigned a value of zero.
     *
     * NOTE: The original matrix must be nonnegative.
    :return:
    """
    # check arguments - matrix size
    if m.ndim != 2:
        exit("matrix=> m must be 2D")
    if m.shape != m_ori.shape:
        exit("m and m_roi should be of same shape")
    # check arguments - orientation tolerance
    if o_tol < 0:
        exit("orientation tolerance must be nonnegative")
    # Get m size
    size_x = m.shape[0]
    size_y = m.shape[1]
    # Initialize result matrix
    nmax = np.zeros((m.shape))
    # perform oriented non-max suppression at each element
    n = 0
    for x in range(0, size_x):
        for y in range(0, size_y):
            # compute direction (in [0,pi)) along which to suppress
            ori = float(m_ori[getXIndex(n, size_y)][getYIndex(n, size_y)])
            theta = float(ori + (math.pi / 2))
            theta -= math.floor(theta/math.pi) * math.pi
            # check nonnegativity
            v = float(m[getXIndex(n, size_y)][getYIndex(n, size_y)])
            if v < 0:
                exit("m must be nonnegative")
            # initialize indices of values in local neighborhood
            ind0a = 0
            ind0b = 0
            ind1a = 0
            ind1b = 0
            # initialize distance weighting
            d = 0
            # initialize boundary flags
            valid0 = False
            valid1 = False
            # compute interpolation indicies
            if theta == 0:
                valid0 = (x > 0)
                valid1 = (x < (size_x - 1))
                if (valid0):
                    ind0a = n-size_y
                    ind0b = ind0a
                if (valid1):
                    ind1a = n+size_y
                    ind1b = ind1a
            elif theta < (math.pi / 4):
                d = math.tan(theta)
                valid0 = ((x > 0) and (y > 0))
                valid1 = ((x < (size_x - 1)) and (y < (size_y - 1)))
                if (valid0):
                    ind0a = n-size_y
                    ind0b = ind0a-1
                if (valid1):
                    ind1a = n+size_y
                    ind1b = ind1a+1
            elif theta < (math.pi / 2):
                d = math.tan((math.pi / 2) - theta)
                valid0 = ((x > 0) and (y > 0))
                valid1 = ((x < (size_x - 1)) and (y < (size_y - 1)))
                if (valid0):
                    ind0a = n-1
                    ind0b = ind0a-size_y
                if (valid1):
                    ind1a = n+1
                    ind1b = ind1a+size_y
            elif theta == (math.pi / 2):
                valid0 = (y > 0)
                valid1 = (y < (size_y - 1))
                if valid0:
                    ind0a = n-1
                    ind0b = ind0a
                if valid1:
                    ind1a = n+1
                    ind1b = ind1a
            elif (theta < (3.0 * math.pi / 4)):
                d = math.tan(theta - (math.pi / 2))
                valid0 = ((x < (size_x-1)) and (y > 0))
                valid1 = ((x > 0) and (y < (size_y-1)))
                if (valid0):
                    ind0a = n-1
                    ind0b = ind0a + size_y
                if (valid1):
                    ind1a = n+1
                    ind1b = ind1a-size_y
            else: #  (theta < pi)
                d = math.tan((math.pi) - theta)
                valid0 = ((x < (size_x - 1)) and (y > 0))
                valid1 = ((x > 0) and (y < (size_y - 1)))
                if (valid0):
                    ind0a = n+size_y
                    ind0b = ind0a-1
                if (valid1):
                    ind1a = n-size_y
                    ind1b = ind1a+1
            # check boundary conditions
            if allow_boundary or (valid0 and valid1):
                # initialize values in local neighborhood
                v0a = 0; v0b = 0; v1a = 0; v1b = 0
                # initialize orientations in local neighborhood
                ori0a = 0; ori0b = 0; ori1a = 0; ori1b = 0
                # grab values and orientations
                if valid0:
                    v0a = float(m[getXIndex(ind0a, size_y)][getYIndex(ind0a, size_y)])
                    v0b = float(m[getXIndex(ind0b, size_y)][getYIndex(ind0b, size_y)])
                    ori0a = float(m_ori[getXIndex(ind0a, size_y)][getYIndex(ind0a, size_y)]) - ori
                    ori0b = float(m_ori[getXIndex(ind0b, size_y)][getYIndex(ind0b, size_y)]) - ori
                if  valid1:
                    v1a = float(m[getXIndex(ind1a, size_y)][getYIndex(ind1a, size_y)])
                    v1b = float(m[getXIndex(ind1b, size_y)][getYIndex(ind1b, size_y)])
                    ori1a = float(m_ori[getXIndex(ind1a, size_y)][getYIndex(ind1a, size_y)]) - ori
                    ori1b = float(m_ori[getXIndex(ind1b, size_y)][getYIndex(ind1b, size_y)]) - ori
                # place orientation difference in [0, pi/2) range
                ori0a -= math.floor(ori0a/(2*math.pi)) * (2*math.pi)
                ori0b -= math.floor(ori0b / (2 * math.pi)) * (2 * math.pi)
                ori1a -= math.floor(ori1a / (2 * math.pi)) * (2 * math.pi)
                ori1b -= math.floor(ori1b / (2 * math.pi)) * (2 * math.pi)
                if (ori0a >= math.pi): ori0a = 2 * math.pi - ori0a
                if (ori0b >= math.pi): ori0b = 2 * math.pi - ori0b
                if (ori1a >= math.pi): ori1a = 2 * math.pi - ori1a
                if (ori1b >= math.pi): ori1b = 2 * math.pi - ori1b
                if (ori0a >= (math.pi/2)): ori0a = math.pi - ori0a
                if (ori0b >= (math.pi/2)): ori0b = math.pi - ori0b
                if (ori1a >= (math.pi/2)): ori1a = math.pi - ori1a
                if (ori1b >= (math.pi/2)): ori1b = math.pi - ori1b
                # correct orientation difference by tolerance
                ori0a = 0 if (ori0a <= o_tol) else (ori0a - o_tol)
                ori0b = 0 if (ori0b <= o_tol) else (ori0b - o_tol)
                ori1a = 0 if (ori1a <= o_tol) else (ori1a - o_tol)
                ori1b = 0 if (ori1b <= o_tol) else (ori1b - o_tol)

                # interpolate
                v0 = float((1.0-d)*v0a*math.cos(ori0a) + d*v0b*math.cos(ori0b))
                v1 = float((1.0-d)*v1a*math.cos(ori1a) + d*v1b*math.cos(ori1b))

                # suppress non-max
                if (v > v0) and (v > v1):
                    nmax[getXIndex(n, size_y)][getYIndex(n, size_y)] = v
            # increment linear coordinate
            n += 1
    return nmax

# This function is used to convert the vector index to row index of 2D array.
# Function assumes that x, y, n are zero indexed. vector, 2D array are zero indexed
def getXIndex(n, n_cols):
    """Returns row index of 2D array given vector index n"""
    return int(int(n)/int(n_cols))

# This function is used to convert the vector index to column index of 2D array.
# Function assumes that x, y, n are zero indexed. vector, 2D array are zero indexed
def getYIndex(n, n_cols):
    """Returns column index of 2D array given vector index n"""
    return int(int(n) % int(n_cols))

# This function is used to convert the 2D array indexs to a vector index(n).
# Function assumes that x, y, n are zero indexed. vector, 2D array are zero indexed
def getNIndex(x, y, no_cols):
    """Returns vector index n, given the row(x), column(y) index of 2D array"""
    return int(int(int(x) * int(no_cols)) + int(y))
