import numpy as np
import math
# from det_mPb import det_mPb
from det_mPb import det_mPb


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

    # [mPb_nmax, mPb_nmax_rsz, bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3, textons]
    return [100, 99, 98, 90, 89, 88, 80, 79, 78, 70, 69, 68, 60, 59, 58]


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





