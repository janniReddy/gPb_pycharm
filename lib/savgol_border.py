import numpy as np
import math

import lib_image

def savgol_border(a, zi, ra, rb, theta):
    h = a.shape[0]
    w = a.shape[1]
    a_out = np.zeros((h,w))
    a_in = convert2OneD(a)
    z = convert2OneD(zi)
    ra = max(1.5, ra)
    rb = max(1.5, rb)
    ira2 = 1 / pow(ra, 2)
    irb2 = 1 / pow(rb, 2)
    wr = int (math.floor(max(ra, rb)))
    sint = math.sin(theta)
    cost = math.cos(theta)
    eps = math.exp(-300)
    for cp in range(0, w*h):
        y = int(cp %h)
        x = int(cp/h)
        if ((x>=wr) and (x<(w-wr)) and (y>=wr) and (y<(h-wr))):
            a_out[cp] = a_in[cp]
        else:
            d1 = d2 = d3 = d4 = d0 = v0 = v1 = v2 = 0.0
            for u in range(-wr, wr+1):
                xi = int(x +u)
                if ((xi<0) or (xi>=w)):
                    continue
                for v in range(-wr, wr+1):
                    yi = int(y + v)
                    if ((yi<0) or (yi>=h)):
                        continue
                    di = -u*sint + v*cost
                    ei = u*cost + v*sint
                    if ( (di*di*ira2 + ei*ei*irb2) > 1):
                        continue
                    cpi = int(yi+xi*h)
                    zi = z[cpi]
                    di2 = di*di
                    d0 = d0 + 1
                    d1 = d1 + di
                    d2 = d2 + di2
                    d3 = d3 + di*di2
                    d4 = d4 + di2*di2
                    v0 = v0 + zi
                    v1 = v1 + zi*di
                    v2 = v2 + zi*di2
            detA = -d2*d2*d2 + 2*d1*d2*d3 - d0*d3*d3 - d1*d1*d4 + d0*d2*d4
            if detA > eps:
                a_out = ((-d3*d3+d2*d4)*v0 + (d2*d3-d1*d4)*v1 + (-d2*d2+d1*d3)*v2)/ detA

    return a_out




