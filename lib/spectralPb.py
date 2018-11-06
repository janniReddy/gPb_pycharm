import numpy as np

def spectralPb(mPb, orig_sz, outFile = '', nvec = 17):
    """global contour cue from local mPb."""
    tx, ty = mPb.shape
    l = np.zeros((2)).tolist()
    l[0] = np.zeros((mPb.shape[0]+1, mPb.shape[1]))
    l[0][1:, :] = mPb
    l[1] = np.zeros((mPb.shape[0], mPb.shape[1]+1))
    l[1][:, 1:] = mPb

    # build the pairwise affinity matrix
    val, I, J = buildW(l[0], l[1])
    W =
