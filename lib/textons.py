
import math
# import imageio
import numpy as np
import scipy.signal
# import ipdb
from sklearn.cluster import KMeans
import cv2

def textons(im, border, fb, k):
    r = border
    # #find the max filter size
    # maxsz = fb[0][0].shape
    # for filter in fb:
    #     for scale in filter:
    #         maxsz = max(maxsz, scale.shape)

    # maxsz = maxsz[0]

    # #pad the image 
    # r = int(math.floor(maxsz/2))
    impad = padReflect(im,r)

	#run the filterbank on the padded image, and crop the result back
	#to the original image size
    fim = np.zeros(np.array(fb).shape).tolist()
    for i in range(np.array(fb).shape[0]):
        for j in range(np.array(fb).shape[1]):
            if fb[i][j].shape[0]<50:
                fim[i][j] = scipy.signal.convolve2d(impad, fb[i][j], 'same')
            else:
                fim[i][j] = scipy.signal.fftconvolve(impad,fb[i][j])
            fim[i][j] = fim[i][j][r:-r,r:-r]
    
    return generate_textons(fim, k)

def generate_textons(fim, k):
	d = np.product(np.array(fim).shape[:2])
	n = np.product(np.array(fim).shape[2:])
	data = np.zeros((d,n))
	count = 0
	for i in range(np.array(fim).shape[0]):
		for j in range(np.array(fim).shape[1]):
			data[count,:] = np.array(fim[i][j]).reshape(-1)
			count += 1
	kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(data.transpose())
	map = kmeans.labels_
	textons = kmeans.cluster_centers_
	w,h = np.array(fim[0][0]).shape
	map = map.reshape(w,h)

	return map, textons

def padReflect(im, border):
	impad = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_REFLECT)
	return impad

# def padReflect(im,r):
# 	impad = np.zeros(np.array(im.shape)+2*r)
# 	impad[r:-r,r:-r] = im # middle
# 	impad[:r,r:-r] = np.flipud(im[:r,:]) # top
# 	impad[-r:,r:-r] = np.flipud(im[-r:,:]); # bottom
# 	impad[r:-r,:r] = np.fliplr(im[:,:r]); # left
# 	impad[r:-r,-r:] = np.fliplr(im[:,-r:]); # right
# 	impad[:r,:r] = np.flipud(np.fliplr(im[:r,:r])); # top-left
# 	impad[:r,-r:] = np.flipud(np.fliplr(im[:r,-r:])); # top-right
# 	impad[-r:,:r] = np.flipud(np.fliplr(im[-r:,:r])); # bottom-left
# 	impad[-r:,-r:] = np.flipud(np.fliplr(im[-r:,-r:])); # bottom-right
# 	return impad
